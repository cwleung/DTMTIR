import torch.nn as nn
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.models.gplvm.latent_variable import *
import math

from model.gplvm import bGPLVM

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_topics, eta_size, hidden, dropout, batchNorm):
        super().__init__()
        if torch.cuda.is_available():
            self.cuda()

        self.num_topics = num_topics
        self.eta_size = eta_size
        self.batchNorm = batchNorm
        self.drop = nn.Dropout(dropout)  # dropout
        self.fc1 = nn.Linear(vocab_size + eta_size, hidden)
        self.fc2 = nn.Linear(hidden + eta_size, hidden)
        self.fcmu = nn.Linear(hidden + eta_size, num_topics, bias=True)  # fully-connected layer output mu
        self.fclv = nn.Linear(hidden + eta_size, num_topics, bias=True)  # fully-connected layer output sigma
        self.act1 = nn.Softplus()
        self.act2 = nn.Softplus()

        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)

        self.bnmu = nn.BatchNorm1d(hidden, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(hidden, affine=False)  # to avoid component collapse

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """
        Gaussian KL Divergence
        Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma).to(device)
            sigma_p_sq = torch.exp(p_logsigma).to(device)
            kl = (sigma_q_sq + (q_mu - p_mu) ** 2) / (sigma_p_sq + 1e-6)
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1).to(device)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1).to(device)
        return kl

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        eps = torch.randn_like(std).to(device)
        return eps.mul_(std).add_(mu)

    def forward(self, bow, eta_lstm_td, eta_td, eva=False):
        bow = torch.cat([bow, eta_lstm_td], dim=1).to(device)
        h = self.act1(self.fc1(bow))
        h = self.norm1(h)  # layernorm
        bow = torch.cat([h, eta_lstm_td], dim=1).to(device)
        h = self.act2(self.fc2(bow))
        h = self.norm2(h)  # layernorm
        h = self.drop(h)
        h = torch.cat([h, eta_lstm_td], dim=1).to(device)
        # μ and Σ are two inference networks
        mu_theta = self.fcmu(h)
        sig_theta = self.fclv(h)
        if eva:
            return mu_theta.softmax(-1)
        z = self.reparameterize(mu_theta, sig_theta)
        theta = torch.softmax(z, -1).to(device)
        kld_theta = self.get_kl(mu_theta, sig_theta, eta_td, 0.005 * torch.randn(self.num_topics).to(device))
        return theta, kld_theta


class Decoder(nn.Module):
    # Pre-trained embedding, alpha, rho, embedding method
    # Need to be refactored with Topic Embedding
    def __init__(self, vocab_size, num_topics, num_times, dropout,
                 useEmbedding=True, rho_size=256, delta=0.005):
        super().__init__()
        # this beta can be refactorized in to BOW, a neural network that to be trained
        self.delta = delta
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.num_times = num_times
        self.useEmbedding = useEmbedding
        self.rho_size = rho_size
        # Changes - Beta
        # < Linear NN (K->V)
        # > Embedding -> Linear NN (K->E->V)
        if self.useEmbedding:
            # Call ρ Topic Embedding
            self.fcrho = nn.Linear(rho_size, vocab_size, bias=False)
            self.bnrho = nn.BatchNorm1d(rho_size, affine=False)
            ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
            self.mu_q_alpha = nn.Parameter(torch.randn(num_topics, num_times, rho_size).to(device))
            self.logsigma_q_alpha = nn.Parameter(torch.randn(num_topics, num_times, rho_size).to(device))
        else:
            # beta
            self.mu_q_beta = nn.Parameter(torch.randn(num_topics, num_times, vocab_size).to(device))
            self.logsig_q_beta = nn.Parameter(torch.randn(num_topics, num_times, vocab_size).to(device))

        if torch.cuda.is_available():
            self.cuda()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        eps = torch.randn_like(std).to(device)
        return eps.mul_(std).add_(mu)

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """ Gaussian KL Divergence
        Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma).to(device)
            sigma_p_sq = torch.exp(p_logsigma).to(device)
            kl = (sigma_q_sq + (q_mu - p_mu) ** 2) / (sigma_p_sq + 1e-6)
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1).to(device)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1).to(device)
        return kl

    def get_alpha_ssm(self):
        alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(device)
        kl_alpha = []
        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])
        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.num_times):
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :])
            p_mu_t = alphas[t - 1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(device))
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).sum()
        assert (alphas.shape == torch.Size([self.num_times, self.num_topics, self.rho_size]))
        return alphas, kl_alpha.sum()

    def get_beta_ssm(self):
        beta = torch.zeros(self.num_times, self.num_topics, self.vocab_size).to(device)
        kl_beta = []
        beta[0] = self.reparameterize(self.mu_q_beta[:, 0, :], self.logsig_q_beta[:, 0, :])
        pu_mu_0 = torch.zeros(self.num_topics, self.vocab_size).to(device)
        logsig_p_0 = torch.zeros(self.num_topics, self.vocab_size).to(device)
        kl_0 = self.get_kl(self.mu_q_beta[:, 0, :], self.logsig_q_beta[:, 0, :], pu_mu_0, logsig_p_0)
        kl_beta.append(kl_0)
        for t in range(1, self.num_times):
            beta[t] = self.reparameterize(self.mu_q_beta[:, t, :], self.logsig_q_beta[:, t, :])
            p_mu_t = beta[t - 1]
            logsig_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.vocab_size).to(device))
            kl_t = self.get_kl(self.mu_q_beta[:, t, :], self.logsig_q_beta[:, t, :], p_mu_t, logsig_p_t)
            kl_beta.append(kl_t)
        kl_beta = torch.stack(kl_beta).sum()

        assert (beta.shape == torch.Size([self.num_times, self.num_topics, self.vocab_size]))
        return beta.softmax(-1), kl_beta.sum()

    def get_beta(self):
        # \rho^T\alpha
        if self.useEmbedding:
            alpha, kl_alpha = self.get_alpha_ssm()
            beta = self.fcrho(alpha.reshape(alpha.size(0) * alpha.size(1), self.rho_size))
            beta = beta.reshape(alpha.size(0), alpha.size(1), -1)
            # assert dimension
            assert (beta.shape == torch.Size([self.num_times, self.num_topics, self.vocab_size])), beta.shape
            # n_alpha = alpha.reshape(alpha.size(0) * alpha.size(1), self.rho_size)
            # rho = self.fcrho.rho.weight.T
            # logit = torch.mm(n_alpha, rho)
            # logit = logit.reshape(alpha.size(0), alpha.size(1), -1)
            # beta = F.softmax(logit, dim=-1)
            # beta = self.bnalpha(self.fcalpha(self.bnrho(self.fcrho.weight))).transpose(1, 0).to(device)
            return beta.softmax(-1), kl_alpha
        # \beta
        else:
            return self.get_beta_ssm()


class DTMTIR(nn.Module):

    def __init__(self, vocab_size, num_topics, num_times, hidden, dropout,
                 delta, data_size, useEmbedding=False, eta_size=256, rho_size=256, batchNorm=True):
        super(DTMTIR, self).__init__()

        if torch.cuda.is_available():
            self.cuda()

        self.delta = delta
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.num_times = num_times
        self.useEmbedding = useEmbedding
        self.rho_size = rho_size
        self.eta_size = eta_size
        self.data_size = data_size

        self.encoder = Encoder(vocab_size, num_topics, eta_size, hidden, dropout, batchNorm).to(device)
        self.decoder = Decoder(vocab_size, num_topics, num_times, dropout,
                               useEmbedding, rho_size, delta).to(device)

        self.gru = nn.GRU(num_topics, eta_size, batch_first=True, num_layers=2, bidirectional=False).to(device)

        self.gplvm = bGPLVM(self.data_size, vocab_size, self.num_topics, 100).to(device)
        self.likelihood = GaussianLikelihood(batch_shape=(num_times, vocab_size)).to(device)

    def get_beta(self):
        return self.decoder.get_beta()

    def get_theta(self, bows, eta, times):
        # two-layers of lstm model
        eta_lstm, _ = self.gru(eta)
        assert (eta_lstm.shape == torch.Size([self.num_times, self.eta_size]))
        # THETA N(η,α^2I)
        eta_lstm_td = eta_lstm[times.type('torch.LongTensor')]
        eta_gp_td = eta[times.type('torch.LongTensor')]
        theta, kld_theta = self.encoder(bows, eta_lstm_td, eta_gp_td)
        return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

    def get_mu(self, rnn_inp):
        mll = VariationalELBO(self.likelihood, self.gplvm, num_data=len(rnn_inp))  # ,combine_terms=False)
        sample = self.gplvm.sample_latent_variable()
        output = self.gplvm(sample)
        loss = -mll(output, rnn_inp.T.to(device))
        # nll + kl_x + kl_u
        return self.gplvm.X.q_mu, loss.sum()

    def forward(self, bows, times, rnn_inp, num_docs):
        assert (rnn_inp.shape == torch.Size([self.num_times, self.vocab_size]))
        bsize = bows.size(0)
        norm_coeff = num_docs / bsize
        ## ETA
        eta_gp, kld_eta_gp = self.get_mu(rnn_inp)
        eta_gp = F.gumbel_softmax(eta_gp, tau=0.5, hard=False)
        assert (eta_gp.shape == torch.Size([self.num_times, self.num_topics]))
        theta, kld_theta = self.get_theta(bows, eta_gp, times)
        kld_theta = kld_theta.sum() * norm_coeff
        # BETA
        beta, kl_beta = self.get_beta()
        beta = beta[times.type('torch.LongTensor')]
        assert (beta.shape == torch.Size(
            [bows.shape[0], self.num_topics, self.vocab_size]
        ))
        theta = theta.unsqueeze(1)
        assert (theta.shape == torch.Size([bows.shape[0], 1, self.num_topics]))
        ## PRED
        pred = torch.bmm(theta, beta).squeeze(1).nan_to_num()
        logp = torch.log(pred + 1e-6).nan_to_num()
        nll = -(logp * bows).sum(-1)
        nll = nll.sum() * norm_coeff
        return nll, kl_beta, kld_theta, kld_eta_gp

    def get_eta_result(self):
        return self.gplvm.X.q_mu, self.gplvm.X.q_log_sigma

    def predict(self, bow, norm_bow, times):
        """give out the test data set, return the corresponding perplexity"""
        self.eval()
        with torch.no_grad():
            # get eta(TxK)
            eta_gp = self.gplvm.X.q_mu
            assert (eta_gp.shape == torch.Size([self.num_times, self.num_topics]))
            # get theta(DxK)
            eta_lstm, _ = self.gru(eta_gp)
            eta_lstm_td = eta_lstm[times.type('torch.LongTensor')]
            eta_td = eta_gp[times.type('torch.LongTensor')]
            theta = self.encoder(bow, eta_lstm_td, eta_td, eva=True)
            print(theta.shape)
            assert (theta.shape == torch.Size([norm_bow.shape[0], self.num_topics]))
            beta, kl_beta = self.get_beta()
            beta = beta[times.type('torch.LongTensor')]
            assert (beta.shape == torch.Size(
                [bow.shape[0], self.num_topics, self.vocab_size]
            ))
            loglik = theta.unsqueeze(2) * beta
            pred = loglik.sum(1)
            logp = torch.log(pred)
            sums = bow.sum(1).unsqueeze(1)
            loss = (-logp * bow).sum(-1) / sums.squeeze()
            loss = loss.nan_to_num().mean().item()
            # ppl check when doing mini-batch
            ppl = round(math.exp(loss), 1)
            return ppl


if __name__ == '__main__':
    # smoke test
    time = 20
    batch_size = 32
    vocab_size = 200
    num_topics = 30

    model = DTMTIR(vocab_size=vocab_size,
                   num_topics=num_topics,
                   num_times=time,
                   hidden=800,
                   dropout=0.0,
                   delta=0.005,
                   useEmbedding=True,
                   rho_size=300,
                   eta_size=128,
                   data_size=time
                   )

    batch_docs = torch.randn(batch_size, vocab_size)
    time_batch = torch.randint(time, size=(batch_size,))
    train_rnn_inp = torch.randn((time, vocab_size))

    print('Test train')
    model(batch_docs, time_batch, train_rnn_inp, batch_size)
    print('Test predict')
    model.predict(batch_docs, batch_docs, time_batch)
