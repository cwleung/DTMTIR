import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.models.gplvm.latent_variable import *
import math

from model.gplvm import bGPLVM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout, batchNorm):
        super().__init__()
        if torch.cuda.is_available():
            self.cuda()

        self.num_topics = num_topics
        self.batchNorm = batchNorm
        self.drop = nn.Dropout(dropout)  # dropout
        self.fc1 = nn.Linear(vocab_size + num_topics, hidden)
        self.fc2 = nn.Linear(hidden + num_topics, hidden)
        self.fcmu = nn.Linear(hidden + num_topics, num_topics, bias=True)  # fully-connected layer output mu
        self.fclv = nn.Linear(hidden + num_topics, num_topics, bias=True)  # fully-connected layer output sigma
        self.act1 = nn.Softplus()
        self.act2 = nn.Softplus()

        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)

        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        eps = torch.randn_like(std).to(device)
        return eps.mul_(std).add_(mu)

    def forward(self, inp, res_inp, eva=False):
        inp = torch.cat([inp, res_inp], dim=1).to(device)
        h = self.act1(self.fc1(inp))
        h = self.norm1(h)  # layernorm
        inp = torch.cat([h, res_inp], dim=1).to(device)
        h = self.act2(self.fc2(inp))
        h = self.norm2(h)  # layernorm
        h = self.drop(h)
        h = torch.cat([h, res_inp], dim=1).to(device)
        # μ and Σ are two inference networks
        mu_theta = self.fcmu(h)
        sig_theta = self.fclv(h)
        if eva:
            return mu_theta.softmax(-1)
        z = self.reparameterize(mu_theta, sig_theta)
        theta = torch.softmax(z, -1).to(device)
        kld_theta = self.get_kl(mu_theta, sig_theta, res_inp, 0.005 * torch.randn(self.num_topics).to(device))
        return theta, kld_theta


class Decoder(nn.Module):
    # Pre-trained embedding, alpha, rho, embedding method
    # Need to be refactored with Topic Embedding
    def __init__(self, vocab_size, num_topics, num_times, dropout,
                 useEmbedding=True, rho_size=256, pre_embedding=None, emb_type='NN',
                 trainEmbedding=True):
        super().__init__()
        # this beta can be refactorized in to BOW, a neural network that to be trained
        self.emb_type = emb_type
        self.trainEmbedding = trainEmbedding
        self.useEmbedding = useEmbedding
        self.rho_size = rho_size
        # Changes - Beta
        # < Linear NN (K->V)
        # > Embedding -> Linear NN (K->E->V)
        self.drop = nn.Dropout(dropout)
        if self.useEmbedding:
            # Call ρ Topic Embedding
            if trainEmbedding:
                self.fcrho = TopicEmbedding(rho_size, vocab_size, pre_embedding,
                                            emb_type, dropout)
                self.bnrho = nn.BatchNorm1d(rho_size, affine=False)
            # use original embedding
            else:
                self.fcrho = TopicEmbedding(rho_size, vocab_size, pre_embedding,
                                            emb_type, dropout, trans_layer, trans_head, trans_dim)
            # Call α
            self.fcalpha = nn.Linear(rho_size, num_topics, bias=False)
            self.bnalpha = nn.BatchNorm1d(num_topics, affine=False)
            # nn.Parameter(torch.randn(rho_size, num_topics))
        else:
            # Call β, Use Original NN (K->V)
            self.fcbeta = nn.Parameter(torch.randn(num_times, num_topics, vocab_size).to(device))
            # self.bnbeta = nn.BatchNorm1d(vocab_size, affine=False)

        self.bn = nn.BatchNorm1d(vocab_size, affine=False)

        if torch.cuda.is_available():
            self.cuda()

    def get_beta(self, alpha):
        if self.trainEmbedding:
            if self.emb_type is 'NN':
                #                 logit = self.fcrho(alpha.reshape(alpha.size(0) * alpha.size(1), self.rho_size))
                #                 logit = logit.reshape(alpha.size(0), alpha.size(1), -1)
                #                 beta = F.softmax(logit, dim=-1)
                n_alpha = alpha.reshape(alpha.size(0) * alpha.size(1), self.rho_size)
                rho = self.fcrho.rho.weight.T
                logit = torch.mm(n_alpha, rho)
                logit = logit.reshape(alpha.size(0), alpha.size(1), -1)
                beta = F.softmax(logit, dim=-1)
            else:
                raise ValueError('Wrong embedding type')
        elif self.useEmbedding:
            beta = self.bnalpha(self.fcalpha(self.bnrho(self.fcrho.weight()))).transpose(1, 0).to(device)
        else:
            beta = self.fcbeta.weight
        return beta


class TopicEmbedding(nn.Module):
    def __init__(self, rho_size, vocab_size, pre_embedding=None,
                 emb_type='NN', dropout=0.0,
                 n_heads=8, n_layer=4, n_dim=128, n_code=8):
        super().__init__()
        self.emb_type = emb_type
        if pre_embedding is None:
            # 1. Embedding layer
            if emb_type is 'NN':
                self.rho = nn.Linear(rho_size, vocab_size, bias=False)
            else:
                raise ValueError('Wrong Embedding Type')
        else:
            self.rho = pre_embedding.clone().float().to(device)

    def forward(self, inputs):
        if self.emb_type is 'NN':
            return self.rho(inputs)
        else:
            raise ValueError('Wrong Embedding Type')


class DTMTIR(nn.Module):

    def __init__(self, vocab_size, num_topics, num_times, hidden, dropout,
                 delta, data_size, useEmbedding=False, eta_size=256, rho_size=256,
                 pre_embedding=None, emb_type='NN', trainEmbedding=False, batchNorm=True):
        super().__init__()

        self.delta = delta
        if torch.cuda.is_available():
            self.cuda()

        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.num_times = num_times
        self.useEmbedding = useEmbedding
        self.trainEmbedding = trainEmbedding
        self.emb_type = emb_type
        self.rho_size = rho_size
        self.eta_size = eta_size

        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout, batchNorm).to(device)
        self.decoder = Decoder(vocab_size, num_topics, num_times, dropout,
                               useEmbedding, rho_size, pre_embedding, emb_type,
                               trainEmbedding).to(device)
        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(num_topics, num_times, rho_size).to(device))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(num_topics, num_times, rho_size).to(device))

        self.data_size = data_size
        self.gplvm = bGPLVM(self.data_size, vocab_size, self.num_topics, 100).to(device)
        self.likelihood = GaussianLikelihood(batch_shape=(num_times, vocab_size)).to(device)

        self.mu_q_eta = nn.Linear(self.num_topics, self.num_topics, bias=True).to(device)
        self.logsigma_q_eta = nn.Linear(self.num_topics, self.num_topics, bias=True).to(device)

        self.eta = nn.Parameter(torch.randn(self.num_times, self.num_topics).to(device))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        eps = torch.randn_like(std).to(device)
        return eps.mul_(std).add_(mu)

    def get_beta(self, alpha):
        beta = self.decoder.get_beta(alpha)
        return beta

    def get_theta(self, eta, bows, times):
        eta_td = eta[times.type('torch.LongTensor')]
        theta, kl_theta = self.encoder(bows, eta_td)
        return theta, kl_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

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

    def get_alpha(self):  ## mean field
        # TxKxL
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
        return alphas, kl_alpha.sum()

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))

    def get_mu(self, rnn_inp):
        mll = VariationalELBO(self.likelihood, self.gplvm, num_data=len(rnn_inp))  # ,combine_terms=False)
        sample = self.gplvm.sample_latent_variable()
        output = self.gplvm(sample)
        loss = mll(output, rnn_inp.T.to(device))
        # nll + kl_x + kl_u
        return self.gplvm.X.q_mu, loss  # loss[0].sum()+loss[1].sum()+loss[2].sum()+loss[3].sum()

    def forward(self, bows, norm_bows, times, rnn_inp, num_docs):
        bsz = bows.size(0)
        coeff = num_docs / bsz
        # eta, kl_eta = self.get_eta(rnn_inp)
        eta_gp, kld_eta_gp = self.get_mu(rnn_inp)
        kld_eta = torch.zeros(()).to(device)
        # eta, kld_eta = self.get_eta(eta_gp.to(device))
        # get theta N(η,α^2I)
        theta, kld_theta = self.get_theta(self.eta, norm_bows, times)
        kld_theta = kld_theta.sum() * coeff
        alpha, kl_alpha = self.get_alpha()
        assert (alpha.shape == torch.Size(
            [self.num_times, self.num_topics, self.rho_size]
        ))
        beta = self.get_beta(alpha)
        beta = beta[times.type('torch.LongTensor')]

        assert (beta.shape == torch.Size(
            [bows.shape[0], self.num_topics, self.vocab_size]
        ))
        theta = theta.unsqueeze(1)
        assert (theta.shape == torch.Size([bows.shape[0], 1, self.num_topics]
                                          ))
        pred = torch.bmm(theta, beta).squeeze(1).nan_to_num()
        logp = torch.log(pred + 1e-6).nan_to_num()
        nll = -(logp * bows).sum(-1)
        nll = nll.sum() * coeff
        return nll, kl_alpha, kld_theta, kld_eta_gp

    def get_beta_result(self):
        alpha = self.mu_q_alpha.clone().contiguous()
        alpha = alpha.permute(1, 0, 2)
        beta = self.get_beta(alpha, torch.arange(0, ts.unique().shape[0]))
        return beta

    def get_eta_result(self):
        return self.gplvm.X.q_mu

    def predict(self, d_bat, norm_d_bat, t_bat, rnn_inp):
        """give out the test data set, return the corresponding perplexity"""
        self.eval()
        with torch.no_grad():
            # get eta(TxK)
            # eta = self.gplvm.X.q_mu
            eta = self.eta
            # eta, kl_eta = self.get_eta(rnn_inp)
            assert (eta.shape == torch.Size([self.num_times, self.num_topics]))
            # get theta(DxK)
            eta_td = eta[t_bat.type('torch.LongTensor')]
            theta = self.encoder(d_bat, eta_td, eva=True)
            # theta = self.get_theta(eta, norm_d_bat, t_bat)[0]
            assert (theta.shape == torch.Size([norm_d_bat.shape[0], self.num_topics]))
            # theta = theta.unsqueeze(1)
            # get alpha(KxTxL)
            alpha = self.mu_q_alpha.clone().contiguous()
            # alpha = alpha.permute(1, 0, 2)
            assert (alpha.shape == torch.Size(
                [self.num_topics, self.num_times, self.rho_size]
            ))
            # alpha_td(KxDxL)
            # alpha_td = alpha[:,t_bat.type('torch.LongTensor'), :]
            #             assert (alpha_td.shape == torch.Size(
            #                 [self.num_topics, d_bat.shape[0], self.rho_size]
            #             ))
            # get beta(T[D]xKxV)
            beta = self.get_beta(alpha)
            if self.trainEmbedding or self.useEmbedding:
                beta = beta.permute(1, 0, 2)
            beta = beta[t_bat.type('torch.LongTensor')]
            assert (beta.shape == torch.Size(
                [d_bat.shape[0], self.num_topics, self.vocab_size]
            ))
            loglik = theta.unsqueeze(2) * beta
            pred = loglik.sum(1)
            # pred = torch.bmm(theta, beta).squeeze(1)
            logp = torch.log(pred)
            sums = d_bat.sum(1).unsqueeze(1)
            loss = (-logp * d_bat).sum(-1) / sums.squeeze()
            loss = loss.nan_to_num().mean().item()
            # ppl check when doing mini-batch
            ppl = round(math.exp(loss), 1)
            return ppl
