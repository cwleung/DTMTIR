import argparse
from doctest import set_unittest_reportflags
import os
import pickle
import torch
import math
import numpy as np
import scipy.io
import torch
from tqdm import trange

from model.dtmtir import DTMTIR
from utils.data import get_batch, get_data, get_rnn_input
from utils.util import get_rnn_inp, get_topic_coherence, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Dynamic Topic Model with Temporal Information Regularizer")
# Environment setting
parser.add_argument("-st", "--smoke-test", default=False, type=bool)
parser.add_argument("-s", "--seed", default=2022, type=int)
# Topic model-related
parser.add_argument("-nt", "--num-topics", default=20, type=int)
# NN-related
parser.add_argument("-hl", "--inf-hidden", default=800, type=int)
parser.add_argument("-dr", "--dropout", default=0.0, type=float)
parser.add_argument("-af", "--batch-norm", default=True, type=bool)
parser.add_argument("-temb", "--train-embedding", default=False, type=bool)
parser.add_argument("-uemb", "--use-embedding", default=False, type=bool)
# Dataset-related
# parser.add_argument("-ds", "--dataset", default="20newsgroups", type=str)
parser.add_argument("-nd", "--min-df", default=50, type=int)
parser.add_argument("-xd", "--max-df", default=0.7, type=float)
parser.add_argument("-bz", "--batch-size", default=256, type=int)
parser.add_argument("-tr", "--test-ratio", default=0.8, type=int)
# Adam related
parser.add_argument("-lr", "--learning-rate", default=5e-3, type=float)
parser.add_argument("-wd", "--weight-decay", default=1e-6, type=float)
#
parser.add_argument("-emb", "--embedding", default="NN", type=str)
parser.add_argument("-d", "--delta", default=0.005, type=float)
parser.add_argument("-rho", "--rho-size", default=300, type=int)
parser.add_argument("-eta", "--eta-size", default=128, type=int)

parser.add_argument("-e", "--epoch", default=20, type=int)

parser.add_argument("-l", "--lambda_", default=1, type=float)
parser.add_argument("-es", "--early-stopping", default=True, type=bool)

args = parser.parse_args()

torch.set_default_tensor_type('torch.FloatTensor')
# set seed
set_seed(args.seed)

print('Getting vocabulary ...')
data_file = os.path.join('./', 'min_df_{}'.format(100))
vocab, train, valid, test = get_data(data_file, temporal=True)
vocab_size = len(vocab)
# 1. training data
print('Getting training data ...')
train_tokens = train['tokens']
train_counts = train['counts']
train_times = train['times']
# 2. valid set
print('Getting validation data ...')
valid_tokens = valid['tokens']
valid_counts = valid['counts']
valid_times = valid['times']
# 3. test data
print('Getting testing data ...')
test_tokens = test['tokens']
test_counts = test['counts']
test_times = test['times']

num_times = train_times.max() - train_times.min() + 1
train_size = len(train_tokens)
valid_size = len(valid_tokens)
test_size = len(test_tokens)

test_rnn_inp = get_rnn_input(test_tokens, test_counts, test_times, num_times, vocab_size, test_size).nan_to_num()
valid_rnn_inp = get_rnn_input(valid_tokens, valid_counts, valid_times, num_times, vocab_size, valid_size).nan_to_num()
train_rnn_inp = get_rnn_input(train_tokens, train_counts, train_times, num_times, vocab_size, train_size).nan_to_num()

train_cvz, train_ts = get_batch(train_tokens, train_counts, torch.tensor(range(train_size)).to(device), vocab_size, temporal=True,times=train_times)
valid_cvz, valid_ts = get_batch(valid_tokens, valid_counts, torch.tensor(range(valid_size)).to(device), vocab_size, temporal=True,times=valid_times)
test_cvz, test_ts = get_batch(test_tokens, test_counts, torch.tensor(range(test_size)).to(device), vocab_size, temporal=True, times=test_times)


def evaluate(model):
    model.eval()
    alpha = model.mu_q_alpha.clone().contiguous()  # KxTxL
    alpha = alpha.permute(1, 0, 2)
    beta = model.get_beta(alpha)
    cnt = 0
    tc = 0
    for time in range(0, beta.shape[0]):
        beta_t = beta[time, :, :]
        cnt += 1
        tc += get_topic_coherence(beta_t, train_cvz)
    tc /= cnt
    print(f'tc: {tc}')

def _diversity_helper(beta, num_tops):
    list_w = torch.zeros((int(beta.shape[0]), num_tops))
    for k in range(int(beta.shape[0])):
        gamma = beta[k, :]
        top_words = gamma.argsort()[-num_tops:]
        list_w[k, :] = top_words
    list_w = list_w.reshape(-1)
    n_unique = len(list_w.unique())
    diversity = n_unique / (beta.shape[0] * num_tops)
    return diversity

# define model
model = DTMTIR(vocab_size=train_cvz.shape[1],
               num_topics=30,
               num_times=num_times,
               hidden=800,
               dropout=0.0,
               delta=0.005,
               useEmbedding=False,
               rho_size=300,
               eta_size=128,
               data_size=train_rnn_inp.shape[0]
               )

# print model parameters
# count_parameters(model)

batch_size = args.batch_size
num_epochs=1000
bar = trange(num_epochs)

num_batches = int(math.ceil(train_cvz.shape[0] / batch_size))
optim = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)

combined_loss_trace = []
val_ppl_trace = []
for epoch in bar:

    batch_loss_trace = []
    for i in range(num_batches):
        model.train()
        model.zero_grad()
        optim.zero_grad()

        if (i + 1) * batch_size > len(train_cvz):
            batch_docs = train_cvz[i * batch_size:, :]
            time_batch = train_ts[i * batch_size:]
        else:
            batch_docs = train_cvz[i * batch_size:(i + 1) * batch_size, :]
            time_batch = train_ts[i * batch_size:(i + 1) * batch_size]

        batch_docs = batch_docs.nan_to_num()

        # Calculate loss of transformer model
        recon_loss, kl_alpha, kl_theta, kld_eta_gp = model(
            batch_docs, batch_docs, time_batch, train_rnn_inp, len(train_cvz))

        # scale the product
        bsz = batch_docs.size(0)
        kl_loss = recon_loss.sum() + kl_alpha.sum() + kl_theta.sum() - 0.2*kld_eta_gp.sum()
        kl_loss = kl_loss.sum()
        # optimizer
        batch_loss = kl_loss
        bar.set_postfix(recon='{:.5e}'.format(recon_loss),
                        alpha='{:.2e}'.format(kl_alpha),
                        theta='{:.2e}'.format(kl_theta.sum()))

        # gradient step
        batch_loss.backward()
        optim.step()

        # add loss
        batch_loss_trace.append(batch_loss.detach().cpu().item())

    combined_loss_trace.append(batch_loss_trace)

    if epoch> 0 and epoch % 100 == 0:
        # KxTxL
        beta = model.get_beta()[0]
        print(beta.shape)
        # calcaulate the topic coherence
        cnt = 0
        tc = 0
        for time in range(0,beta.shape[0]):
            beta_t = beta[time,:,:]
            cnt+=1
            tc_k=get_topic_coherence(beta_t, train_cvz)
            print(tc_k)
            tc+=tc_k

        tc/=cnt
        print(f'tc: {tc}')

    model.eval()
    valid_cvz = valid_cvz.nan_to_num().to(device)
    ppl = model.predict(valid_cvz, valid_cvz, valid_ts.nan_to_num().to(device), valid_rnn_inp.nan_to_num().to(device))
    print(f'Validation perplexity: {ppl}')
    val_ppl_trace.append(ppl)
    # KxTxL
    beta = model.get_beta()[0]
    td = 0
    for t in range(beta.shape[0]):
        d=_diversity_helper(beta[t],25)
        td+=d
    print(f'TD: {td/beta.shape[0]}')