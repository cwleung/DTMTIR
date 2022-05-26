import os
import pickle

import numpy as np
import scipy.io
import torch
from tqdm import trange

from model.dtmtir import TETM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fetch(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens')
        count_file = os.path.join(path, 'bow_tr_counts')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens')
        count_file = os.path.join(path, 'bow_va_counts')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens')
        count_file = os.path.join(path, 'bow_ts_counts')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'tokens_1': tokens_1, 'counts_1': counts_1, 'tokens_2': tokens_2,
                'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts}


def _fetch_temporal(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens')
        count_file = os.path.join(path, 'bow_tr_counts')
        time_file = os.path.join(path, 'bow_tr_timestamps')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens')
        count_file = os.path.join(path, 'bow_va_counts')
        time_file = os.path.join(path, 'bow_va_timestamps')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens')
        count_file = os.path.join(path, 'bow_ts_counts')
        time_file = os.path.join(path, 'bow_ts_timestamps')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    times = scipy.io.loadmat(time_file)['timestamps'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'times': times,
                'tokens_1': tokens_1, 'counts_1': counts_1,
                'tokens_2': tokens_2, 'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts, 'times': times}


def get_data(path, temporal=False):
    ### load vocabulary
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    if not temporal:
        train = _fetch(path, 'train')
        valid = _fetch(path, 'valid')
        test = _fetch(path, 'test')
    else:
        train = _fetch_temporal(path, 'train')
        valid = _fetch_temporal(path, 'valid')
        test = _fetch_temporal(path, 'test')

    return vocab, train, valid, test


def get_batch(tokens, counts, ind, vocab_size, temporal=False, times=None):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    if temporal:
        times_batch = np.zeros((batch_size,))
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        if temporal:
            timestamp = times[doc_id]
            times_batch[i] = timestamp
        L = count.shape[1]
        if len(doc) == 1:
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    if temporal:
        times_batch = torch.from_numpy(times_batch).to(device)
        return data_batch, times_batch
    return data_batch


def get_rnn_input(tokens, counts, times, num_times, vocab_size, num_docs):
    # (data_batch,times_batch, num_times, vocab_size):
    ind = torch.randperm(num_docs).to(device)
    data_batch, times_batch = get_batch(tokens, counts, ind, vocab_size, temporal=True, times=times)
    rnn_input = torch.zeros(num_times, vocab_size).to(device)
    cnt = torch.zeros(num_times, ).to(device)
    for t in range(num_times):
        tmp = (times_batch == t).nonzero()
        docs = data_batch[tmp].squeeze().sum(0)
        rnn_input[t] += docs
        cnt[t] += tmp.shape[0]
    rnn_input = rnn_input / cnt.unsqueeze(1)
    return rnn_input


print('Getting vocabulary ...')
data_file = os.path.join('./', 'min_df_{}'.format(100))
vocab, train, valid, test = get_data(data_file, temporal=True)
vocab_size = len(vocab)

# 1. training data
print('Getting training data ...')
train_tokens = train['tokens']
train_counts = train['counts']
train_times = train['times']
num_times = train_times.max() - train_times.min() + 1  # len(np.unique(train_times))
num_docs_train = len(train_tokens)

batch_size = 1024
test_ratio = 0.8
train_size = int(np.floor(test_ratio * len(train_tokens)))
test_size = int(np.floor(0.15 * len(train_tokens)))
# test_size = test_size-(test_size%batch_size)
valid_size = len(train_tokens) - train_size - test_size

# 2. dev set
print('Getting validation data ...')
valid_tokens = valid['tokens']
valid_counts = valid['counts']
valid_times = valid['times']
num_docs_valid = len(valid_tokens)
valid_rnn_inp = get_rnn_input(
    valid_tokens, valid_counts, valid_times, num_times, vocab_size, valid_size).nan_to_num()

# 3. test data
print('Getting testing data ...')
test_tokens = test['tokens']
test_counts = test['counts']
test_times = test['times']
num_docs_test = len(test_tokens)
test_rnn_inp = get_rnn_input(
    test_tokens, test_counts, test_times, num_times, vocab_size, test_size).nan_to_num()

train_rnn_inp = get_rnn_input(
    train_tokens, train_counts, train_times, num_times, vocab_size, train_size).nan_to_num()

# %%

train_cvz, train_ts = get_batch(
    train_tokens, train_counts, torch.tensor(range(train_size)).to(device), vocab_size, temporal=True,
    times=train_times)
valid_cvz, valid_ts = get_batch(
    valid_tokens, valid_counts, torch.tensor(range(valid_size)).to(device), vocab_size, temporal=True,
    times=valid_times)
test_cvz, test_ts = get_batch(
    test_tokens, test_counts, torch.tensor(range(test_size)).to(device), vocab_size, temporal=True, times=test_times)


import torch
import math


def evaluate(model):
    model.eval()
    alpha = model.mu_q_alpha.clone().contiguous()  # KxTxL
    alpha = alpha.permute(1, 0, 2)
    beta = model.get_beta(alpha)
    beta = beta[:, :, :-3]
    cnt = 0
    tc = 0
    for time in range(0, beta.shape[0]):
        beta_t = beta[time, :, :]
        cnt += 1
        tc += get_topic_coherence(beta_t, cvz)
    tc /= cnt
    print(f'tc: {tc}')


torch.set_default_tensor_type('torch.FloatTensor')
# get index first
# train_rnn_inp = get_rnn_inp(train_cvz.to(device), train_ts.to(device), len(times), train_cvz.shape[1])
# test_rnn_inp = get_rnn_inp(test_cvz.to(device), test_ts.to(device), len(times), test_cvz.shape[1])
# valid_rnn_inp = get_rnn_inp(valid_cvz.to(device), valid_ts.to(device), len(times), valid_cvz.shape[1])
# define model
model = TETM(vocab_size=cvz.shape[1],
             num_topics=30,
             num_times=len(times),
             hidden=800,
             dropout=0.0,
             delta=0.005,
             emb_type='NN',
             useEmbedding=True,
             trainEmbedding=True,
             rho_size=300,
             eta_size=128,
             data_size=train_rnn_inp.shape[0]
             )


# print_top_words(beta[:,:,:-3],vocab)
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


# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# print model parameters
# count_parameters(model)
print(model)

num_epochs = 1000
bar = trange(num_epochs)

# dataloader loop
# ODO save the loss/batch-loss
# - reconstruction loss
recon_loss_trace = []
# - kl-loss
kl_loss_trace = []
# - transformer loss
trans_loss_trace = []
# - combined loss, can be done with post-processing
combined_loss_trace = []
# - validation perplexity
val_ppl_trace = []

num_batches = int(math.ceil(train_cvz.shape[0] / batch_size))
optim = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)
for epoch in bar:
    batch_recon_loss = []
    batch_kl_loss = []
    batch_trans_loss = []
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
        # normalize batch
        sums = batch_docs.sum(1).unsqueeze(1)
        normalized_data_batch = batch_docs  # / sums
        # Calculate loss of transformer model
        recon_loss, kl_alpha, kl_eta, kl_theta, kld_eta_gp = model(batch_docs, normalized_data_batch, time_batch,
                                                                   train_rnn_inp, len(train_cvz))
        # scale the product
        bsz = batch_docs.size(0)
        # coeff = len(cvz) / bsz
        kl_loss = recon_loss + kl_alpha + kl_theta + kl_eta - 0.5 * kld_eta_gp
        kl_loss = kl_loss.sum()
        # optimizer
        batch_loss = kl_loss  # + recon_loss
        bar.set_postfix(recon='{:.5e}'.format(recon_loss),
                        alpha='{:.2e}'.format(kl_alpha),
                        theta='{:.2e}'.format(kl_theta.sum()),
                        eta='{:.2e}'.format(kl_eta.sum()))

        # gradient step
        batch_loss.backward()
        optim.step()
        batch_loss.item()
        #         batch_recon_loss.append(recon_loss.item())
        batch_kl_loss.append(kl_loss.item())
    # store the average loss
    #     recon_loss_trace.append(batch_recon_loss)
    kl_loss_trace.append(batch_kl_loss)

    #   if epoch % 100 == 0 and epoch > 0:
    # topic coherence
    #        evaluate(model)
    if epoch > 0 and epoch % 980 == 0:
        # KxTxL
        alpha = model.get_alpha()[0]
        beta = model.get_beta(alpha)
        print(beta.shape)

        cnt = 0
        tc = 0
        for time in range(0, beta.shape[0]):
            beta_t = beta[time, :, :]
            cnt += 1
            tc_k = get_topic_coherence(beta_t, train_cvz)
            print(tc_k)
            tc += tc_k

        tc /= cnt
        print(f'tc: {tc}')

    model.eval()
    sums = valid_cvz.sum(1).unsqueeze(1)
    valid_cvz = valid_cvz.nan_to_num()
    normalized_valid_batch = valid_cvz.nan_to_num()  # / sums
    ppl = model.predict(valid_cvz, normalized_valid_batch, valid_ts.nan_to_num(), valid_rnn_inp.nan_to_num())
    print(f'Validation perplexity: {ppl}')
    val_ppl_trace.append(ppl)
    # KxTxL
    alpha = model.get_alpha()[0]
    beta = model.get_beta(alpha)
    td = 0
    for t in range(beta.shape[0]):
        d = _diversity_helper(beta[t], 25)
        td += d
    print(f'TD: {td / beta.shape[0]}')

model.eval()
print(test_cvz.shape)
test_cvz = test_cvz.nan_to_num().float()
normalized_test_batch = test_cvz  # / sums
ppl = model.predict(test_cvz, normalized_test_batch, test_ts, test_rnn_inp.nan_to_num())
print(f'Validation perplexity: {ppl}')

# %%

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=1, shuffle=False)
# test_rnn_inp = get_rnn_input(test_loader, len(ts.unique()),cvz.shape[1]).cuda().nan_to_num(0)
print(d_batch, t_batch, len(dataset))

test_ppl = 0
test_cnt = 0
for data in test_loader:
    d_batch, t_batch = cvz[data['index'] - 1, :].to(device), ts[data['index'] - 1].to(device)
    test_rnn_inp = get_rnn_inp(d_batch, t_batch, len(ts.unique()), cvz.shape[1]).cuda().nan_to_num(0)
    test_ppl += model.predict(d_batch, t_batch, test_rnn_inp)
    test_cnt += 1
# perplexity
test_ppl /= test_cnt
print(f'Test perplpexity: {test_ppl}')
