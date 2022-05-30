import bokeh.plotting as bp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_df(data, wi, wj=None):
    """
    Obtain the document frequency
    :param data: document vocabulary matrix
    :param wi: word index w_i
    :param wj: word index w_j
    :return: document frequency for word w_i , w_i ∩ w_j
    """
    if wj is None:
        return torch.where(data[:, wi] > 0, 1, 0).sum(-1)
    else:
        df_wi = torch.where(data[:, wi] > 0, 1, 0)
        df_wj = torch.where(data[:, wj] > 0, 1, 0)
        return df_wj.sum(-1), (df_wi & df_wj).sum(-1)


def get_topic_coherence(beta, data):
    D = torch.tensor(len(data))  ## number of docs...data is list of documents
    TC = []
    num_topics = len(beta)
    counter = 0
    for k in range(num_topics):
        top_10 = list(torch.flip(beta[k].argsort()[-11:], [0]))
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            D_wi = get_df(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                D_wj, D_wi_wj = get_df(data, word, top_10[j])
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + (torch.log(D_wi) + torch.log(D_wj) - 2.0 * torch.log(D)) / (
                            torch.log(D_wi_wj) - torch.log(D))
                tmp += f_wi_wj
                j += 1
                counter += 1
            TC_k += tmp
        TC.append(TC_k.detach().cpu().numpy())
    TC = np.mean(TC) / counter
    # print('Topic coherence is: {}'.format(TC))
    return TC


def visualize(docs, _lda_keys, topics, theta):
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    # project to 2D
    tsne_lda = tsne_model.fit_transform(theta)
    colormap = []
    for name, hex in matplotlib.colors.cnames.items():
        colormap.append(hex)

    colormap = colormap[:len(theta[0, :])]
    colormap = np.array(colormap)

    title = '20 newsgroups TE embedding V viz'
    num_example = len(docs)

    plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                         title=title,
                         tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                         x_axis_type=None, y_axis_type=None, min_border=1)

    plt.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1],
                color=colormap[_lda_keys][:num_example])
    plt.show()


def get_rnn_inp(data_batch, times_batch, num_times, vocab_size):
    rnn_input = torch.zeros(num_times, vocab_size).to(device)
    cnt = torch.zeros(num_times, ).to(device)
    for t in range(num_times):
        tmp = (times_batch == t).nonzero()
        docs = data_batch[tmp].squeeze().sum(0)
        rnn_input[t] += docs
        cnt[t] += tmp.shape[0]
    rnn_input = rnn_input / cnt.unsqueeze(1)
    return rnn_input
