import os
import sys
import numpy as np
from time import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import (manifold, datasets, decomposition,
                     ensemble, random_projection)
import random
from time import time,sleep

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets



style_encoder_path = "/home/zhaoxt20/vae_tac_myself/exp_multi/Libritts_styles"

embedding_dict={}
speakers = os.listdir(style_encoder_path)


random.shuffle(speakers)
for speaker in speakers[:10]:
    speaker_path = os.path.join(style_encoder_path,speaker)
    sentences = os.listdir(speaker_path)
    if speaker not in embedding_dict:
        embedding_dict[speaker]=[]

    for sentence in sentences:
        sentence_path = os.path.join(speaker_path,sentence)
        embeddings = os.listdir(sentence_path)
        for embedding in embeddings:
            embedding_path = os.path.join(sentence_path,embedding)
            embedding_np = np.load(embedding_path)[0]
            embedding_dict[speaker].append(embedding_np)
    if len(embedding_dict[speaker])<=1:
        embedding_dict.pop(speaker)
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, verbose=1)
color_num = len(speakers) * 2
colors = cm.rainbow(np.linspace(0, 1, color_num))
Y=None
data=None
label=[]
for x in embedding_dict:
    if data is None:
        data = embedding_dict[x]
    else:
        data = data + embedding_dict[x]
    #data = tsne.fit_transform(data)
    # if Y is None:
    #     Y=np.array(data)
    # else:
    #     Y = np.concatenate((Y,data),axis=

    tmp=[]
    for i in range(len(embedding_dict[x])):
        tmp.append(x)
    label=label+tmp
print(len(data))
Y = tsne.fit_transform(data)





def plot_embedding_2d(X, title=None, save_path=None):
    # linear to [0,1]
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # print('plot X', X)
    # print('plot Char', Char)
    # at x0, x1, draw text
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        c = plt.cm.Set1(speakers.index(label[i]) % 10 / 10.)
        ax.text(X[i, 0], X[i, 1], str(label[i]), color=c,
                fontdict={'weight': 'bold', 'size': 4})

    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300)
    plt.close()


def plot_embedding_2d_focus(X, focus, title=None, save_path=None):
    # linear to [0,1]
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # print('plot X', X)
    # print('plot Char', Char)
    # at x0, x1, draw text
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ch = get_symbol(i)
        c = get_color(i)
        if ch in focus:
            # Log(ch)
            # Log(X[i, 0])
            # Log(X[i, 1])
            # ax.text(X[i, 0], X[i, 1], ch, color=c,
            #         fontdict={'weight': 'bold', 'size': 9})
            ax.text(X[i, 0]+np.random.normal(0, 0.0075), X[i, 1]+np.random.normal(0, 0.0075), ch, color=c,
                    fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300)
    plt.close()


# def plot_embedding(data, label, title):
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     data = (data - x_min) / (x_max - x_min)
#
#     fig = plt.figure()
#     ax = plt.subplot(111)
#     for i in range(data.shape[0]):
#         plt.text(data[i, 0], data[i, 1], str(speakers.index(label[i])),
#                  ,
#                  fontdict={'weight': 'bold', 'size': 9})
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(title)
#     return fig
#
# t0=time()
# fig = plot_embedding(Y,label,
#                          't-SNE embedding of the digits (time %.2fs)'
#                          % (time() - t0))
# plt.show(fig)
# fig.savefig('./temp.png')

plot_embedding_2d(Y, "t-SNE 2D", style_encoder_path+'/t-sne.png')
