import random
from sklearn import manifold
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')

# source_path
# 	├── speaker_1 (dir storing embedding files,named sentence_id.npy,shape [1,embedding_size])
# 	└── speaker_2
#     ....
#   └── speaker_n


# source_path
source_path = "data"

embedding_dict = {}
speakers = os.listdir(source_path)
speaker_list = []
# random select 3 speakers
random.shuffle(speakers)
sentence_label = []
speaker_label = []
for i, speaker in enumerate(speakers[:3]):
    speaker_list.append(speaker)
    speaker_path = os.path.join(source_path, speaker)
    sentences = os.listdir(speaker_path)
    embedding_dict[i] = []
    for sentence in sentences:
        sentence_path = os.path.join(speaker_path, sentence)
        sentence_embedding = np.load(sentence_path)[0]
        embedding_dict[i].append(sentence_embedding)
        speaker_label.append(i)
        sentence_label.append(int(sentence.replace('.npy', '')))
    if len(embedding_dict[i]) <= 1:
        embedding_dict.pop(i)
print("speaker list:", speaker_list)
tsne = manifold.TSNE(n_components=2, init='pca', perplexity=3, random_state=0, verbose=1)
data = []
for i in range(3):
    data = data + embedding_dict[i]
print("num data:", len(data))
data = np.array(data)
data = (data - data.mean(axis=0)) / data.std(axis=0)
X = tsne.fit_transform(data)
markers = ['x', '^', '.']
# linear to [0,1]
x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
X = (X - x_min) / (x_max - x_min)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(X.shape[0]):
    c = plt.cm.Set1(sentence_label[i])
    ax.scatter(X[i, 0], X[i, 1], color=c, marker=markers[speaker_label[i]])

plt.title('t-SNE 2D')
plt.savefig(os.path.join(source_path, 'tsne.png'), format='png', dpi=300)
plt.close()
