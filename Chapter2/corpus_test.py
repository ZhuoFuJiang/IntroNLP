import matplotlib.pyplot as plt
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print(corpus)
print(id_to_word)
c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print(cos_similarity(c0, c1))
print(most_similar('you', word_to_id, id_to_word, C, top=5))
W = ppmi(C)
np.set_printoptions(precision=3)
print('covariance matrix')
print(C)
print('-' * 50)
print('PPMI')
print(W)

# 通过奇异值分解降维
U, S, V = np.linalg.svd(W)
print(C[0])
print(W[0])
print(U[0])

# 绘制图形展示
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()