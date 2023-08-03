import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb


window_size = 2
word_vec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('counting PPMI ...')
W = ppmi(C, verbose=True)
print('calculating SVD ...')
try:
    from sklearn.utils.extmath import randomized_svd
    # n_components代表要提取的奇异值数量
    U, S, V = randomized_svd(W, n_components=word_vec_size, n_iter=5, random_state=None)
except ImportError:
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :word_vec_size]
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
