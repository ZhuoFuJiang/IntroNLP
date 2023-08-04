import numpy as np
from common.layers import Embedding
from negative_sampling_layer import NegativeSamplingLoss


class SkipGram:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # 生成层
        self.in_layer = Embedding(W_in)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 将所有的权重和梯度整理到列表中
        layers = [self.in_layer, self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in
        self.batch_size = None
        self.context_size = None
        self.hidden_size = H

    def forward(self, contexts, target):
        batch_size, context_size = contexts.shape
        self.batch_size = batch_size
        self.context_size = context_size
        h = self.in_layer.forward(target)
        h = h.repeat(context_size, axis=0)
        contexts = contexts.reshape(-1)
        loss = self.ns_loss.forward(h, contexts)
        return loss

    def backward(self, dout=1):
        # dh的形状为batch_size * context_size hidden_size
        dh = self.ns_loss.backward(dout)
        dh = dh.reshape(self.batch_size, self.context_size, self.hidden_size).transpose(0, 2, 1).sum(axis=2)
        self.in_layer.backward(dh)
        return None
