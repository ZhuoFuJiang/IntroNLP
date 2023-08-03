import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 生成层
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 将所有的权重和梯度整理到列表中
        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context_size = None

    def forward(self, contexts, target):
        # 前向传播
        # contexts的形状应为batch_size 2*window_size vocab_size
        batch_size, context_size, vocab_size = contexts.shape
        self.context_size = context_size
        contexts = contexts.reshape(-1, vocab_size)
        # h的形状为batch_size*context_size hidden_size
        h = self.in_layer.forward(contexts)
        h = h.reshape(batch_size, context_size, -1).transpose(0, 2, 1).mean(axis=2)
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        # 反向传播
        ds = self.loss_layer.backward(dout)
        dh = self.out_layer.backward(ds)
        dh = np.expand_dims(dh, axis=2).repeat(self.context_size, axis=2) / self.context_size
        dh = dh.transpose(0, 2, 1).reshape(-1, self.hidden_size)
        self.in_layer.backward(dh)
        return None
