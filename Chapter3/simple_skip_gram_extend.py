import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleSkipGram:
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
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        # contexts的形状为batch_size 2*window_size vocab_size
        batch_size, context_size, vocab_size = contexts.shape
        self.context_size = context_size
        contexts = contexts.reshape(-1, vocab_size)
        s = s.repeat(context_size, axis=0)
        loss = self.loss_layer.forward(s, contexts)
        return loss

    def backward(self, dout=1):
        # 反向传播
        dl = self.loss_layer.backward(dout)
        ds = dl.reshape(-1, self.context_size, self.vocab_size).transpose(0, 2, 1).sum(axis=2)
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None
