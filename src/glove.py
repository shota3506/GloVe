import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class GloVe(nn.Module):
    def __inin__(self, vocab_size, co_occurences, embedding_dim=50, x_max=100, alpha=0.75):
        super(GloVe, self).__init__()

        self.co_occurences = co_occurences
        self.x_max = x_max
        self.alpha = alpha

        self.W = nn.Embedding(vocab_size, embedding_dim)
        self.b = nn.Embedding(vocab_size, 1)
        self.tilde_W = nn.Embedding(vocab_size, embedding_dim)
        self.tilde_b = nn.Embedding(vocab_size, 1)

        init.kaiming_normal(self.W.weight.data)
        init.kaiming_normal(self.b.weight.data)
        init.kaiming_normal(self.tilde_W.weight.data)
        init.kaiming_normal(self.tilde_b.weight.data)

    def forward(self, words, target_words):
        batch_size = len(words)

        co_ocs = np.array([self.co_occurences[words[i], target_words[i]] for i in range(batch_size)])
        weights = np.array([min(np.power((x / self.x_max), self.alpha), 1) for x in co_ocs])

        co_ocs = torch.from_numpy(np.array(co_ocs)).type(torch.FloatTensor)
        weights = torch.from_numpy(np.array(weights)).type(torch.FloatTensor)

        embedded_words = self.W(words)
        bias = self.b(words)
        embedded_target_words = self.tilde_W(target_words)
        target_bias = self.tilde_b(target_words)

        loss = torch.sum(
            torch.pow(
                weights * ((embedded_words * embedded_target_words).sum(1) + bias + target_bias).squeeze(1) - torch.log(co_ocs),
                2
            )
        )
        return loss

    def embedding(self):
        return self.W.weight.data.cpu().numpy() + self.tilde_W.weight.data.cpu().numpy()
