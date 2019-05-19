import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class GloVe(nn.Module):
    def __init__(self, vocab_size, co_occurences, embed_size=50, x_max=100, alpha=0.75):
        super(GloVe, self).__init__()

        self.co_occurences = co_occurences
        self.x_max = x_max
        self.alpha = alpha

        self.W = nn.Embedding(vocab_size, embed_size)
        self.b = nn.Embedding(vocab_size, 1)
        self.tilde_W = nn.Embedding(vocab_size, embed_size)
        self.tilde_b = nn.Embedding(vocab_size, 1)

        init.kaiming_normal_(self.W.weight.data)
        init.kaiming_normal_(self.b.weight.data)
        init.kaiming_normal_(self.tilde_W.weight.data)
        init.kaiming_normal_(self.tilde_b.weight.data)

    def forward(self, word_ids, target_ids):
        batch_size = len(word_ids)

        co_ocs = np.array([self.co_occurences[word_ids[i], target_ids[i]] for i in range(batch_size)])
        weights = np.array([min(np.power((x / self.x_max), self.alpha), 1) for x in co_ocs])

        co_ocs = torch.from_numpy(np.array(co_ocs)).type(torch.FloatTensor)
        weights = torch.from_numpy(np.array(weights)).type(torch.FloatTensor)

        embedded_words = self.W(word_ids)
        bias = self.b(word_ids)
        embedded_target_words = self.tilde_W(target_ids)
        target_bias = self.tilde_b(target_ids)

        loss = torch.sum(
            torch.pow(
                weights * ((embedded_words * embedded_target_words).sum(1) + bias + target_bias).squeeze(1) - torch.log(co_ocs),
                2
            )
        )
        return loss

    def embedding(self):
        return self.W.weight.data.cpu().numpy() + self.tilde_W.weight.data.cpu().numpy()
