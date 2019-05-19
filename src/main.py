import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from stanfordnlp.server import CoreNLPClient
from glove import *


def tokenize(text):
    with CoreNLPClient(annotators=['tokenize'],
                       timeout=30000, memory='16G') as client:
        ann = client.annotate(text)

    word_list = [token.word for token in ann.sentencelessToken]
    vocab = np.unique(word_list)

    word2index = {word: index for index, word in enumerate(vocab)}
    index2word = {index: word for index, word in enumerate(vocab)}

    return word_list, vocab, word2index, index2word


def calc_co_occurences(word_list, vocab, word2index, context_size):
    vocab_size = len(vocab)
    co_occurences = np.zeros((vocab_size, vocab_size))

    for i in range(len(word_list)):
        for j in range(1, context_size + 1):
            index = word2index[word_list[i]]
            if i - j >= 0:
                lindex = word2index[word_list[i - j]]
                co_occurences[index][lindex] += 1.0 / j
            if i + j < len(word_list):
                rindex = word2index[word_list[i + j]]
                co_occurences[index][rindex] += 1.0 / j

    return co_occurences


class TextDataset(Dataset):
    def __init__(self, text_file, context_size, transform=None):
        self.transform = transform

        with open(text_file, 'r') as f:
            text = f.read().lower()
        self.text = text
        self.word_list, self.vocab, self.word2index, self.index2word = tokenize(text)
        self.co_occurences = calc_co_occurences(self.word_list, self.vocab, self.word2index, context_size)
        self.nonzero_index = np.transpose(np.nonzero(self.co_occurences))

    def __len__(self):
        return len(self.nonzero_index)

    def __getitem__(self, idx):
        word_id, target_id = self.nonzero_index[idx]

        if self.transform:
            word_id, target_id = self.transform(word_id), self.transform(target_id)

        return word_id, target_id


if __name__ == '__main__':
    context_size = 3
    embed_size = 50
    lr = 0.05
    batch_size = 16
    num_epochs = 10
    text_file = "short_story.txt"

    dataset = TextDataset(text_file, context_size)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    glove = GloVe(len(dataset.vocab), dataset.co_occurences, embed_size=embed_size)
    optimizer = optim.Adagrad(glove.parameters(), lr)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i_batch, (word_ids, target_ids) in enumerate(dataloader):
            loss = glove(word_ids, target_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i_batch + 1) % 100 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 100))
                running_loss = 0.0

    word_embedding = glove.embedding()
















    # with open("short_story.txt", 'r') as f:
    #     text = f.read().lower()
    #
    # with CoreNLPClient(annotators=['tokenize'],
    #                    timeout=30000, memory='16G') as client:
    #     ann = client.annotate(text)
    #
    # word_list = [token.word for token in ann.sentencelessToken]
    # vocab = np.unique(word_list)
    # vocab_size = len(vocab)
    #
    # word2index = {word: index for index, word in enumerate(vocab)}
    # index2word = {index: word for index, word in enumerate(vocab)}
    #
    # co_occurences = np.zeros((vocab_size, vocab_size))
    #
    # for i in range(len(word_list)):
    #     for j in range(1, context_size+1):
    #         index = word2index[word_list[i]]
    #         if i - j >= 0:
    #             lindex = word2index[word_list[i - j]]
    #             co_occurences[index][lindex] += 1.0 / j
    #         if i + j < len(word_list):
    #             rindex = word2index[word_list[i + j]]
    #             co_occurences[index][rindex] += 1.0 / j

    # nonzero_index = np.transpose(np.nonzero(co_occurences))