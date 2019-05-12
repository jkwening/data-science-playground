import itertools
import torch

# Project modules
from simple_chatbot.voc import EOS_TOKEN, PAD_TOKEN


def indices_from_sentence(voc, sentence):
    word2index = voc.get_word2index()
    return [word2index[word] for word in sentence.split()] + [EOS_TOKEN]


def add_zero_padding(l):
    return list(itertools.zip_longest(*l, fillvalue=PAD_TOKEN))


def binary_matrix(l):
    m = list()
    for i, seq in enumerate(l):
        m.append(list())
        for token in seq:
            m[i].append(0 if token == PAD_TOKEN else 1)
    return m


def input_var(l, voc):
    """Returns padded input sentence tensor and lengths"""
    indices_batch = [indices_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.tensor(data=[len(idx) for idx in indices_batch])
    pad_list = add_zero_padding(indices_batch)
    pad_var = torch.LongTensor(data=pad_list)
    return pad_var, lengths


def output_var(l, voc):
    """Returns padded target sequence tensor, padding mask, and max target length."""
    indices_batch = [indices_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indices) for indices in indices_batch])
    pad_list = add_zero_padding(indices_batch)
    mask = binary_matrix(pad_list)
    mask = torch.ByteTensor(data=mask)
    pad_var = torch.LongTensor(data=pad_list)
    return pad_var, mask, max_target_len


def batch_to_train_data(voc, pair_batch):
    """Returns all items for given batch of pairs."""
    pair_batch.sort(key=lambda x: len(x[0].split()), reverse=True)
    input_batch, output_batch = list(), list()

    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

    inp, lengths = input_var(input_batch, voc)
    output, mask, max_target_len = output_var(output_batch, voc)

    return inp, lengths, output, mask, max_target_len
