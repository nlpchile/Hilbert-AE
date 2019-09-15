import os
import re
import json
import torch
import numpy as np
import h5py

def remove_numbers(text):
    text = re.sub(r'\d+', '', text)
    return text


def remove_punctuation(text):
    text = re.sub(r'[^\w\s]','',text)
    return text


def tokenizer_function(text):
    # text = clean_str(text)
    text = [x for x in text.split(" ") if x != "" and x.find(" ") == -1]
    return text

def preprocess_text(text):
    text = text.replace("\\xa0", " ")
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = text.strip()
    text = text.lower()
    return tokenizer_function(text)

def get_sentences_in_batch(x, vocab):
    for sent in x:
        str1 = ""
        for word in sent:
            str1 += vocab.itos[word] + " "
        print(str1)


def helper_h5py(dset1,
                dset2,
                text_encodded,
                current_buffer_allocation,
                current_buffer_size,
                max_sequence_length,):

    if current_buffer_allocation >= current_buffer_size:
        current_buffer_size = current_buffer_size + 1
        dset1.resize((current_buffer_size, max_sequence_length))
        dset2.resize((current_buffer_size, max_sequence_length))

    len_text = len(text_encodded)

    if len_text > max_sequence_length:
        return dset1, dset2, current_buffer_allocation, current_buffer_size

    text_padded = np.zeros(max_sequence_length)
    mask_padded = np.zeros(max_sequence_length)

    text_padded[:len_text] = text_encodded
    mask_padded[:len_text] = np.ones(len_text)
    dset1[current_buffer_allocation] = text_padded
    dset2[current_buffer_allocation] = mask_padded
    current_buffer_allocation += 1

    return dset1, dset2, current_buffer_allocation, current_buffer_size


class Corpus(object):
    def __init__(self, max_sequence_length):
        self.idx2word = {}
        self.word2idx = {}
        self.frequencies = {}
        self.corpus = []
        self.max_sequence_length = max_sequence_length
        self.max_sequence_dataset = 0

    def add_word(self, word):
        last_idx = len(self.word2idx) + 1 # TODO: parten desde 1
        if word not in self.word2idx:
            self.word2idx[word] = last_idx
            self.idx2word[last_idx] = word
            self.frequencies[word] = 1
        else:
            self.frequencies[word] += 1

    def encode(self, text):
        text_encodded = []
        for word in text:
            if word not in self.word2idx:
                self.add_word(word)
            text_encodded.append(self.word2idx[word])
        return text_encodded

    def decode(self, text_encodded):
        text_decoded = []
        for idx in text_encodded:
            if idx not in self.idx2word:
                word = "<unk>"
            else:
                word = self.idx2word[idx]
            text_decoded.append(word)
        return text_decoded

    def process_text(self, path_to_ifile):
        with open(path_to_ifile, "r") as freader:
            for line in freader:
                text = preprocess_text(line)
                for word in text:
                    self.add_word(word)
                self.corpus.append(text)
                if len(text) > self.max_sequence_dataset:
                    self.max_sequence_dataset = len(text)
                
    def save_h5py(self, path_to_ofile, path_to_ifile=None):
        f = h5py.File(path_to_ofile, 'w')
        current_buffer_size = 1
        current_buffer_allocation = 0
        dset1 = f.create_dataset('sequence', 
                             (current_buffer_size, self.max_sequence_length),
                             maxshape=(None, self.max_sequence_length), 
                             dtype='int32')
        dset2 = f.create_dataset('mask', 
                                (current_buffer_size, self.max_sequence_length), 
                                maxshape=(None, self.max_sequence_length),
                                dtype='uint8')

        if path_to_ifile == None:
            if len(self.corpus) == 0:
                raise ValueError
            else: 
                for text in self.corpus:
                    dset1, dset2, current_buffer_allocation, current_buffer_size = helper_h5py(dset1,
                                                                                               dset2,
                                                                                               self.encode(text),
                                                                                               current_buffer_allocation,
                                                                                               current_buffer_size,
                                                                                               self.max_sequence_length)
        else:
            with open(path_to_ifile, "r") as freader:
                for line in freader:
                    text = preprocess_text(line)
                    dset1, dset2, current_buffer_allocation, current_buffer_size = helper_h5py(dset1,
                                                                                               dset2,
                                                                                               self.encode(text),
                                                                                               current_buffer_allocation,
                                                                                               current_buffer_size,
                                                                                               self.max_sequence_length)

    def save_corpus_parameters(self, outfile):
        json_dict = {"max_sequence_dataset":self.max_sequence_dataset,
                     "vocab_size":len(self.word2idx),
                     "word2idx":self.word2idx,
                     "idx2word":self.idx2word,
                     "frequencies":self.frequencies
                     }

        with open(outfile, 'w') as fp:
            json.dump(json_dict, fp,  indent=4)


class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename, keys_types):
        """
        filename (string): path to file with dataset in format h5py
        keys_types (dictionary): dictionary with dataset name and data type.
                                 example: {"sequence":torch.long, "mask":torch.uint8}
        """
        super(H5PytorchDataset, self).__init__()

        self.h5pyfile   = h5py.File(filename, 'r')
        self.keys_types = keys_types
        key0 = list(self.keys_types)[0]
        self.num_sequences = self.h5pyfile[key0].shape[0]

    def __getitem__(self, index):
        items = []
        for key, dtype in self.keys_types.items():
            items.append(torch.Tensor(self.h5pyfile[key][index,:]).type(dtype=dtype))
        return items

    def __len__(self):
        return self.num_sequences

    def merge_samples_to_minibatch(samples):
        samples_list = []
        for s in samples:
            samples_list.append(s)
        # sort according to length of sequence
        samples_list.sort(key=lambda x: len(x[0]), reverse=True)
        return zip(*samples_list)
