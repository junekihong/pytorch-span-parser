"""
Bi-LSTM network for span-based constituency parsing.
"""

from __future__ import print_function
from __future__ import division

import time
import random
import sys

#import dynet
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.functional as F
import torch.optim as optim

from phrase_tree import PhraseTree, FScore
from features import FeatureMapper
from parser import Parser
torch.manual_seed(1)

from pprint import pprint


class SubNetwork(nn.Module):
    
    def __init__(
            self,
            word_count, tag_count,
            word_dims, tag_dims,
            lstm_units,
            hidden_units,
            out_dim,
            droprate,
            spans,
            GPU=None,
    ):
        super(SubNetwork, self).__init__()

        self.word_count, self.tag_count = word_count, tag_count
        self.word_dims, self.tag_dims = word_dims, tag_dims

        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.out_dim = out_dim
        self.droprate = droprate
        self.spans = spans

        self.drop = nn.Dropout(droprate)
        self.GPU = GPU

        if GPU is not None:
            self.word_embed = nn.Embedding(word_count, word_dims).cuda(GPU)
            self.tag_embed = nn.Embedding(tag_count, tag_dims).cuda(GPU)
            self.lstm = nn.LSTM(word_dims + tag_dims,
                                lstm_units,
                                num_layers=2,
                                dropout=droprate,
                                bidirectional=True).cuda(GPU)
            self.span2hidden = nn.Linear(2 * spans * lstm_units, hidden_units).cuda(GPU)
            self.hidden2out = nn.Linear(hidden_units, out_dim).cuda(GPU)
        else:
            self.word_embed = nn.Embedding(word_count, word_dims)
            self.tag_embed = nn.Embedding(tag_count, tag_dims)
            self.lstm = nn.LSTM(word_dims + tag_dims,
                                lstm_units,
                                num_layers=2,
                                bidirectional=True)
            self.span2hidden = nn.Linear(2 * spans * lstm_units, hidden_units)
            self.hidden2out = nn.Linear(hidden_units, out_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.word_embed.weight.data.uniform_(-initrange, initrange)
        self.tag_embed.weight.data.uniform_(-initrange, initrange)
        self.span2hidden.bias.data.fill_(0)
        self.span2hidden.weight.data.uniform_(-initrange, initrange)
        self.hidden2out.bias.data.fill_(0)
        self.hidden2out.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        num_layers = 2
        if self.GPU is not None:
            return (autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_().cuda(self.GPU)),
                    autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_()).cuda(self.GPU))
        return (autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_()),
                autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_()))

    def evaluate_recurrent(self, word_inds, tag_inds, test=False):
        sentence = []

        wordvecs = self.word_embed(word_inds)
        tagvecs = self.tag_embed(tag_inds)
        wordvecs = wordvecs.view(self.word_dims, len(word_inds))
        tagvecs = tagvecs.view(self.tag_dims, len(tag_inds))

        sentence = torch.cat([wordvecs, tagvecs])
        sentence = sentence.view(1, sentence.size()[1], sentence.size()[0])
        lstm_out, self.hidden_lstm = self.lstm(sentence)

        #print(lstm_out)
        #print(lstm_out.size())
        #exit()

        self.embeddings = lstm_out.view(lstm_out.size()[1], lstm_out.size()[2])
        return lstm_out


    def forward(self, lefts, rights, test=False):
        span_out = []
        for left_index, right_index in zip(lefts, rights):
            embedding = self.embeddings[right_index] - self.embeddings[left_index - 1]
            span_out.append(embedding)
        hidden_input = torch.cat(span_out)
        hidden_input = hidden_input.view(1, (len(hidden_input)))
        
        if self.droprate > 0 and not test:
            pass
            #hidden_input = dynet.dropout(hidden_input, self.droprate)
        hidden_output = self.span2hidden(hidden_input)
        scores = self.hidden2out(hidden_output)
        return scores




#class Network(nn.Module):
class Network:

    def __init__(
        self,
        word_count, tag_count,
        word_dims, tag_dims,
        lstm_units,
        hidden_units,
        struct_out,
        label_out,
        droprate=0,
        struct_spans=4,
        label_spans=3,
        GPU=None,
    ):
        self.word_count = word_count
        self.tag_count = tag_count
        self.word_dims = word_dims
        self.tag_dims = tag_dims
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.struct_out = struct_out
        self.label_out = label_out
        self.droprate = droprate
        self.GPU = GPU

        self.struct = SubNetwork(word_count, tag_count, 
                                 word_dims, tag_dims, 
                                 lstm_units, hidden_units, struct_out, droprate, struct_spans,
                                 GPU)
        self.label = SubNetwork(word_count, tag_count,
                                word_dims, tag_dims,
                                lstm_units, hidden_units, label_out, droprate, label_spans,
                                GPU)



    def save(self, filename):
        """
        Saves the model using Torch's save functionality.
        """
        torch.save({
            "struct_state_dict": self.struct.state_dict(),
            "label_state_dict": self.label.state_dict(),
            "word_count": self.word_count,
            "tag_count": self.tag_count,
            "word_dims": self.word_dims,
            "tag_dims": self.tag_dims,
            "lstm_units": self.lstm_units,
            "hidden_units": self.hidden_units,
            "struct_out": self.struct_out,
            "label_out": self.label_out,
        }, filename)

    @staticmethod
    def load(filename, GPU=None):
        """
        Loads file created by save() method.
        """
        checkpoint = torch.load(filename)
        word_count = checkpoint["word_count"]
        tag_count = checkpoint["tag_count"]
        word_dims = checkpoint["word_dims"]
        tag_dims = checkpoint["tag_dims"]
        lstm_units = checkpoint["lstm_units"]
        hidden_units = checkpoint["hidden_units"]
        struct_out = checkpoint["struct_out"]
        label_out = checkpoint["label_out"]
        
        network = Network(
            word_count=word_count,
            tag_count=tag_count,
            word_dims=word_dims,
            tag_dims=tag_dims,
            lstm_units=lstm_units,
            hidden_units=hidden_units,
            struct_out=struct_out,
            label_out=label_out,
            GPU=GPU,
        )
        network.struct.load_state_dict(checkpoint["struct_state_dict"])
        network.label.load_state_dict(checkpoint["label_state_dict"])
        return network

    @staticmethod
    def train(
        feature_mapper,
        word_dims,
        tag_dims,
        lstm_units,
        hidden_units,
        epochs,
        batch_size,
        train_data_file,
        dev_data_file,
        model_save_file,
        droprate,
        unk_param,
        alpha=1.0,
        beta=0.0,
        GPU=None,
    ):

        start_time = time.time()

        fm = feature_mapper
        word_count = fm.total_words()
        tag_count = fm.total_tags()

        network = Network(
            word_count=word_count,
            tag_count=tag_count,
            word_dims=word_dims,
            tag_dims=tag_dims,
            lstm_units=lstm_units,
            hidden_units=hidden_units,
            struct_out=2,
            label_out=fm.total_label_actions(),
            droprate=droprate,
            GPU=GPU,
        )

        if GPU is not None:
            struct_loss_function = nn.NLLLoss().cuda(GPU)
            label_loss_function = nn.NLLLoss().cuda(GPU)
        else:
            struct_loss_function = nn.NLLLoss()
            label_loss_function = nn.NLLLoss()

        struct_trainer = optim.Adam(network.struct.parameters(), lr = 0.0001)
        label_trainer = optim.Adam(network.label.parameters(), lr = 0.0001)


        print('Hidden units: {},  per-LSTM units: {}'.format(
            hidden_units,
            lstm_units,
        ))
        print('Embeddings: word={}  tag={}'.format(
            (word_count, word_dims),
            (tag_count, tag_dims),
        ))
        print('Dropout rate: {}'.format(droprate))
        print('Parameters initialized in [-0.01, 0.01]')
        print('Random UNKing parameter z = {}'.format(unk_param))
        print('Exploration: alpha={} beta={}'.format(alpha, beta))

        training_data = fm.gold_data_from_file(train_data_file)
        num_batches = -(-len(training_data) // batch_size) 
        print('Loaded {} training sentences ({} batches of size {})!'.format(
            len(training_data),
            num_batches,
            batch_size,
        ))
        parse_every = -(-num_batches // 4)

        dev_trees = PhraseTree.load_treefile(dev_data_file)
        print('Loaded {} validation trees!'.format(len(dev_trees)))

        best_acc = FScore()

        for epoch in xrange(1, epochs + 1):
            print('........... epoch {} ...........'.format(epoch))

            total_cost = 0.0
            total_states = 0
            training_acc = FScore()

            np.random.shuffle(training_data)

            for b in xrange(num_batches):
                batch = training_data[(b * batch_size) : ((b + 1) * batch_size)]
                network.struct.hidden = network.struct.init_hidden(batch_size)
                network.label.hidden = network.label.init_hidden(batch_size)

                explore = [
                    Parser.exploration(
                        example,
                        fm,
                        network,
                        alpha=alpha,
                        beta=beta,
                    ) for example in batch
                ]
                for (_, acc) in explore:
                    training_acc += acc

                batch = [example for (example, _) in explore]
                errors = []

                for example in batch:
                    network.struct.zero_grad()
                    network.label.zero_grad()
                    network.struct.hidden = network.struct.init_hidden(batch_size)
                    network.label.hidden = network.label.init_hidden(batch_size)

                    ## random UNKing ##
                    for (i, w) in enumerate(example['w']):
                        if w <= 2:
                            continue

                        freq = fm.word_freq_list[w]
                        drop_prob = unk_param / (unk_param + freq)
                        r = np.random.random()
                        if r < drop_prob:
                            example['w'][i] = 0

                    network.struct.evaluate_recurrent(
                        example['w'],
                        example['t'],
                    )
                    network.label.evaluate_recurrent(
                        example['w'],
                        example['t'],
                    )

                    struct_scores, struct_corrects = [], []
                    for (left, right), correct in example['struct_data'].items():
                        struct_scoring = network.struct(left, right)
                        struct_scores.append(struct_scoring)
                        struct_corrects.append(correct)
                    struct_scores = torch.cat(struct_scores)

                    if GPU is not None:
                        struct_corrects = autograd.Variable(torch.LongTensor(struct_corrects)).view(-1).cuda(GPU)
                    else:
                        struct_corrects = autograd.Variable(torch.LongTensor(struct_corrects)).view(-1)
                    logsoftmax = nn.LogSoftmax()
                    softmax = logsoftmax(struct_scores)
                    loss = struct_loss_function(softmax, struct_corrects)
                    total_cost += loss.data[0]
                    loss.backward()


                    label_scores, label_corrects = [],[]
                    for (left, right), correct in example['label_data'].items():
                        label_scoring = network.label(left, right)
                        label_scores.append(label_scoring)
                        label_corrects.append(correct)
                    label_scores = torch.cat(label_scores)

                    if GPU is not None:
                        label_corrects = autograd.Variable(torch.LongTensor(label_corrects)).cuda(GPU)
                    else:
                        label_corrects = autograd.Variable(torch.LongTensor(label_corrects))
                    softmax = logsoftmax(label_scores)
                    loss = label_loss_function(softmax, label_corrects)
                    total_cost += loss.data[0]
                    loss.backward()

                    total_states += len(example['struct_data'])
                    total_states += len(example['label_data'])


                #batch_error = dynet.esum(errors)
                #total_cost += batch_error.scalar_value()
                #batch_error.backward()
                #trainer.update()


                mean_cost = (total_cost / total_states)

                print(
                    '\rBatch {}  Mean Cost {:.4f} [Train: {}]'.format(
                        b,
                        mean_cost,
                        training_acc,
                    ),
                    end='',
                )
                sys.stdout.flush()

                if ((b + 1) % parse_every) == 0 or b == (num_batches - 1):
                    dev_acc = Parser.evaluate_corpus(
                        dev_trees,
                        fm,
                        network,
                    )
                    print('  [Val: {}]'.format(dev_acc))

                    if dev_acc > best_acc:
                        best_acc = dev_acc 
                        network.save(model_save_file)
                        print('    [saved model: {}]'.format(model_save_file)) 

            current_time = time.time()
            runmins = (current_time - start_time)/60.
            print('  Elapsed time: {:.2f}m'.format(runmins))


