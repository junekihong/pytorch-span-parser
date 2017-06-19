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

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == autograd.Variable:
        return autograd.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class LSTM(nn.Module):
    def __init__(
            self,
            word_count, tag_count,
            word_dims, tag_dims,
            lstm_units,
            droprate,
            GPU=None,
    ):
        super(LSTM, self).__init__()
        self.word_dims, self.tag_dims = word_dims, tag_dims

        self.lstm_units = lstm_units
        self.droprate = droprate
        self.GPU = GPU
        if GPU is not None:
            self.word_embed = nn.Embedding(word_count, word_dims).cuda(GPU)
            self.tag_embed = nn.Embedding(tag_count, tag_dims).cuda(GPU)
            self.lstm = nn.LSTM(word_dims + tag_dims,
                                lstm_units,
                                num_layers=2,
                                dropout=droprate,
                                bidirectional=True).cuda(GPU)
        else:
            self.word_embed = nn.Embedding(word_count, word_dims)
            self.tag_embed = nn.Embedding(tag_count, tag_dims)
            self.lstm = nn.LSTM(word_dims + tag_dims,
                                lstm_units,
                                num_layers=2,
                                dropout=droprate,
                                bidirectional=True)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_embed.weight.data.uniform_(-initrange, initrange)
        self.tag_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, word_inds, tag_inds, test=False):
        wordvecs = self.word_embed(word_inds)
        tagvecs = self.tag_embed(tag_inds)
        wordvecs = wordvecs.view(self.word_dims, len(word_inds))
        tagvecs = tagvecs.view(self.tag_dims, len(tag_inds))

        sentence = torch.cat([wordvecs, tagvecs])
        sentence = sentence.view(1, sentence.size()[1], sentence.size()[0])

        lstm_out, self.hidden_lstm = self.lstm(sentence)
        self.embeddings = torch.cat(self.hidden_lstm)
        self.embeddings = self.embeddings.view(self.embeddings.size(1), self.embeddings.size(2), self.embeddings.size(0))
        return self.embeddings
        
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        num_layers = 2
        if self.GPU is not None:
            return (autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_().cuda(self.GPU)),
                    autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_()).cuda(self.GPU))
        return (autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_()),
                autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_()))



class Action_Network(nn.Module):
    
    def __init__(
            self,
            lstm_units,
            hidden_units,
            out_dim,
            droprate,
            spans,
    ):
        super(Action_Network, self).__init__()

        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.out_dim = out_dim
        self.droprate = droprate
        self.spans = spans

        self.drop = nn.Dropout(droprate)
        self.span2hidden = nn.Linear(8 * spans * lstm_units, hidden_units)
        self.hidden2out = nn.Linear(hidden_units, out_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.span2hidden.bias.data.fill_(0)
        self.span2hidden.weight.data.uniform_(-initrange, initrange)
        self.hidden2out.bias.data.fill_(0)
        self.hidden2out.weight.data.uniform_(-initrange, initrange)

    def forward(self, embeddings, lefts, rights, test=False):
        span_out = []
        for left_index, right_index in zip(lefts, rights):
            embedding = embeddings[right_index] - embeddings[left_index - 1]
            span_out.append(embedding)

        hidden_input = torch.cat(span_out)
        hidden_input = hidden_input.view(hidden_input.size(0)*hidden_input.size(1))


        if self.droprate > 0 and not test:
            hidden_input = self.drop(hidden_input)
        hidden_output = nn.functional.relu(self.span2hidden(hidden_input.view(1, hidden_input.size(0))))
        scores = self.hidden2out(hidden_output)
        return scores




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

        self.struct = Action_Network(lstm_units, hidden_units, 
                                     struct_out, droprate, struct_spans)
        self.label = Action_Network(lstm_units, hidden_units, 
                                    label_out, droprate, label_spans)
        self.lstm = LSTM(word_count, tag_count,
                         word_dims, tag_dims,
                         lstm_units, droprate)
        
        if GPU is not None:
            self.struct.cuda(GPU)
            self.label.cuda(GPU)
            self.lstm.cuda(GPU)


    def save(self, filename):
        """
        Saves the model using Torch's save functionality.
        """
        torch.save({
            "struct_state_dict": self.struct.state_dict(),
            "label_state_dict": self.label.state_dict(),
            "lstm_state_dict": self.lstm.state_dict(),
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
        network.lstm.load_state_dict(checkpoint["lstm_state_dict"])
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

        f_loss = nn.CrossEntropyLoss()
        struct_optimizer = optim.Adam(network.struct.parameters(), lr = 0.0001)
        label_optimizer = optim.Adam(network.label.parameters(), lr = 0.0001)
        lstm_optimizer = optim.Adam(network.lstm.parameters(), lr = 0.0001)


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
        hidden = network.lstm.init_hidden(batch_size)

        
        lstm_prev_params = [x for x in network.lstm.lstm.parameters()]
        struct_prev_params = [x for x in network.struct.parameters()]
        

        for epoch in xrange(1, epochs + 1):
            print('........... epoch {} ...........'.format(epoch))

            total_cost = 0.0
            total_states = 0
            training_acc = FScore()

            np.random.shuffle(training_data)

            for b in xrange(num_batches):
                batch = training_data[(b * batch_size) : ((b + 1) * batch_size)]
                
                hidden = repackage_hidden(hidden)
                network.struct.zero_grad()
                network.label.zero_grad()
                network.lstm.zero_grad()

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

                    ## random UNKing ##
                    for (i, w) in enumerate(example['w']):
                        if w <= 2:
                            continue
                        freq = fm.word_freq_list[w]
                        drop_prob = unk_param / (unk_param + freq)
                        r = np.random.random()
                        if r < drop_prob:
                            example['w'][i] = 0

                    embeddings = network.lstm(
                        example['w'],
                        example['t'],
                    )

                    for (left, right), correct in example['struct_data'].items():
                        correct = autograd.Variable(torch.LongTensor([correct]))
                        if network.GPU is not None:
                            correct = correct.cuda(network.GPU)

                        scores = network.struct(embeddings, left, right)
                        loss = f_loss(scores, correct)
                        errors.append(loss)
                    total_states += len(example['struct_data'])

                    for (left, right), correct in example['label_data'].items():
                        correct = autograd.Variable(torch.LongTensor([correct]))
                        if network.GPU is not None:
                            correct = correct.cuda(network.GPU)

                        scores = network.label(embeddings, left, right)
                        loss = f_loss(scores, correct)
                        errors.append(loss)
                    total_states += len(example['label_data'])



                #batch_error = dynet.esum(errors)
                #total_cost += batch_error.scalar_value()
                #batch_error.backward()
                #trainer.update()

                batch_error = torch.sum(torch.cat(errors))
                batch_error.backward()
                struct_optimizer.step()
                label_optimizer.step()
                lstm_optimizer.step()

                total_cost += batch_error.data[0]

                mean_cost = (total_cost / total_states)
                
                


                """
                print()
                view = []
                struct_params = [x for x in network.struct.parameters()]
                for curr,prev in zip(struct_params, struct_prev_params):
                    if not curr.eq(prev):
                        print("STRUCT UPDATE HAPPENED")
                print("STRUCT UPDATE CHECK FINISHED")
                for x in struct_params:
                    if len(x.size()) == 2:
                        view.append(x[0][0])
                    else:
                        view.append(x[0])
                print(torch.cat(view))

                view = []
                lstm_params = [x for x in network.lstm.lstm.parameters()]
                for curr,prev in zip(lstm_params, lstm_prev_params):
                    if not curr.eq(prev):
                        print("LSTM UPDATE HAPPENED")
                print("LSTM UPDATE CHECK FINISHED")
                for x in lstm_params:
                    if len(x.size()) == 2:
                        view.append(x[0][0])
                    else:
                        view.append(x[0])
                print(torch.cat(view))
                #raw_input()
                struct_prev_params = struct_params
                lstm_prev_params = lstm_params
                """



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


