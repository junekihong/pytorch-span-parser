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

from constants import GPU
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

        self.word_embed = nn.Embedding(word_count, word_dims).cuda(GPU)
        self.tag_embed = nn.Embedding(tag_count, tag_dims).cuda(GPU)
        self.lstm = nn.LSTM(word_dims + tag_dims,
                            lstm_units,
                            num_layers=2,
                            bidirectional=True).cuda(GPU)
        


        self.span2hidden = nn.Linear(2 * spans * lstm_units, hidden_units).cuda(GPU)
        self.hidden2out = nn.Linear(hidden_units, out_dim).cuda(GPU)
        self.hidden_linear = self.init_hidden_linear()
        self.hidden_lstm = self.init_hidden_lstm()

    def init_hidden_linear(self):
        return autograd.Variable(torch.zeros(1, 1, self.hidden_units)).cuda(GPU)

    def init_hidden_lstm(self):
        return (autograd.Variable(torch.zeros(1, 1, self.lstm_units).cuda(GPU)),
                autograd.Variable(torch.zeros(1, 1, self.lstm_units).cuda(GPU)),
                autograd.Variable(torch.zeros(1, 1, self.lstm_units).cuda(GPU)),
                autograd.Variable(torch.zeros(1, 1, self.lstm_units).cuda(GPU)))

    def evaluate_recurrent(self, word_inds, tag_inds, test=False):
        sentence = []
        wordvecs = self.word_embed(word_inds)
        tagvecs = self.tag_embed(tag_inds)
        wordvecs = wordvecs.view(self.word_dims, len(word_inds))
        tagvecs = tagvecs.view(self.tag_dims, len(tag_inds))

        sentence = torch.cat([wordvecs, tagvecs])
        sentence = sentence.view(-1, 1, len(sentence))
        lstm_out, self.hidden_lstm = self.lstm(sentence)
        self.embeddings = lstm_out
        return lstm_out


    def forward(self, lefts, rights, test=False):
        #print("reminder, the input size for the fully connected layer is:", 2*self.spans*self.lstm_units)
        #print("that is: {} spans, {} lstm_units, * 2".format(self.spans, self.lstm_units))

        span_out = []
        for left_index, right_index in zip(lefts, rights):
            embedding = self.embeddings[right_index] - self.embeddings[left_index - 1]
            span_out.append(embedding.view(embedding.size()[1]))
            #span_out.append(embedding)

        #print(len(span_out))
        #print(span_out[0].size())
        hidden_input = torch.cat(span_out)
        hidden_input = hidden_input.view((1, len(hidden_input)))
        
        #print("num elements of span_out:", len(span_out), span_out[0].size())

        #print("hidden input:")
        #print(hidden_input.size())
        
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

        self.struct = SubNetwork(word_count, tag_count, 
                                 word_dims, tag_dims, 
                                 lstm_units, hidden_units, struct_out, droprate, struct_spans)
        self.label = SubNetwork(word_count, tag_count,
                                word_dims, tag_dims,
                                lstm_units, hidden_units, label_out, droprate, label_spans)



    def save(self, filename):
        """
        Appends architecture hyperparameters to end of dynet model file.
        """
        """
        self.model.save(filename)

        with open(filename, 'a') as f:
            f.write('\n')
            f.write('word_count = {}\n'.format(self.word_count))
            f.write('tag_count = {}\n'.format(self.tag_count))
            f.write('word_dims = {}\n'.format(self.word_dims))
            f.write('tag_dims = {}\n'.format(self.tag_dims))
            f.write('lstm_units = {}\n'.format(self.lstm_units))
            f.write('hidden_units = {}\n'.format(self.hidden_units))
            f.write('struct_out = {}\n'.format(self.struct_out))
            f.write('label_out = {}\n'.format(self.label_out))
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
    def load(filename):
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
        )

        struct_loss_function = nn.NLLLoss().cuda(GPU)
        label_loss_function = nn.NLLLoss().cuda(GPU)
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
                #network.zero_grad()
                #network.hidden = network.init_hidden()

                network.struct.hidden_linear = network.struct.init_hidden_linear()
                network.struct.hidden_lstm = network.struct.init_hidden_lstm()
                network.label.hidden_linear = network.label.init_hidden_linear()
                network.label.hidden_lstm = network.label.init_hidden_lstm()


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
                total_loss = autograd.Variable(torch.FloatTensor([0])).cuda(GPU)

                for example in batch:
                    network.struct.zero_grad()
                    network.label.zero_grad()
                    network.struct.hidden_lstm = network.struct.init_hidden_lstm()
                    network.struct.hidden_linear = network.struct.init_hidden_linear()
                    network.label.hidden_lstm = network.label.init_hidden_lstm()
                    network.label.hidden_linear = network.label.init_hidden_linear()


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
                        struct_scoring = network.struct.forward(left, right)
                        struct_scores.append(struct_scoring)
                        struct_corrects.append(correct)

                    struct_scores = torch.cat(struct_scores)
                    struct_corrects = autograd.Variable(torch.LongTensor(struct_corrects)).cuda(GPU)
                    loss = struct_loss_function(struct_scores, struct_corrects)
                    total_loss += loss

                    loss.backward()
                    struct_trainer.step()


                    # Temporary: Not train the labeler. Just the struct predicter.
                    """
                    label_scores, label_corrects = [],[]
                    for (left, right), correct in example['label_data'].items():
                        label_scoring = network.label.forward(left, right)
                        label_scores.append(label_scoring)
                        label_corrects.append(correct)

                    label_scores = torch.cat(label_scores)
                    label_corrects = autograd.Variable(torch.LongTensor(label_corrects)).cuda(GPU)
                    loss = label_loss_function(label_scores, label_corrects)
                    total_loss += loss

                    loss.backward()
                    label_trainer.step()
                    """
                    

                    """
                    print(struct_scores.size())
                    print(label_scores.size())
                    scores = torch.cat([struct_scores, label_scores])
                    corrects = torch.cat([struct_corrects, label_corrects])
                    loss = loss_function(scores, corrects)
                    total_loss += loss
                    """

                    total_states += len(example['struct_data'])
                    total_states += len(example['label_data'])


                #batch_error = dynet.esum(errors)
                #total_cost += batch_error.scalar_value()
                #batch_error.backward()
                #trainer.update()


                mean_cost = total_cost / total_states

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


