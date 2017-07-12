"""
Bi-LSTM network for span-based constituency parsing.
"""

from __future__ import print_function
from __future__ import division

import time
import random
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.functional as F
import torch.optim as optim
#from torch.nn.utils.rnn import pad_packed_sequence

from phrase_tree import PhraseTree, FScore
from features import FeatureMapper
from parser import Parser
from pprint import pprint


    

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
        self.batch_size = 1

        self.drop = nn.Dropout(droprate)
        self.activation = nn.functional.relu
        self.word_embed = nn.Embedding(word_count, word_dims)
        self.tag_embed = nn.Embedding(tag_count, tag_dims)

        self.lstm1 = nn.LSTM(word_dims + tag_dims, lstm_units, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * lstm_units, lstm_units, bidirectional=True)

        self.struct_hidden_W = nn.Linear(4 * struct_spans * lstm_units, hidden_units)
        self.struct_output_W = nn.Linear(hidden_units, struct_out)

        self.label_hidden_W = nn.Linear(4 * label_spans * lstm_units, hidden_units)
        self.label_output_W = nn.Linear(hidden_units, label_out)

        if GPU is not None:
            self.drop = self.drop.cuda(GPU)
            self.word_embed = self.word_embed.cuda(GPU)
            self.tag_embed = self.tag_embed.cuda(GPU)

            self.lstm1 = self.lstm1.cuda(GPU)
            self.lstm2 = self.lstm2.cuda(GPU)

            self.struct_hidden_W = self.struct_hidden_W.cuda(GPU)
            self.struct_output_W = self.struct_output_W.cuda(GPU)
            self.label_hidden_W = self.label_hidden_W.cuda(GPU)
            self.label_output_W = self.label_output_W.cuda(GPU)

        self.trainer = optim.Adadelta([x for x in self.word_embed.parameters()] +
                                      [x for x in self.tag_embed.parameters()] + 
                                      [x for x in self.lstm1.parameters()] +
                                      [x for x in self.lstm2.parameters()] +
                                      [x for x in self.struct_hidden_W.parameters()] + 
                                      [x for x in self.struct_output_W.parameters()] + 
                                      [x for x in self.label_hidden_W.parameters()] + 
                                      [x for x in self.label_output_W.parameters()],
                                      rho=0.99, eps=1e-7, weight_decay=1e-5)
        self.init_weights()


    def evaluate_mode(self):
        self.word_embed.eval()
        self.tag_embed.eval()
        self.lstm1.eval()
        self.lstm2.eval()
        self.struct_hidden_W.eval()
        self.struct_output_W.eval()
        self.label_hidden_W.eval()
        self.label_output_W.eval()

    
    def train_mode(self):
        self.word_embed.train()
        self.tag_embed.train()
        self.lstm1.train()
        self.lstm2.train()
        self.struct_hidden_W.train()
        self.struct_output_W.train()
        self.label_hidden_W.train()
        self.label_output_W.train()


    def init_weights(self):
        initrange = 0.01
        self.word_embed.weight.data.uniform_(-initrange, initrange)
        self.tag_embed.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, batch_size=1):
        num_layers = 1

        weight1 = next(self.lstm1.parameters()).data
        hidden1 = (autograd.Variable(weight1.new(num_layers * 2, 1, self.lstm_units).zero_()),
                   autograd.Variable(weight1.new(num_layers * 2, 1, self.lstm_units).zero_()))

        weight2 = next(self.lstm2.parameters()).data
        hidden2 = (autograd.Variable(weight2.new(num_layers * 2, 1, self.lstm_units).zero_()),
                   autograd.Variable(weight2.new(num_layers * 2, 1, self.lstm_units).zero_()))


        hidden1_batch = (autograd.Variable(weight1.new(num_layers * 2, batch_size, self.lstm_units).zero_()),
                         autograd.Variable(weight1.new(num_layers * 2, batch_size, self.lstm_units).zero_()))

        hidden2_batch = (autograd.Variable(weight2.new(num_layers * 2, batch_size, self.lstm_units).zero_()),
                         autograd.Variable(weight2.new(num_layers * 2, batch_size, self.lstm_units).zero_()))
        
        if self.GPU is not None:
            hidden1 = tuple((x.cuda(self.GPU) for x in hidden1))
            hidden2 = tuple((x.cuda(self.GPU) for x in hidden2))
            hidden1_batch = tuple((x.cuda(self.GPU) for x in hidden1_batch))
            hidden2_batch = tuple((x.cuda(self.GPU) for x in hidden2_batch))

        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden1_batch = hidden1_batch
        self.hidden2_batch = hidden2_batch


    def pad(self, tensor, length):
        if length - tensor.size(0) == 0:
            return tensor.view(tensor.size(0), 1)
        
        padding = autograd.Variable(torch.LongTensor(length - tensor.size(0), *tensor.size()[1:]).zero_())
        if self.GPU is not None:
            padding = padding.cuda(self.GPU)
        result = torch.cat([tensor, padding])
        result = result.view(result.size(0), 1)
        return result


    def evaluate_recurrent(self, word_inds, tag_inds, test=False):
        if test:
            self.evaluate_mode()
        else:
            self.train_mode()

        sentence = []
        wordvecs = self.word_embed(word_inds)
        tagvecs = self.tag_embed(tag_inds)

        #print(wordvecs.size())
        #print(tagvecs.size())
        sentence = torch.cat([wordvecs, tagvecs], 2)


        """
        for (w, t) in zip(word_inds, tag_inds):
            wordvec = self.word_embed(w)
            tagvec = self.tag_embed(t)
            vec = torch.cat([wordvec,tagvec], 1)
            sentence.append(vec)
        sentence = torch.cat(sentence)
        """

        #sentence = sentence.view(sentence.size(0), 1, sentence.size(1))
        
        if sentence.size(1) == 1:
            lstm_out1, hidden1 = self.lstm1(sentence, self.hidden1)
            fwd1, back1 = torch.split(lstm_out1, self.lstm_units, dim=2)
            if self.droprate > 0 and not test:
                lstm_out1 = self.drop(lstm_out1)

            lstm_out2, hidden2 = self.lstm2(lstm_out1, self.hidden2)
            fwd2, back2 = torch.split(lstm_out2, self.lstm_units, dim=2)
        
            fwd = torch.cat([fwd1, fwd2], 2)
            back = torch.cat([back1, back2],2)
        else:
            lstm_out1, hidden1 = self.lstm1(sentence, self.hidden1_batch)
            fwd1, back1 = torch.split(lstm_out1, self.lstm_units, dim=2)
            if self.droprate > 0 and not test:
                lstm_out1 = self.drop(lstm_out1)

            lstm_out2, hidden2 = self.lstm2(lstm_out1, self.hidden2_batch)
            fwd2, back2 = torch.split(lstm_out2, self.lstm_units, dim=2)
        
            fwd = torch.cat([fwd1, fwd2], 2)
            back = torch.cat([back1, back2],2)

        #return torch.cat([lstm_out1, lstm_out2], 2)            
        return fwd, back





    def evaluate_struct(self, fwd_out, back_out, indices, test=False):
        if test:
            self.evaluate_mode()
        else:
            self.train_mode()

        scores = []
        span_vecs = []


        for lefts, rights in indices:
            fwd_span_out = []
            for left_index, right_index in zip(lefts, rights):
                fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index - 1])
            fwd_span_vec = torch.cat(fwd_span_out)
            back_span_out = []
            for left_index, right_index in zip(lefts, rights):
                back_span_out.append(back_out[left_index] - back_out[right_index + 1])
            back_span_vec = torch.cat(back_span_out)
            
            vec = torch.cat([fwd_span_vec, back_span_vec])
            span_vecs.append(vec.view(1, vec.size(0)))
            #span_vecs.append(torch.cat([fwd_span_vec, back_span_vec]))
        hidden_input = torch.cat(span_vecs)
        #hidden_input = hidden_input.view(1, hidden_input.size(0))

        if self.droprate > 0 and not test:
            hidden_input = self.drop(hidden_input)
        
        hidden_output = self.activation(self.struct_hidden_W(hidden_input))
        scores = self.struct_output_W(hidden_output)
        return scores


    def evaluate_label(self, fwd_out, back_out, indices, test=False):
        if test:
            self.evaluate_mode()
        else:
            self.train_mode()

        scores = []
        span_vecs = []

        for lefts, rights in indices:
            fwd_span_out = []
            for left_index, right_index in zip(lefts, rights):
                fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index - 1])
            fwd_span_vec = torch.cat(fwd_span_out)
            back_span_out = []
            for left_index, right_index in zip(lefts, rights):
                back_span_out.append(back_out[left_index] - back_out[right_index + 1])
            back_span_vec = torch.cat(back_span_out)

            vec = torch.cat([fwd_span_vec, back_span_vec])
            span_vecs.append(vec.view(1, vec.size(0)))

        hidden_input = torch.cat(span_vecs)
        #hidden_input = hidden_input.view(1, hidden_input.size(0))

        if self.droprate > 0 and not test:
            hidden_input = self.drop(hidden_input)
        
        hidden_output = self.activation(self.label_hidden_W(hidden_input))
        scores = self.label_output_W(hidden_output)
        return scores


    def save(self, filename):
        """
        Saves the model using Torch's save functionality.
        """
        torch.save({
            "word_embed_state_dict": self.word_embed.state_dict(),
            "tag_embed_state_dict": self.tag_embed.state_dict(),
            "lstm1_state_dict": self.lstm1.state_dict(),
            "lstm2_state_dict": self.lstm2.state_dict(),
            "struct_hidden_W_state_dict": self.struct_hidden_W.state_dict(),
            "struct_output_W_state_dict": self.struct_output_W.state_dict(),
            "label_hidden_W_state_dict": self.label_hidden_W.state_dict(),
            "label_output_W_state_dict": self.label_output_W.state_dict(),


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
        network.word_embed.load_state_dict(checkpoint["word_embed_state_dict"])
        network.tag_embed.load_state_dict(checkpoint["tag_embed_state_dict"])
        network.lstm1.load_state_dict(checkpoint["lstm1_state_dict"])
        network.lstm2.load_state_dict(checkpoint["lstm2_state_dict"])
        network.struct_hidden_W.load_state_dict(checkpoint["struct_hidden_W_state_dict"])
        network.struct_output_W.load_state_dict(checkpoint["struct_output_W_state_dict"])
        network.label_hidden_W.load_state_dict(checkpoint["label_hidden_W_state_dict"])
        network.label_output_W.load_state_dict(checkpoint["label_output_W_state_dict"])
        network.init_hidden()

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
        
        f_loss = nn.CrossEntropyLoss(size_average=False)
        if GPU is not None:
            f_loss = f_loss.cuda(GPU)
        random.seed(1)
        torch.manual_seed(1)

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
        network.init_hidden(batch_size)
        
        for epoch in xrange(1, epochs + 1):
            print('........... epoch {} ...........'.format(epoch))

            total_cost = 0.0
            total_states = 0
            training_acc = FScore()

            np.random.shuffle(training_data)

            for b in xrange(num_batches):
                batch = training_data[(b * batch_size) : ((b + 1) * batch_size)]
                
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

                sorted_batch = []
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

                    """
                    fwd, back = network.evaluate_recurrent(
                        example['w'],
                        example['t'],
                    )
                    """
                    #w_batch.append(example['w'])
                    #t_batch.append(example['t'])

                    sorted_batch.append((example['w'], example['t'], example))
                    
                sorted_batch = sorted(sorted_batch, key=lambda x:x[0].size(0), reverse=True)
                max_length = sorted_batch[0][0].size(0)
                padded_w = torch.cat([network.pad(w, max_length) for w,_,_ in sorted_batch], 1)
                padded_t = torch.cat([network.pad(t, max_length) for _,t,_ in sorted_batch], 1)

                fwds,backs = network.evaluate_recurrent(padded_w, padded_t)
                
                for batch_index, (w, t, example) in enumerate(sorted_batch):
                    fwd = fwds[:, batch_index, :]
                    back = backs[:, batch_index, :]

                    indices,targets = [], []
                    for (left, right), correct in example['struct_data'].items():
                        indices.append((left,right))
                        targets.append(correct)
                    targets = autograd.Variable(torch.LongTensor(targets))
                    if network.GPU is not None:
                        targets = targets.cuda(network.GPU)
                    scores = network.evaluate_struct(fwd, back, indices)
                    for i in xrange(len(targets)):
                        score = scores[i]
                        target = targets[i]
                        loss = f_loss(score,target)
                        errors.append(loss)
                    total_states += len(example['struct_data'])

                    indices,targets = [], []
                    for (left, right), correct in example['label_data'].items():
                        indices.append((left,right))
                        targets.append(correct)
                    targets = autograd.Variable(torch.LongTensor(targets))
                    if network.GPU is not None:
                        targets = targets.cuda(network.GPU)
                    scores = network.evaluate_label(fwd, back, indices)
                    for i in xrange(len(targets)):
                        score = scores[i]
                        target = targets[i]
                        loss = f_loss(score,target)
                        errors.append(loss)
                    total_states += len(example['label_data'])

                batch_loss = torch.sum(torch.cat(errors))
                network.trainer.zero_grad()
                batch_loss.backward()
                network.trainer.step()
                
                total_cost += batch_loss.data[0]
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


