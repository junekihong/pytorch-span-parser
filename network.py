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

from phrase_tree import PhraseTree, FScore
from features import FeatureMapper
from parser import Parser
torch.manual_seed(1)
from pprint import pprint


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

        self.drop = nn.Dropout(droprate)
        self.word_embed = nn.Embedding(word_count, word_dims)
        self.tag_embed = nn.Embedding(tag_count, tag_dims)
        self.lstm1 = nn.LSTM(word_dims + tag_dims,
                            lstm_units,
                            num_layers=1,
                            #dropout=droprate,
                            bidirectional=True)
        self.lstm2 = nn.LSTM(2 * lstm_units,
                             lstm_units,
                             num_layers=1,
                             #dropout=droprate,
                             bidirectional=True)
        
        if GPU is not None:
            self.word_embed = self.word_embed.cuda(GPU)
            self.tag_embed = self.tag_embed.cuda(GPU)
            self.lstm = self.lstm.cuda(GPU)
        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.word_embed.weight.data.uniform_(-initrange, initrange)
        self.tag_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, word_inds, tag_inds, test=False):
        # Setting this to true or false will enable/disable dropout.
        if test:
            self.eval()
        else:
            self.train()

        sentence = []
        for (w, t) in zip(word_inds, tag_inds):
            wordvec = self.word_embed(w)
            tagvec = self.tag_embed(t)
            vec = torch.cat([wordvec,tagvec], 1)
            sentence.append(vec)
        sentence = torch.cat(sentence)
        sentence = sentence.view(sentence.size(0), 1, sentence.size(1))

        lstm_out1, hidden1 = self.lstm1(sentence, self.hidden1)

        if self.droprate > 0 and not test:
            lstm_out1 = self.drop(lstm_out1)

        lstm_out2, hidden2 = self.lstm2(lstm_out1, hidden1)
        output = torch.cat([lstm_out1, lstm_out2], 2)
        return output
        
    def init_hidden(self, batch_size=1):
        weight = next(self.parameters()).data
        num_layers = 1
        hidden1 = (autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_()),
                   autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_()))
        if self.GPU is not None:
            hidden1 = (x.cuda(self.GPU) for x in hidden1)
        self.hidden1 = hidden1



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
        self.span2hidden = nn.Linear(4 * spans * lstm_units, hidden_units)
        self.hidden2out = nn.Linear(hidden_units, out_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.span2hidden.bias.data.fill_(0)
        self.span2hidden.weight.data.uniform_(-initrange, initrange)
        self.hidden2out.bias.data.fill_(0)
        self.hidden2out.weight.data.uniform_(-initrange, initrange)

    def forward(self, embeddings, indices, test=False):

        # Setting test here will enable/disable dropout. Probably not necessary to do here.
        if test:
            self.eval()
        else:
            self.train()

        #fwd_out,back_out = torch.split(embeddings, self.lstm_units, dim=2)

        hidden_inputs = []
        for lefts,rights in indices:
            span_out = []
            """
            fwd_span_out = []
            for left_index, right_index in zip(lefts, rights):
                fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index - 1])
            fwd_span_vec = torch.cat(fwd_span_out)
            back_span_out = []
            for left_index, right_index in zip(lefts, rights):
                back_span_out.append(back_out[left_index] - back_out[right_index + 1])
            back_span_vec = torch.cat(back_span_out)
            hidden_input = torch.cat([fwd_span_vec, back_span_vec])
            hidden_input = hidden_input.view(1, hidden_input.size(0) * hidden_input.size(1))
            hidden_inputs.append(hidden_input)
            """

            for left_index, right_index in zip(lefts, rights):
                embedding = embeddings[right_index] - embeddings[left_index - 1]
                span_out.append(embedding.view(embedding.size(1)))
                
            hidden_input = torch.cat(span_out)
            hidden_input = hidden_input.view(1, hidden_input.size(0))

            hidden_inputs.append(hidden_input)


        hidden_inputs = torch.cat(hidden_inputs)

        if self.droprate > 0 and not test:
            hidden_inputs = self.drop(hidden_inputs)

        hidden_outputs = self.span2hidden(hidden_inputs)
        hidden_outputs = nn.functional.relu(hidden_outputs)
        scores = self.hidden2out(hidden_outputs)
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



        self.drop = nn.Dropout(droprate)
        self.activation = nn.functional.relu
        self.word_embed = nn.Embedding(word_count, word_dims)
        self.tag_embed = nn.Embedding(tag_count, tag_dims)


        """
        self.fwd_lstm1 = nn.LSTM(word_dims + tag_dims, lstm_units)
        self.back_lstm1 = nn.LSTM(word_dims + tag_dims, lstm_units)
        self.fwd_lstm2 = nn.LSTM(2 * lstm_units, lstm_units)
        self.back_lstm2 = nn.LSTM(2 * lstm_units, lstm_units)
        """


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


            """
            self.fwd_lstm1 = self.fwd_lstm1.cuda(GPU)
            self.back_lstm1 = self.back_lstm1.cuda(GPU)
            self.fwd_lstm2 = self.fwd_lstm2.cuda(GPU)
            self.back_lstm2 = self.back_lstm2.cuda(GPU)
            """


            self.lstm1 = self.lstm1.cuda(GPU)
            self.lstm2 = self.lstm2.cuda(GPU)

            self.struct_hidden_W = self.struct_hidden_W.cuda(GPU)
            self.struct_output_W = self.struct_output_W.cuda(GPU)
            self.label_hidden_W = self.label_hidden_W.cuda(GPU)
            self.label_output_W = self.label_output_W.cuda(GPU)


        self.trainer = optim.Adadelta([x for x in self.word_embed.parameters()] +
                                      [x for x in self.tag_embed.parameters()] + 
                                      #[x for x in self.fwd_lstm1.parameters()] + 
                                      #[x for x in self.back_lstm1.parameters()] +
                                      #[x for x in self.fwd_lstm2.parameters()] + 
                                      #[x for x in self.back_lstm2.parameters()] + 
                                      [x for x in self.lstm1.parameters()] +
                                      [x for x in self.lstm2.parameters()] +
                                      [x for x in self.struct_hidden_W.parameters()] + 
                                      [x for x in self.struct_output_W.parameters()] + 
                                      [x for x in self.label_hidden_W.parameters()] + 
                                      [x for x in self.label_output_W.parameters()],
                                      rho=0.99, eps=1e-7)



        if GPU is not None:
            self.struct.cuda(GPU)
            self.label.cuda(GPU)
            self.lstm.cuda(GPU)

        self.init_weights()


    def init_weights(self):
        initrange = 0.01
        self.word_embed.weight.data.uniform_(-initrange, initrange)
        self.tag_embed.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self):
        weight = next(self.lstm1.parameters()).data
        num_layers = 1
        batch_size = 1
        hidden = (autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_()),
                  autograd.Variable(weight.new(num_layers * 2, batch_size, self.lstm_units).zero_()))
        if self.GPU is not None:
            hidden = tuple((x.cuda(self.GPU) for x in hidden))
        self.hidden = hidden
        

    def evaluate_recurrent(self, word_inds, tag_inds, test=False):
        sentence = []
        for (w, t) in zip(word_inds, tag_inds):
            wordvec = self.word_embed(w)
            tagvec = self.tag_embed(t)
            vec = torch.cat([wordvec,tagvec], 1)
            sentence.append(vec)
        sentence = torch.cat(sentence)
        sentence = sentence.view(sentence.size(0), 1, sentence.size(1))
        
        lstm_out1, hidden1 = self.lstm1(sentence, self.hidden)
        
        if self.droprate > 0 and not test:
            lstm_out1 = self.drop(lstm_out1)

        lstm_out2, hidden2 = self.lstm2(lstm_out1, hidden1)
        return torch.cat([lstm_out1, lstm_out2], 2)


    def evaluate_struct(self, outputs, lefts, rights, test=False):
        span_out = []
        for left_index, right_index in zip(lefts, rights):
            span_out.append(outputs[right_index] - outputs[left_index - 1])
        span_vec = torch.cat(span_out, 1)
        
        hidden_input = span_vec
        if self.droprate > 0 and not test:
            hidden_input = self.drop(hidden_input)
            
        hidden_output = self.activation(self.struct_hidden_W(hidden_input))
        scores = self.struct_output_W(hidden_output)
        return scores


    def evaluate_label(self, outputs, lefts, rights, test=False):
        span_out = []
        for left_index, right_index in zip(lefts, rights):
            span_out.append(outputs[right_index] - outputs[left_index - 1])
        span_vec = torch.cat(span_out, 1)

        hidden_input = span_vec
        if self.droprate > 0 and not test:
            hidden_input = self.drop(hidden_input)

        hidden_output = self.label_hidden_W(hidden_input)
        hidden_output = self.activation(hidden_output)
        scores = self.label_output_W(hidden_output)
        return scores


    def save(self, filename):
        """
        Saves the model using Torch's save functionality.
        """
        torch.save({
            #"struct_state_dict": self.struct.state_dict(),
            #"label_state_dict": self.label.state_dict(),
            #"lstm_state_dict": self.lstm.state_dict(),


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
        """
        network.struct.load_state_dict(checkpoint["struct_state_dict"])
        network.label.load_state_dict(checkpoint["label_state_dict"])
        network.lstm.load_state_dict(checkpoint["lstm_state_dict"])
        """

        network.word_embed.load_state_dict(checkpoint["word_embed_state_dict"])
        network.tag_embed.load_state_dict(checkpoint["tag_embed_state_dict"])
        network.lstm1.load_state_dict(checkpoint["lstm1_state_dict"])
        network.lstm2.load_state_dict(checkpoint["lstm2_state_dict"])
        network.struct_hidden_W.load_state_dict(checkpoint["struct_hidden_W_state_dict"])
        network.struct_output_W.load_state_dict(checkpoint["struct_output_W_state_dict"])
        network.label_hidden_W.load_state_dict(checkpoint["label_hidden_W_state_dict"])
        network.label_output_W.load_state_dict(checkpoint["label_output_W_state_dict"])

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
        if GPU is not None:
            f_loss = f_loss.cuda(GPU)

        """
        optimizer = optim.Adadelta([x for x in network.struct.parameters()] + 
                                   [x for x in network.label.parameters()] + 
                                   [x for x in network.lstm.parameters()], 
                                   rho=0.99,
                                   eps=1e-7)
        """

        """
        struct_optimizer = optim.Adadelta([x for x in network.struct.parameters()] + 
                                          [x for x in network.lstm.parameters()], 
                                          rho=0.99,
                                          eps=1e-7)
        label_optimizer = optim.Adadelta([x for x in network.label.parameters()] + 
                                         [x for x in network.lstm.parameters()], 
                                         rho=0.99,
                                         eps=1e-7)
        """


        random.seed(1)

        #optimizer = optim.Adam(network.struct.parameters(), lr = 0.0001)
        #optimizer = optim.Adam(network.label.parameters(), lr = 0.0001)
        #optimizer = optim.Adam(network.lstm.parameters(), lr = 0.0001)

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
        #network.lstm.init_hidden()
        network.init_hidden()
        
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

                    #embeddings = network.lstm(
                    embeddings = network.evaluate_recurrent(
                        example['w'],
                        example['t'],
                    )

                    #indices,targets = [],[]
                    for (left, right), correct in example['struct_data'].items():
                        scores = network.evaluate_struct(embeddings, left, right)
                        target = autograd.Variable(torch.LongTensor([correct]))
                        if network.GPU is not None:
                            target = target.cuda(network.GPU)

                        #print(scores)
                        #print(target)
                        loss = f_loss(scores, target)
                        errors.append(loss)

                        """
                        indices.append((left,right))
                        targets.append(correct)
                    scores = network.struct(embeddings, indices)
                    targets = autograd.Variable(torch.LongTensor(targets))
                    if network.GPU is not None:
                        targets = targets.cuda(network.GPU)
                    loss = f_loss(scores, targets)
                    errors.append(loss)
                        """
                    total_states += len(example['struct_data'])


                    #indices,targets = [],[]
                    for (left, right), correct in example['label_data'].items():
                        scores = network.evaluate_label(embeddings, left, right)
                        target = autograd.Variable(torch.LongTensor([correct]))
                        if network.GPU is not None:
                            target = target.cuda(network.GPU)

                        loss = f_loss(scores, target)
                        errors.append(loss)
                        
                        """
                        indices.append((left,right))
                        targets.append(correct)
                    scores = network.label(embeddings, indices)
                    targets = autograd.Variable(torch.LongTensor(targets))
                    if network.GPU is not None:
                        targets = targets.cuda(network.GPU)
                    loss = f_loss(scores, targets)
                    errors.append(loss)
                        """

                    total_states += len(example['label_data'])


                """
                label_loss = torch.sum(torch.cat(label_errors))
                label_optimizer.zero_grad()
                label_loss.backward()
                label_optimizer.step()
                """

                batch_loss = torch.sum(torch.cat(errors))
                network.trainer.zero_grad()
                batch_loss.backward()
                network.trainer.step()

                


                total_cost += batch_loss.data[0]
                #total_cost += struct_loss.data[0] # + label_loss.data[0]
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


