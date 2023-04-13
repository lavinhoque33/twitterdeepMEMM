from __future__ import division
import subprocess
import argparse
import sys

import nltk
import scipy
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

# Load training and testing data
from torch.nn.utils.rnn import pack_padded_sequence


def load_data(path, lowercase=True):
    sents = []
    tags = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.read().splitlines():
            sent = []
            tag = []
            for pair in line.split('####')[1].split(' '):
                tn, tg = pair.rsplit('=', 1)
                if lowercase:
                    sent.append(tn.lower())
                else:
                    sent.append(tn)
                tag.append(tg)
            sents.append(sent)
            tags.append(tag)
    return sents, tags

def reload_data(path, sentwords, tagwords):
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(sentwords)):
            t1 = ' '.join(sentwords[i])
            t2 = ""
            for j in range(len(sentwords[i])):
                t2 += sentwords[i][j]+'='+tagwords[i][j]
                if(j != len(sentwords[i])-1):
                    t2 += ' '
            t = t1+'####'+t2+'\n'
            f.write(t)


class Classifier(object):
    def __init__(self):
        pass

    def train(self):
        raise NotImplementedError("Train method not implemented")

    def inference(self):
        raise NotImplementedError("Inference method not implemented")


class MEMM(nn.Module):
    def __init__(self, w2vweights, hidden_layer_size):
        super(MEMM, self).__init__()
        # set up the model
        self.w2vweights = w2vweights

        self.word_embedding_layer = nn.Embedding.from_pretrained(w2vweights)

        self.word_embedding_size = self.word_embedding_layer.embedding_dim
        self.tag_embedding_size = self.word_embedding_layer.embedding_dim
        self.hidden_layer_size = hidden_layer_size
        self.output_class_size = 5
        self.vocab_size = self.word_embedding_layer.num_embeddings


        self.tag_embedding_layer = nn.Embedding(self.output_class_size, self.word_embedding_size)
        self.fc1 = nn.Linear(self.word_embedding_size + self.word_embedding_size, self.hidden_layer_size)
        self.relu_activation = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_layer_size, self.output_class_size)
        self.relu_activation2 = nn.ReLU()

        self.dropout = nn.Dropout(0.1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_layer_size // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_layer_size // 2)))

    def forward_pass(self, current_word, previous_tag):
        word_embeddings = self.word_embedding_layer(current_word)
        tag_embeddings = self.tag_embedding_layer(previous_tag)
        fc_input = torch.cat((word_embeddings, tag_embeddings), 1)
        out1 = self.fc1(fc_input)
        dropout_output = self.dropout(out1)
        activation1 = self.relu_activation(dropout_output)
        out2 = self.fc2(activation1)
        return out2

    def train(self, data_lex, data_y, optimizer, start_tag, train=False):
        total_count = 0
        accuracy = 0.0
        sum_correct_result = 0
        f1_sc = 0.0
        total_loss_list = []

        for i in np.arange(len(data_lex)):
            batch_sentences = data_lex[i]
            tag_vectors = data_y[i]

            total_count += len(batch_sentences)

            # create a batch from the sentence
            input_sentence = autograd.Variable(torch.from_numpy(batch_sentences).long())
            # the previous tags
            previous_tag_vector = np.asarray([start_tag] + list(tag_vectors[:-1]))
            input_targets = autograd.Variable(torch.from_numpy(previous_tag_vector).long())
            # the current tags
            output_targets = autograd.Variable(torch.from_numpy(tag_vectors).long())
            # training output
            output = self.forward_pass(input_sentence, input_targets)
            # calculate how many are correct
            max_index = output.max(dim=1)[1]
            sum_correct_result = sum_correct_result + ((max_index == output_targets).sum().data.numpy())
            tp = ((output_targets == max_index) & (output_targets != 0)).sum().data.numpy()
            fp = ((output_targets != max_index) & (
                        (output_targets != 0) & (max_index != 0) | (output_targets == 0))).sum().data.numpy()
            fn = ((output_targets != max_index) & (
                        (output_targets != 0) & (max_index != 0) | (max_index == 0))).sum().data.numpy()
            f1 = (tp) / (tp + 0.5 * (fp + fn))
            f1_sc += f1

            if train == True:
                # loss
                we = torch.from_numpy(np.array([1.0, 2.0, 2.0, 2.0, 0.0])).float()
                l = F.cross_entropy(input=output, target=output_targets, weight=we)

                total_loss_list.append(l.data.mean())
                # train
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

        if train == True:
            current_average_loss = np.mean(total_loss_list)
        accuracy = 100 * (sum_correct_result / total_count)
        f1_score = f1_sc / len(data_lex)

        if train == True:
            return current_average_loss, accuracy, f1_score
        else:
            return accuracy, f1_score

    def forward_inference(self, current_word, previous_tag):
        input_word = autograd.Variable(torch.from_numpy(np.asarray([current_word])).long())
        input_targets = autograd.Variable(torch.from_numpy(np.asarray([previous_tag])).long())
        output = self.forward_pass(input_word, input_targets)
        softmax_probs = F.softmax(output)
        return softmax_probs

    # Viterbi algorithm
    def inference(self, data_lex, num_words, num_states, start_tag):
        num_states = num_states - 1

        # table to store the probability of the most likely path so far
        table_1 = np.zeros((num_words, num_states))
        # table to store the backpointers of the most likely path so far
        table_2 = np.zeros((num_words, num_states))

        # initialization

        # get the first word from the data_lex (list of sentences) and make a batch that has all the states, all the initial states, and all the first words in 3 separate vectors to pass in for batch inference from the MLP
        probabilities = self.forward_inference(data_lex[0], start_tag).data.numpy()

        max_tag = np.argmax(probabilities, 1)[0]

        # for all states, table_1(1, s) = p(s | s_o, x1..x_m)
        # fill in the table using the start state given random initial and the current word
        for i in range(num_states):
            table_1[0, i] = probabilities[0, i]
            table_2[0, i] = 0

        for j in range(1, num_words):
            # create a transition matrix that stores the probability of each state given previous state and current word
            transition_matrix = []
            for l in range(num_states):
                output = self.forward_inference(data_lex[j], l)
                transition_matrix.append(output.data.numpy()[0])
            transition_matrix = np.asarray(transition_matrix)

            # i = current states
            for i in range(num_states):
                max_prob = 0
                max_index = 0
                # k = old states
                for k in range(num_states):
                    current_prob = table_1[j - 1, k] * transition_matrix[k, i]
                    if current_prob > max_prob:
                        max_prob = current_prob
                        max_index = k

                        if max_index == 127:
                            print(max_prob)
                            print(k)
                            print(i)
                    max_prob = max(table_1[j - 1, k] * transition_matrix[k, i], max_prob)
                # max over probability of a state in previous row * probability of going to this current state given that previous state
                table_1[j, i] = max_prob
                table_2[j, i] = max_index

        # for the last word, what is the state with the best probability
        maxPreviousTag = np.argmax(table_1[num_words - 1, :])

        reverse_final_tag_sequence = []
        reverse_final_tag_sequence.append(maxPreviousTag)
        for i in reversed(range(1, num_words)):
            maxPreviousTag = int(table_2[i, int(maxPreviousTag)])
            reverse_final_tag_sequence.append(maxPreviousTag)

        final_tag_sequence = []
        for i in reversed(reverse_final_tag_sequence):
            final_tag_sequence.append(i)

        return final_tag_sequence

class MEMM2(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, hidden_layer_size, hidden_layer_size2, n_layers):
        super(MEMM2, self).__init__()
        # set up the model
        self.word_embedding_size = word_embedding_size
        self.tag_embedding_size = word_embedding_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_size2 = hidden_layer_size2
        self.n_layers = n_layers
        self.output_class_size = 5
        self.vocab_size = vocab_size

        self.word_embedding_layer = nn.Embedding(self.vocab_size, self.word_embedding_size)
        self.tag_embedding_layer = nn.Embedding(self.output_class_size, self.word_embedding_size)
        # lstm layer
        self.lstm = nn.LSTM(input_size=self.word_embedding_size,
                            hidden_size=self.hidden_layer_size,
                            num_layers=self.n_layers,
                            bidirectional=True,
                            batch_first=False)
        self.fc1 = nn.Linear(self.hidden_layer_size * 2, self.hidden_layer_size2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_layer_size2, self.output_class_size)

        self.dropout = nn.Dropout(0.1)

    def forward_pass(self, current_word, previous_tag):
        word_embeddings = self.word_embedding_layer(current_word)
        tag_embeddings = self.tag_embedding_layer(previous_tag)
        # sz = torch.full([current_word.shape[0]], 300)
        # packed sequence
        # packed_embedded = pack_padded_sequence(word_embeddings, sz, batch_first=True)  # unpad
        # print(packed_embedded)
        packed_output, (hidden, cell) = self.lstm(word_embeddings.unsqueeze(0))
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # output, output_lengths = pad_packed_sequence(packed_output)  # pad the sequence to the max length in the batch

        rel = self.relu(cat)
        dense1 = self.fc1(rel)

        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds

    def train(self, data_lex, data_y, optimizer, start_tag, train=False):
        total_count = 0
        accuracy = 0.0
        sum_correct_result = 0
        f1_sc = 0.0
        total_loss_list = []

        for i in np.arange(len(data_lex)):
            batch_sentences = data_lex[i]
            tag_vectors = data_y[i]

            total_count += len(batch_sentences)

            # create a batch from the sentence
            input_sentence = autograd.Variable(torch.from_numpy(batch_sentences).long())
            # the previous tags
            previous_tag_vector = np.asarray([start_tag] + list(tag_vectors[:-1]))
            input_targets = autograd.Variable(torch.from_numpy(previous_tag_vector).long())
            # the current tags
            output_targets = autograd.Variable(torch.from_numpy(tag_vectors).long())
            # training output
            output = self.forward_pass(input_sentence, input_targets)
            # calculate how many are correct
            max_index = output.max(dim=1)[1]
            sum_correct_result = sum_correct_result + ((max_index == output_targets).sum().data.numpy())
            tp = ((output_targets == max_index) & (output_targets != 0)).sum().data.numpy()
            fp = ((output_targets != max_index) & (
                        (output_targets != 0) & (max_index != 0) | (output_targets == 0))).sum().data.numpy()
            fn = ((output_targets != max_index) & (
                        (output_targets != 0) & (max_index != 0) | (max_index == 0))).sum().data.numpy()
            f1 = (tp) / (tp + 0.5 * (fp + fn))
            f1_sc += f1

            if train == True:
                # loss
                we = torch.from_numpy(np.array([1.0, 2.0, 2.0, 2.0, 0.0])).float()
                l = F.cross_entropy(input=output, target=output_targets, weight=we)

                total_loss_list.append(l.data.mean())
                # train
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

        if train == True:
            current_average_loss = np.mean(total_loss_list)
        accuracy = 100 * (sum_correct_result / total_count)
        f1_score = f1_sc / len(data_lex)

        if train == True:
            return current_average_loss, accuracy, f1_score
        else:
            return accuracy, f1_score

    def forward_inference(self, current_word, previous_tag):
        input_word = autograd.Variable(torch.from_numpy(np.asarray([current_word])).long())
        input_targets = autograd.Variable(torch.from_numpy(np.asarray([previous_tag])).long())
        output = self.forward_pass(input_word, input_targets)
        softmax_probs = F.softmax(output)
        return softmax_probs

    # Viterbi algorithm
    def inference(self, data_lex, num_words, num_states, start_tag):
        num_states = num_states - 1

        # table to store the probability of the most likely path so far
        table_1 = np.zeros((num_words, num_states))
        # table to store the backpointers of the most likely path so far
        table_2 = np.zeros((num_words, num_states))

        # initialization

        # get the first word from the data_lex (list of sentences) and make a batch that has all the states, all the initial states, and all the first words in 3 separate vectors to pass in for batch inference from the MLP
        probabilities = self.forward_inference(data_lex[0], start_tag).data.numpy()

        max_tag = np.argmax(probabilities, 1)[0]

        # for all states, table_1(1, s) = p(s | s_o, x1..x_m)
        # fill in the table using the start state given random initial and the current word
        for i in range(num_states):
            table_1[0, i] = probabilities[0, i]
            table_2[0, i] = 0


        for j in range(1, num_words):
            # create a transition matrix that stores the probability of each state given previous state and current word
            transition_matrix = []
            for l in range(num_states):
                output = self.forward_inference(data_lex[j], l)
                transition_matrix.append(output.data.numpy()[0])
            transition_matrix = np.asarray(transition_matrix)

            # i = current states
            for i in range(num_states):
                max_prob = 0
                max_index = 0
                # k = old states
                for k in range(num_states):
                    current_prob = table_1[j-1, k] * transition_matrix[k, i]
                    if current_prob > max_prob:
                        max_prob = current_prob
                        max_index = k

                        if max_index == 127:
                            print(max_prob)
                            print(k)
                            print(i)
                    max_prob = max(table_1[j-1, k] * transition_matrix[k, i], max_prob)
                # max over probability of a state in previous row * probability of going to this current state given that previous state
                table_1[j, i] = max_prob
                table_2[j, i] = max_index

        # for the last word, what is the state with the best probability
        maxPreviousTag = np.argmax(table_1[num_words - 1, :])

        reverse_final_tag_sequence = []
        reverse_final_tag_sequence.append(maxPreviousTag)
        for i in reversed(range(1, num_words)):
            maxPreviousTag = int(table_2[i, int(maxPreviousTag)])
            reverse_final_tag_sequence.append(maxPreviousTag)

        final_tag_sequence = []
        for i in reversed(reverse_final_tag_sequence):
            final_tag_sequence.append(i)


        return final_tag_sequence

class MEMM3(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, hidden_layer_size):
        super(MEMM3, self).__init__()
        # set up the model
        self.word_embedding_size = word_embedding_size
        self.tag_embedding_size = word_embedding_size
        self.hidden_layer_size = hidden_layer_size
        self.output_class_size = 5
        self.vocab_size = vocab_size

        self.word_embedding_layer = nn.Embedding(self.vocab_size, self.word_embedding_size)
        self.tag_embedding_layer = nn.Embedding(self.output_class_size, self.word_embedding_size)
        self.fc1 = nn.Linear(self.word_embedding_size + self.word_embedding_size, self.hidden_layer_size)
        self.relu_activation = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_layer_size, self.output_class_size)
        self.relu_activation2 = nn.ReLU()

        self.dropout = nn.Dropout(0.1)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_layer_size // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_layer_size // 2)))

    def forward_pass(self, current_word, previous_tag):
        word_embeddings = self.word_embedding_layer(current_word)
        tag_embeddings = self.tag_embedding_layer(previous_tag)
        fc_input = torch.cat((word_embeddings, tag_embeddings), 1)
        out1 = self.fc1(fc_input)
        dropout_output = self.dropout(out1)
        activation1 = self.relu_activation(dropout_output)
        out2 = self.fc2(activation1)
        return out2

    def train(self, data_lex, data_y, optimizer, start_tag, train=False):
        total_count = 0
        accuracy = 0.0
        sum_correct_result = 0
        f1_sc = 0.0
        total_loss_list = []

        for i in np.arange(len(data_lex)):
            batch_sentences = data_lex[i]
            tag_vectors = data_y[i]

            total_count += len(batch_sentences)

            # create a batch from the sentence
            input_sentence = autograd.Variable(torch.from_numpy(batch_sentences).long())
            # the previous tags
            previous_tag_vector = np.asarray([start_tag] + list(tag_vectors[:-1]))
            input_targets = autograd.Variable(torch.from_numpy(previous_tag_vector).long())
            # the current tags
            output_targets = autograd.Variable(torch.from_numpy(tag_vectors).long())
            # training output
            output = self.forward_pass(input_sentence, input_targets)
            # calculate how many are correct
            max_index = output.max(dim = 1)[1]
            sum_correct_result = sum_correct_result + ((max_index == output_targets).sum().data.numpy())
            tp = ((output_targets == max_index) & (output_targets != 0)).sum().data.numpy()
            fp = ((output_targets != max_index) & ((output_targets != 0) & (max_index != 0) | (output_targets == 0))).sum().data.numpy()
            fn = ((output_targets != max_index) & ((output_targets != 0) & (max_index != 0) | (max_index == 0))).sum().data.numpy()
            f1 = (tp)/(tp + 0.5*(fp+fn))
            f1_sc += f1


            if train == True:
                #loss
                we = torch.from_numpy(np.array([1.0, 2.0, 2.0, 2.0, 0.0])).float()
                l = F.cross_entropy(input=output, target=output_targets, weight=we)

                total_loss_list.append(l.data.mean())
                # train
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

        if train == True:
            current_average_loss = np.mean(total_loss_list)
        accuracy = 100 * (sum_correct_result / total_count)
        f1_score = f1_sc/len(data_lex)

        if train == True:
            return current_average_loss, accuracy, f1_score
        else:
            return accuracy, f1_score

    def forward_inference(self, current_word, previous_tag):
        input_word = autograd.Variable(torch.from_numpy(np.asarray([current_word])).long())
        input_targets = autograd.Variable(torch.from_numpy(np.asarray([previous_tag])).long())
        output = self.forward_pass(input_word, input_targets)
        softmax_probs = F.softmax(output)
        return softmax_probs

    # Viterbi algorithm
    def inference(self, data_lex, num_words, num_states, start_tag):
        num_states = num_states - 1

        # table to store the probability of the most likely path so far
        table_1 = np.zeros((num_words, num_states))
        # table to store the backpointers of the most likely path so far
        table_2 = np.zeros((num_words, num_states))

        # initialization

        # get the first word from the data_lex (list of sentences) and make a batch that has all the states, all the initial states, and all the first words in 3 separate vectors to pass in for batch inference from the MLP
        probabilities = self.forward_inference(data_lex[0], start_tag).data.numpy()

        max_tag = np.argmax(probabilities, 1)[0]

        # for all states, table_1(1, s) = p(s | s_o, x1..x_m)
        # fill in the table using the start state given random initial and the current word
        for i in range(num_states):
            table_1[0, i] = probabilities[0, i]
            table_2[0, i] = 0


        for j in range(1, num_words):
            # create a transition matrix that stores the probability of each state given previous state and current word
            transition_matrix = []
            for l in range(num_states):
                output = self.forward_inference(data_lex[j], l)
                transition_matrix.append(output.data.numpy()[0])
            transition_matrix = np.asarray(transition_matrix)

            # i = current states
            for i in range(num_states):
                max_prob = 0
                max_index = 0
                # k = old states
                for k in range(num_states):
                    current_prob = table_1[j-1, k] * transition_matrix[k, i]
                    if current_prob > max_prob:
                        max_prob = current_prob
                        max_index = k

                        if max_index == 127:
                            print(max_prob)
                            print(k)
                            print(i)
                    max_prob = max(table_1[j-1, k] * transition_matrix[k, i], max_prob)
                # max over probability of a state in previous row * probability of going to this current state given that previous state
                table_1[j, i] = max_prob
                table_2[j, i] = max_index

        # for the last word, what is the state with the best probability
        maxPreviousTag = np.argmax(table_1[num_words - 1, :])

        reverse_final_tag_sequence = []
        reverse_final_tag_sequence.append(maxPreviousTag)
        for i in reversed(range(1, num_words)):
            maxPreviousTag = int(table_2[i, int(maxPreviousTag)])
            reverse_final_tag_sequence.append(maxPreviousTag)

        final_tag_sequence = []
        for i in reversed(reverse_final_tag_sequence):
            final_tag_sequence.append(i)


        return final_tag_sequence

def preprocess_data():
    pass

def getTrainIdx(train,idx2word, word2idx):
    ret = []
    for sent in train:
        tmp=[]
        for word in sent:
            try:
                tmp.append(word2idx[word])
            except:
                idx2word.append(word)
                word2idx.update({word:(len(idx2word)-1)})
                tmp.append(len(idx2word)-1)
        ret.append(np.array(tmp))
    return ret

def getIdx2Word(train, idx2word):
    ret = []
    for sent in train:
        tmp = []
        for word in sent:
            tmp.append(idx2word[word])
        ret.append(tmp)
    return ret


def main():
    start_time = time.time()

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_file", type=str, default="Hw2/data/twitter1_train.txt", help="The training dataset")
    argparser.add_argument("--test_file", type=str, default="Hw2/data/twitter1_test.txt", help="The test dataset")
    argparser.add_argument("--option", type=int, default=1, help="The options")

    parsed_args = argparser.parse_args(sys.argv[1:])

    train_sents, train_tags = load_data(parsed_args.train_file)
    test_sents, test_tags = load_data(parsed_args.test_file)

    if(parsed_args.option == 2):
        wv_from_bin = KeyedVectors.load_word2vec_format('w2v.bin', binary=True)
        vects = torch.FloatTensor(wv_from_bin.vectors)
        idx2word = wv_from_bin.index2word
        idx2label = ['O','T-POS','T-NEG','T-NEU']
        word2idx = {}
        label2idx = {}
        for i in range(len(idx2word)):
            word2idx[idx2word[i]] = i
        for i in range(len(idx2label)):
            label2idx[idx2label[i]] = i

        train_lex = getTrainIdx(train_sents, idx2word, word2idx)
        train_y = getTrainIdx(train_tags, idx2label, label2idx)
        test_lex = getTrainIdx(test_sents, idx2word, word2idx)
        test_y = getTrainIdx(test_tags, idx2label, label2idx)

        print(len(idx2word), "WORD INDEX", len(idx2label), "LABEL INDEX")

        extra_row = torch.randn(len(idx2word)-3000000, 300)
        vects = torch.cat((vects, extra_row), 0)
        # valid_lex, _, valid_y = valid_set
        print(vects.shape)
        '''
            implement you training loop here
            '''
        model_save_path = "model2.pt"
        start_tag = 4
        # word_embedding_size = 100
        hidden_layer_size = 256
        learning_rate = 0.001
        num_epochs = 50
        mymemm = MEMM(vects, hidden_layer_size)
        optimizer = optim.Adam(mymemm.parameters(), lr=learning_rate, weight_decay=1e-5)

        # best parameters
        best_accuracy = 0.0
        best_loss = 1000000
        # train the model to do tag classification when taking a previous tag and current word and predict what the tag should be
        for epoch in range(num_epochs):
            print("EPOCH", epoch)
            train_loss, train_accuracy, train_f1 = mymemm.train(train_lex, train_y, optimizer, start_tag, train=True)
            test_accuracy, test_f1 = mymemm.train(test_lex, test_y, optimizer, start_tag, train=False)
            print(train_loss, train_accuracy, train_f1)
            print(test_accuracy, test_f1)

            if train_accuracy >= best_accuracy:
                torch.save(mymemm, model_save_path)

            best_accuracy = max(best_accuracy, train_accuracy)
            best_loss = min(best_loss, train_loss)

        # run the viterbi algorithm for inference -> no training required
        # self, data_lex, num_words, num_states, start_tag
        tot_prec = 0
        tot_rec = 0
        tot_f1 = 0
        pred_all = []
        for i in range(len(test_lex)):
            pred = mymemm.inference(test_lex[i], len(test_lex[i]), start_tag + 1, start_tag)
            pred_all.append(pred)
            pred = np.array(pred, dtype=int)
            act = test_y[i]
            tp = int(((act == pred) & (act != 0)).sum())
            fp = int(((act != pred) & (((act != 0) & (pred != 0)) | (act == 0))).sum())
            fn = int(((act != pred) & (((act != 0) & (pred != 0)) | (pred == 0))).sum())
            try:
                precision = (tp) / (tp + fp)
            except:
                precision = 0
            try:
                recall = (tp) / (tp + fn)
            except:
                recall = 0
            try:
                f1 = (2 * precision * recall) / (precision + recall)
            except:
                f1 = 0
            tot_prec += precision
            tot_rec += recall
            tot_f1 += f1
        tot_prec = tot_prec / len(test_lex)
        tot_rec = tot_rec / len(test_lex)
        tot_f1 = tot_f1 / len(test_lex)
        print("PREC:", tot_prec, "REC: ", tot_rec, "F1: ", tot_f1)

        predicted = getIdx2Word(pred_all, idx2label)
        reload_data('prediction.txt', test_sents, predicted)

        elapsed_time = time.time() - start_time

        print(elapsed_time)

    elif(parsed_args.option == 3):
        idx2word = []
        idx2label = ['O', 'T-POS', 'T-NEG', 'T-NEU']
        word2idx = {}
        label2idx = {}
        for i in range(len(idx2label)):
            label2idx[idx2label[i]] = i
        train_lex = getTrainIdx(train_sents, idx2word, word2idx)
        train_y = getTrainIdx(train_tags, idx2label, label2idx)
        test_lex = getTrainIdx(test_sents, idx2word, word2idx)
        test_y = getTrainIdx(test_tags, idx2label, label2idx)

        print(len(idx2word), "WORD INDEX", len(idx2label), "LABEL INDEX")

        '''
            implement you training loop here
        '''
        model_save_path = "model3.pt"
        start_tag = 4
        vocab_size = len(idx2word)
        word_embedding_size = 300
        hidden_layer_size = 93
        hidden_layer_size2 = 256
        learning_rate = 0.001
        num_layers = 2  # LSTM layers
        num_epochs = 5
        mymemm = MEMM2(vocab_size, word_embedding_size, hidden_layer_size, hidden_layer_size2,num_layers)
        optimizer = optim.Adam(mymemm.parameters(), lr=learning_rate, weight_decay=1e-5)

        # best parameters
        best_accuracy = 0.0
        best_loss = 1000000
        # train the model to do tag classification when taking a previous tag and current word and predict what the tag should be
        for epoch in range(num_epochs):
            print("EPOCH", epoch)
            train_loss, train_accuracy, train_f1 = mymemm.train(train_lex, train_y, optimizer, start_tag, train=True)
            test_accuracy, test_f1 = mymemm.train(test_lex, test_y, optimizer, start_tag, train=False)
            print(train_loss, train_accuracy, train_f1)
            print(test_accuracy, test_f1)

            if train_accuracy >= best_accuracy:
                torch.save(mymemm, model_save_path)

            best_accuracy = max(best_accuracy, train_accuracy)
            best_loss = min(best_loss, train_loss)

        # run the viterbi algorithm for inference -> no training required
        # self, data_lex, num_words, num_states, start_tag
        tot_prec = 0
        tot_rec = 0
        tot_f1 = 0
        pred_all = []
        for i in range(len(test_lex)):
            pred = mymemm.inference(test_lex[i], len(test_lex[i]), start_tag + 1, start_tag)
            pred_all.append(pred)
            pred = np.array(pred, dtype=int)
            act = test_y[i]
            tp = int(((act == pred) & (act != 0)).sum())
            fp = int(((act != pred) & (((act != 0) & (pred != 0)) | (act == 0))).sum())
            fn = int(((act != pred) & (((act != 0) & (pred != 0)) | (pred == 0))).sum())
            try:
                precision = (tp) / (tp + fp)
            except:
                precision = 0
            try:
                recall = (tp) / (tp + fn)
            except:
                recall = 0
            try:
                f1 = (2 * precision * recall) / (precision + recall)
            except:
                f1 = 0
            tot_prec += precision
            tot_rec += recall
            tot_f1 += f1
        tot_prec = tot_prec / len(test_lex)
        tot_rec = tot_rec / len(test_lex)
        tot_f1 = tot_f1 / len(test_lex)
        print("PREC:", tot_prec, "REC: ", tot_rec, "F1: ", tot_f1)

        predicted = getIdx2Word(pred_all,idx2label)
        reload_data('prediction.txt',test_sents,predicted)

        elapsed_time = time.time() - start_time

        print(elapsed_time)

    elif (parsed_args.option == 1):
        idx2word = []
        idx2label = ['O', 'T-POS', 'T-NEG', 'T-NEU']
        word2idx = {}
        label2idx = {}
        for i in range(len(idx2label)):
            label2idx[idx2label[i]] = i
        train_lex = getTrainIdx(train_sents, idx2word, word2idx)
        train_y = getTrainIdx(train_tags, idx2label, label2idx)
        test_lex = getTrainIdx(test_sents, idx2word, word2idx)
        test_y = getTrainIdx(test_tags, idx2label, label2idx)

        print(len(idx2word), "WORD INDEX", len(idx2label), "LABEL INDEX")

        '''
            implement you training loop here
        '''
        model_save_path = "model1.pt"
        start_tag = 4
        vocab_size = len(idx2word)
        word_embedding_size = 300
        hidden_layer_size = 256
        learning_rate = 0.001
        num_epochs = 5
        mymemm = MEMM3(vocab_size, word_embedding_size, hidden_layer_size)
        optimizer = optim.Adam(mymemm.parameters(), lr=learning_rate, weight_decay=1e-5)

        # best parameters
        best_accuracy = 0.0
        best_loss = 1000000
        # train the model to do tag classification when taking a previous tag and current word and predict what the tag should be
        for epoch in range(num_epochs):
            print("EPOCH", epoch)
            train_loss, train_accuracy, train_f1 = mymemm.train(train_lex, train_y, optimizer, start_tag, train=True)
            test_accuracy, test_f1 = mymemm.train(test_lex, test_y, optimizer, start_tag, train=False)
            print(train_loss, train_accuracy, train_f1)
            print(test_accuracy, test_f1)

            if train_accuracy >= best_accuracy:
                torch.save(mymemm, model_save_path)

            best_accuracy = max(best_accuracy, train_accuracy)
            best_loss = min(best_loss, train_loss)
            # best_test_accuracy = max(best_test_accuracy, test_accuracy)

        # run the viterbi algorithm for inference -> no training required
        # self, data_lex, num_words, num_states, start_tag
        tot_prec = 0
        tot_rec = 0
        tot_f1 = 0
        pred_all = []
        for i in range(len(test_lex)):
            pred = mymemm.inference(test_lex[i], len(test_lex[i]), start_tag + 1, start_tag)
            pred_all.append(pred)
            pred = np.array(pred, dtype=int)
            act = test_y[i]
            tp = int(((act==pred)&(act!=0)).sum())
            fp = int(((act!=pred) & (((act!=0)&(pred!=0)) | (act==0))).sum())
            fn = int(((act!=pred) & (((act!=0)&(pred!=0)) | (pred==0))).sum())
            try:
                precision = (tp) / (tp + fp)
            except:
                precision = 0
            try:
                recall = (tp) / (tp + fn)
            except:
                recall = 0
            try:
                f1 = (tp) / (tp + 0.5*(fp + fn))
            except:
                f1 = 0
            tot_prec += precision
            tot_rec += recall
            tot_f1 += f1
        tot_prec = tot_prec / len(test_lex)
        tot_rec = tot_rec / len(test_lex)
        tot_f1 = tot_f1 / len(test_lex)
        print("PREC:", tot_prec, "REC: ", tot_rec, "F1: ", tot_f1)

        predicted = getIdx2Word(pred_all, idx2label)
        reload_data('prediction.txt', test_sents, predicted)

        elapsed_time = time.time() - start_time

        print(elapsed_time)




if __name__ == '__main__':
    main()