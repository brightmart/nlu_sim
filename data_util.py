# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') #gb2312
import codecs
import random
import numpy as np
from tflearn.data_utils import pad_sequences
from collections import Counter
import os
import pickle
import csv
import jieba
PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"


def load_data(traning_data_path,vocab_word2index, vocab_label2index,sentence_len,training_portion=0.95):
    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    csvfile = open(traning_data_path, 'r')
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    label_size=len(vocab_label2index)
    X1_ = []
    X2_ = []
    Y_ = []

    print("vocab_label2index:",vocab_label2index)
    for i, row in enumerate(spamreader):##row:['\ufeff1', '\ufeff怎么更改花呗手机号码', '我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号', '1']
        x1_list=token_string_as_list(row[1])
        x1 = [vocab_word2index.get(x, UNK_ID) for x in x1_list]

        x2_list=token_string_as_list(row[2])
        x2 = [vocab_word2index.get(x, UNK_ID) for x in x2_list]

        y_=row[3]
        y=vocab_label2index[y_]

        X1_.append(x1)
        X2_.append(x2)
        Y_.append(y)

        if i==0:
            print("row[1]:",row[1],";x1:","".join(str(x1)))
            print("row[2]:", row[2], ";x2:", "".join(str(x2)))
            print("row[3]:", row[3], ";y:", str(y))

    number_examples = len(Y_)

    #shuffle
    X1=[]
    X2=[]
    Y=[]
    permutation = np.random.permutation(number_examples)
    for index in permutation:
        X1.append(X1_[index])
        X2.append(X2_[index])
        Y.append(Y_[index])

    X1 = pad_sequences(X1, maxlen=sentence_len, value=0.)  # padding to max length
    X2 = pad_sequences(X2, maxlen=sentence_len, value=0.)  # padding to max length
    valid_number=min(1600,int((1-training_portion)*number_examples))
    test_number=800
    training_number=number_examples-valid_number-test_number
    valid_end=training_number+valid_number
    print(";training_number:",training_number,"valid_number:",valid_number,";test_number:",test_number)
    train = (X1[0:training_number],X2[0:training_number], Y[0:training_number])
    valid = (X1[training_number+ 1:valid_end],X2[training_number+ 1:valid_end],Y[training_number + 1:valid_end])
    test=(X1[valid_end+1:],X2[valid_end:],Y[valid_end:])

    true_label_numbers=len([y for y in Y if y==1])
    true_label_pert=float(true_label_numbers)/float(number_examples)
    return train,valid,test,true_label_pert

#use pretrained word embedding to get word vocabulary and labels, and its relationship with index
def create_vocabulary(training_data_path,vocab_size,name_scope='cnn'):
    """
    create vocabulary
    :param training_data_path:
    :param vocab_size:
    :param name_scope:
    :return:
    """

    cache_vocabulary_label_pik='cache'+"_"+name_scope # path to save cache
    if not os.path.isdir(cache_vocabulary_label_pik): # create folder if not exists.
        os.makedirs(cache_vocabulary_label_pik)

    # if cache exists. load it; otherwise create it.
    cache_path =cache_vocabulary_label_pik+"/"+'vocab_label.pik'
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            return pickle.load(data_f)
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        vocabulary_word2index[_PAD]=PAD_ID
        vocabulary_index2word[PAD_ID]=_PAD
        vocabulary_word2index[_UNK]=UNK_ID
        vocabulary_index2word[UNK_ID]=_UNK

        vocabulary_label2index={'0':0,'1':1}
        vocabulary_index2label={0:'0',1:'1'}

        #1.load raw data
        csvfile = open(training_data_path, 'r')
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')

        #2.loop each line,put to counter
        c_inputs=Counter()
        c_labels=Counter()
        for i,row in enumerate(spamreader):#row:['\ufeff1', '\ufeff怎么更改花呗手机号码', '我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号', '1']
            string_list_1=token_string_as_list(row[1])
            string_list_2 = token_string_as_list(row[2])
            c_inputs.update(string_list_1)
            c_inputs.update(string_list_1)

        #return most frequency words
        vocab_list=c_inputs.most_common(vocab_size)
        #put those words to dict
        for i,tuplee in enumerate(vocab_list):
            word,_=tuplee
            vocabulary_word2index[word]=i+2
            vocabulary_index2word[i+2]=word

        #save to file system if vocabulary of words not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label), data_f)
    return vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label

def token_string_as_list(string):
    listt=jieba.lcut(string,cut_all=True)
    listt=[x for x in listt if x!='']
    return listt

#training_data_path='./data/atec_nlp_sim_train.csv'
##vocab_size=50000
#create_vocabulary(training_data_path,vocab_size)
#sentence_len=30
#cache_path='cache_cnn/vocab_label.pik'
#vocab_word2index={}
#vocab_label2index={}
#with open(cache_path, 'rb') as data_f:
#    vocab_word2index, _, vocab_label2index, _=pickle.load(data_f)
#load_data(training_data_path,vocab_word2index, vocab_label2index,sentence_len,training_portion=0.95)
