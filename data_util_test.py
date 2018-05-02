# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import jieba
import numpy as np
PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"
TRUE_LABEL='1'
splitter="|||"
#from pypinyin import pinyin,lazy_pinyin
#import pickle

#use pretrained word embedding to get word vocabulary and labels, and its relationship with index
def load_vocabulary(training_data_path,vocab_size,name_scope='cnn',tokenize_style='char'):
    """
    create vocabulary
    :param training_data_path:
    :param vocab_size:
    :param name_scope:
    :return:
    """
    cache_vocab_label_pik = 'cache' + "_" + name_scope
    vocab_word2index_object=open(cache_vocab_label_pik+'/'+'vocab_word2index.txt',mode='r')
    vocab_word2index_lines=vocab_word2index_object.readlines()
    print("len of vocab_word2index_lines:",len(vocab_word2index_lines))
    vocab_word2index_dict={}
    for line in vocab_word2index_lines:
        word,index=line.strip().split(splitter)
        word=word.decode("utf-8")
        vocab_word2index_dict[word]=int(index)
    vocab_word2index_object.close()

    vocab_index2label_object = open(cache_vocab_label_pik + '/' + 'vocab_index2label.txt', mode='r')
    vocab_index2label_lines=vocab_index2label_object.readlines()
    vocab_index2label_dict={}
    for line in vocab_index2label_lines:
        index,label=line.strip().split(splitter)
        vocab_index2label_dict[int(index)]=str(label)

    return vocab_word2index_dict,vocab_index2label_dict

def load_test_data(test_data_path,vocab_word2index,max_sentence_len,tokenize_style='char'):
    """
    load test data, transfer to data suitable for model
    :param test_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :param max_sentence_len:
    :return:
    """
    #1.load test data
    fin=open(test_data_path, 'r')
    X1=[]
    X2=[]
    lineno_list=[]
    count=0
    for line in fin:
        lineno, sen1, sen2 = line.strip().split('\t')
        lineno_list.append(lineno)
        sen1=sen1.decode("utf-8")
        x1_list_ = token_string_as_list(sen1, tokenize_style=tokenize_style)
        sen2=sen2.decode("utf-8")
        x2_list_ = token_string_as_list(sen2, tokenize_style=tokenize_style)
        x1_list = [vocab_word2index.get(x, UNK_ID) for x in x1_list_]
        x2_list = [vocab_word2index.get(x, UNK_ID) for x in x2_list_]
        x1_list=pad_sequences(x1_list, max_sentence_len)
        x2_list=pad_sequences(x2_list,max_sentence_len)
        if count<10:#print some message
            print("x1_list:",x1_list)
            print("x2_list:",x2_list)
            count=count+1
        X1.append(x1_list)
        X2.append(x2_list)


    test=(lineno_list,X1,X2)
    return test

def pad_sequences(x_list_,max_sentence_len):
    length_x = len(x_list_)
    x_list=[]
    for i in range(0, max_sentence_len):
        if i < length_x:
            x_list.append(x_list_[i])
        else:
            x_list.append(PAD_ID)
    return x_list

def token_string_as_list(string,tokenize_style='char'): #this should be keep same as method in data_util.py, usually you can invoke from data_util.py without rewrite it again.
    length=len(string)
    if tokenize_style=='char':
        listt=[string[i] for i in range(length)]
    elif tokenize_style=='word':
        listt=jieba.lcut(string,cut_all=True)
    elif tokenize_style=='pinyin':
        lazy_pinyin=None #TODO if you want to use pinyin, you need to remove this line and import it first. temp remove as this package may not install in test env yet.
        string=" ".join(jieba.lcut(string))
        listt = ''.join(lazy_pinyin(string)).split() #list:['nihao', 'wo', 'de', 'pengyou']

    listt=[x for x in listt if x.strip()]
    return listt


def get_label_by_logits(logits, vocabulary_index2label):
    """
    get label by logits using index2label dict
    :param logits:
    :param vocabulary_index2label:
    :return:
    """
    # logits:[batch_size,num_classes]
    pred_labels = np.argmax(logits, axis=1)  # [batch_size]
    #result = [vocabulary_index2label[l] for l in pred_labels]
    result=[]
    for l in pred_labels:
        r=vocabulary_index2label[l]
        result.append(r)

    return result


def write_predict_result(line_no_list, label_list, file_object):
    for index, label in enumerate(label_list):
        file_object.write(line_no_list[index] + "\t" + label + "\n")

#x_list_=[1,2,3,4,5,5,6,7,8,9]
#max_sentence_len=5
#result1=pad_sequences(x_list_,max_sentence_len)
#print(result1)
#x_list_=[1,2,3,4,5,5,6,7,8,9]
#max_sentence_len=15
#result2=pad_sequences(x_list_,max_sentence_len)
#print(result2)