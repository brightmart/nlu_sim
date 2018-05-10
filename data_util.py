# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8') #gb2312
import random
import numpy as np
from tflearn.data_utils import pad_sequences
#from pypinyin import pinyin,lazy_pinyin

from collections import Counter
import os
import pickle
import csv
import jieba
from data_mining.data_util_tfidf import cos_distance_bag_tfidf,get_tfidf_score_and_save
jieba.add_word('花呗')
jieba.add_word('借呗')
PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"
TRUE_LABEL='1'
splitter="&|&"
special_start_token=[u'怎么',u'如何',u'为什么',u'为何']

def load_data(traning_data_path,vocab_word2index, vocab_label2index,sentence_len,name_scope,training_portion=0.95,tokenize_style='char'):
    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    cache_data_dir = 'cache' + "_" + name_scope  # path to save cache
    cache_file =cache_data_dir+"/"+'train_valid_test.pik'
    print("cache_path:",cache_file,"train_valid_test_file_exists:",os.path.exists(cache_file))
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as data_f:
            print("going to load cache file from file system and return")
            return pickle.load(data_f)

    csvfile = open(traning_data_path, 'r')
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    label_size=len(vocab_label2index)
    X1_ = []
    X2_ = []
    Y_ = []

    tfidf_source_file = './data/atec_nl_sim_train.txt'
    tfidf_target_file = './data/atec_nl_sim_tfidf.txt'
    if not os.path.exists(tfidf_target_file):
        get_tfidf_score_and_save(tfidf_source_file,tfidf_target_file)

    BLUE_SCORES_=[]
    word_vec_fasttext_dict=load_word_vec('data/fasttext_fin_model_50.vec') #word embedding from fasttxt
    word_vec_word2vec_dict = load_word_vec('data/word2vec.txt') #word embedding from word2vec
    #word2vec.word2vec('/Users/test/PycharmProjects/question_answering_similarity/data/atec_additional_cropus.txt',
    #                  '/Users/test/PycharmProjects/question_answering_similarity/data/word2vec_fin.bin', size=50, verbose=True,kind='txt')
    #print("word_vec_word2vec_dict:",word_vec_word2vec_dict)
    tfidf_dict=load_tfidf_dict('data/atec_nl_sim_tfidf.txt')

    for i, row in enumerate(spamreader):##row:['\ufeff1', '\ufeff怎么更改花呗手机号码', '我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号', '1']
        x1_list=token_string_as_list(row[1],tokenize_style=tokenize_style)
        x1 = [vocab_word2index.get(x, UNK_ID) for x in x1_list]
        x2_list=token_string_as_list(row[2],tokenize_style=tokenize_style)
        x2 = [vocab_word2index.get(x, UNK_ID) for x in x2_list]
        #add blue score features 2018-05-06
        features_vector=data_mining_features(i,row[1], row[2],vocab_word2index,word_vec_fasttext_dict,word_vec_word2vec_dict,tfidf_dict, n_gram=8)
        BLUE_SCORES_.append(features_vector)
        y_=row[3]
        y=vocab_label2index[y_]
        X1_.append(x1)
        X2_.append(x2)
        Y_.append(y)

        if i==0 or i==1 or i==2:
            print(i,"row[1]:",row[1],";x1:");print(row[1].decode("utf-8"))
            print(i,"row[2]:", row[2], ";x2:");print(row[2].decode("utf-8"))
            print(i,"row[3]:", row[3], ";y:", str(y))
            print(i,"row[4].feature vectors:",features_vector)

    number_examples = len(Y_)

    #shuffle
    X1=[]
    X2=[]
    Y=[]
    BLUE_SCORES=[]
    permutation = np.random.permutation(number_examples)
    for index in permutation:
        X1.append(X1_[index])
        X2.append(X2_[index])
        Y.append(Y_[index])
        BLUE_SCORES.append(BLUE_SCORES_[index])

    X1 = pad_sequences(X1, maxlen=sentence_len, value=0.)  # padding to max length
    X2 = pad_sequences(X2, maxlen=sentence_len, value=0.)  # padding to max length
    valid_number=min(1600,int((1-training_portion)*number_examples))
    test_number=800
    training_number=number_examples-valid_number-test_number
    valid_end=training_number+valid_number
    print(";training_number:",training_number,"valid_number:",valid_number,";test_number:",test_number)
    #generate more training data, while still keep data distribution for valid and test.
    X1_final, X2_final, BLUE_SCORE_final,Y_final,training_number_big=get_training_data(X1[0:training_number], X2[0:training_number], BLUE_SCORES[0:training_number],Y[0:training_number], training_number)
    train = (X1_final,X2_final, BLUE_SCORE_final,Y_final)
    valid = (X1[training_number+ 1:valid_end],X2[training_number+ 1:valid_end],BLUE_SCORES[training_number + 1:valid_end],Y[training_number + 1:valid_end])
    test=(X1[valid_end+1:],X2[valid_end:],BLUE_SCORES[valid_end:],Y[valid_end:])

    true_label_numbers=len([y for y in Y if y==1])
    true_label_pert=float(true_label_numbers)/float(number_examples)

    #save train/valid/test/true_label_pert to file system as cache
    # save to file system if vocabulary of words not exists(pickle).
    if not os.path.exists(cache_file):
        with open(cache_file, 'ab') as data_f:
            print("going to dump train/valid/test data to file sytem.")
            pickle.dump((train,valid,test,true_label_pert),data_f)
    return train,valid,test,true_label_pert

#use pretrained word embedding to get word vocabulary and labels, and its relationship with index
def create_vocabulary(training_data_path,vocab_size,name_scope='cnn',tokenize_style='char'):
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
            string_list_1=token_string_as_list(row[1],tokenize_style=tokenize_style)
            string_list_2 = token_string_as_list(row[2],tokenize_style=tokenize_style)
            c_inputs.update(string_list_1)
            c_inputs.update(string_list_2)

        #return most frequency words
        vocab_list=c_inputs.most_common(vocab_size)
        #put those words to dict
        for i,tuplee in enumerate(vocab_list):
            word,_=tuplee
            vocabulary_word2index[word]=i+2
            vocabulary_index2word[i+2]=word

        #save to file system if vocabulary of words not exists(pickle).
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label), data_f)
        #save to file system as file(added. for predict purpose when only few package is supported in test env)
        save_vocab_as_file(vocabulary_word2index,vocabulary_index2label,vocab_list,name_scope=name_scope)
    return vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label

def save_vocab_as_file(vocab_word2index,vocab_index2label,vocab_list,name_scope='cnn'):
    #1.1save vocabulary_word2index
    cache_vocab_label_pik = 'cache' + "_" + name_scope
    vocab_word2index_object=open(cache_vocab_label_pik+'/'+'vocab_word2index.txt',mode='a')
    for word,index in vocab_word2index.items():
        vocab_word2index_object.write(word+splitter+str(index)+"\n")
    vocab_word2index_object.close()

    #1.2 save word and frequent
    word_freq_object=open(cache_vocab_label_pik+'/'+'word_freq.txt',mode='a')
    for tuplee in vocab_list:
        word,count=tuplee
        word_freq_object.write(word+"|||"+str(count)+"\n")
    word_freq_object.close()

    #2.vocabulary_index2label
    vocab_index2label_object = open(cache_vocab_label_pik + '/' + 'vocab_index2label.txt',mode='a')
    for index,label in vocab_index2label.items():
        vocab_index2label_object.write(str(index)+splitter+str(label)+"\n")
    vocab_index2label_object.close()

def get_training_data(X1,X2,BLUE_SCORES,Y,training_number,shuffle_word_flag=False):
    # 1.form more training data by swap sentence1 and sentence2
    X1_big = []
    X2_big = []
    BLUE_SCORE_big=[]
    Y_big = []

    X1_final = []
    X2_final = []
    BLUE_SCORE_final=[]
    Y_final = []
    for index in range(0, training_number):
        X1_big.append(X1[index])
        X2_big.append(X2[index])
        BLUE_SCORE_big.append(BLUE_SCORES[index])
        y_temp = Y[index]
        Y_big.append(y_temp)
        #a.swap sentence1 and sentence2
        if str(y_temp) == TRUE_LABEL:
            X1_big.append(X2[index])
            X2_big.append(X1[index])
            BLUE_SCORE_big.append(BLUE_SCORES[index])
            Y_big.append(y_temp)

        #b.random change location of words
        if shuffle_word_flag:
            for x in range(5):
                x1=X1[index]
                x2=X2[index]
                x1_random=[x1[i] for i in range(len(x1))]
                x2_random = [x2[i] for i in range(len(x2))]
                random.shuffle(x1_random)
                random.shuffle(x2_random)
                X1_big.append(x1_random)
                X2_big.append(x2_random)
                BLUE_SCORE_big.append(BLUE_SCORES[index])
                Y_big.append(Y[index])

    # shuffle data
    training_number_big = len(X1_big)
    permutation2 = np.random.permutation(training_number_big)
    for index in permutation2:
        X1_final.append(X1_big[index])
        X2_final.append(X2_big[index])
        BLUE_SCORE_final.append(BLUE_SCORE_big[index])
        Y_final.append(Y_big[index])

    return X1_final,X2_final,BLUE_SCORE_final,Y_final,training_number_big

def token_string_as_list(string,tokenize_style='char'):
    string=string.decode("utf-8")
    string=string.replace("***","*")
    length=len(string)
    if tokenize_style=='char':
        listt=[string[i] for i in range(length)]
    elif tokenize_style=='word':
        listt=jieba.lcut(string) #cut_all=True
    elif tokenize_style=='pinyin':
        string=" ".join(jieba.lcut(string))
        listt = ''.join(lazy_pinyin(string)).split() #list:['nihao', 'wo', 'de', 'pengyou']

    listt=[x for x in listt if x.strip()]
    return listt




def data_mining_features(index,input_string_x1,input_string_x2,vocab_word2index,word_vec_fasttext_dict,word_vec_word2vec_dict,tfidf_dict,n_gram=8):
    """
    get data mining feature given two sentences as string.
    1)n-gram similiarity(blue score);
    2) get length of questions, difference of length
    3) how many words are same, how many words are unique
    4) question 1,2 start with how/why/when(为什么，怎么，如何，为何）
    5）edit distance
    6) cos similiarity using bag of words
    :param input_string_x1:
    :param input_string_x2:
    :return:
    """
    input_string_x1=input_string_x1.decode("utf-8")
    input_string_x2 = input_string_x2.decode("utf-8")
    #1. get blue score vector
    feature_list=[]
    #get blue score with n-gram
    for i in range(n_gram):
        x1_list=split_string_as_list_by_ngram(input_string_x1,i+1)
        x2_list = split_string_as_list_by_ngram(input_string_x2, i + 1)
        blue_score_i_1 = compute_blue_ngram(x1_list,x2_list)
        blue_score_i_2 = compute_blue_ngram(x2_list,x1_list)
        feature_list.append(blue_score_i_1)
        feature_list.append(blue_score_i_2)

    #2. get length of questions, difference of length
    length1=float(len(input_string_x1))
    length2=float(len(input_string_x2))
    length_diff=(float(abs(length1-length2)))/((length1+length2)/2.0)
    feature_list.append(length_diff)

    #3. how many words are same, how many words are unique
    sentence_diff_overlap_features_list=get_sentence_diff_overlap_pert(index,input_string_x1,input_string_x2)
    feature_list.extend(sentence_diff_overlap_features_list)

    #4. question 1,2 start with how/why/when(为什么，怎么，如何，为何）
    #how_why_feature_list=get_special_start_token(input_string_x1,input_string_x2,special_start_token)
    #print("how_why_feature_list:",how_why_feature_list)
    #feature_list.extend(how_why_feature_list)

    #5.edit distance
    edit_distance=float(edit(input_string_x1, input_string_x2))/30.0
    feature_list.append(edit_distance)

    #6.cos distance from sentence embedding
    x1_list=token_string_as_list(input_string_x1, tokenize_style='word')
    x2_list = token_string_as_list(input_string_x2, tokenize_style='word')
    distance_list_fasttext = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_fasttext_dict, tfidf_dict)
    distance_list_word2vec = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_word2vec_dict, tfidf_dict)
    #distance_list2 = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_fasttext_dict, tfidf_dict,tfidf_flag=False)
    #sentence_diffence=np.abs(np.subtract(sentence_vec_1,sentence_vec_2))
    #sentence_multiply=np.multiply(sentence_vec_1,sentence_vec_2)
    feature_list.extend(distance_list_fasttext)
    feature_list.extend(distance_list_word2vec)
    #feature_list.extend(list(sentence_diffence))
    #feature_list.extend(list(sentence_multiply))
    return feature_list

def load_word_vec(file_path):
    source_object = open(file_path, 'r')
    word_vec_dict={}
    for i,line in enumerate(source_object):
        if i==0 and 'word2vec' in file_path:
            continue
        line=line.strip()
        line_list=line.split()
        word=line_list[0].decode("utf-8")
        vec_list=[float(x) for x in line_list[1:]]
        word_vec_dict[word]=np.array(vec_list)
    #print("word_vec_dict:",word_vec_dict)
    return word_vec_dict


def load_tfidf_dict(file_path):#今后|||11.357012399387852
    source_object = open(file_path, 'r')
    tfidf_dict={}
    for line in source_object:
        word,tfidf_value=line.strip().split(splitter)
        word=word.decode("utf-8")
        tfidf_dict[word]=float(tfidf_value)
    #print("tfidf_dict:",tfidf_dict)
    return tfidf_dict

def get_special_start_token(input_string_x1,input_string_x2,special_token_list):
    feature_list1=[0.0 for i in range(len(special_token_list))]
    feature_list2=[0.0 for i in range(len(special_token_list))]

    for i,speical_token in enumerate(special_token_list):
        if input_string_x1.find(speical_token)>0: #speical_token in input_string_x1:
            feature_list1[i]=1.0
        if input_string_x2.find(speical_token)>0:
            feature_list2[i]=1.0

    feature_list1.extend(feature_list2)
    return feature_list1


def get_sentence_diff_overlap_pert(index,input_string_x1,input_string_x2):
    #0. get list from string
    input_list1=[input_string_x1[token] for token in range(len(input_string_x1)) if input_string_x1[token].strip()]
    input_list2 = [input_string_x2[token] for token in range(len(input_string_x2)) if input_string_x2[token].strip()]
    length1=len(input_list1)
    length2=len(input_list2)

    num_same=0
    same_word_list=[]
    #1.compute percentage of same tokens
    for word1 in input_list1:
        for word2 in input_list2:
           if word1==word2:
               num_same=num_same+1
               same_word_list.append(word1)
               continue
    num_same_pert_min=float(num_same)/float(max(length1,length2))
    num_same_pert_max = float(num_same) / float(min(length1, length2))
    num_same_pert_avg = float(num_same) / (float(length1+length2)/2.0)

    #2.compute percentage of unique tokens in each string
    input_list1_unique=set([x for x in input_list1 if x not in same_word_list])
    input_list2_unique = set([x for x in input_list2 if x not in same_word_list])
    num_diff_x1=float(len(input_list1_unique))/float(length1)
    num_diff_x2= float(len(input_list2_unique)) / float(length2)

    if index==0:#print debug message
        print("input_string_x1:",input_string_x1)
        print("input_string_x2:",input_string_x2)
        print("same_word_list:",same_word_list)
        print("input_list1_unique:",input_list1_unique)
        print("input_list2_unique:",input_list2_unique)
        print("num_same:",num_same,";length1:",length1,";length2:",length2,";num_same_pert_min:",num_same_pert_min,
              ";num_same_pert_max:",num_same_pert_max,";num_same_pert_avg:",num_same_pert_avg,
             ";num_diff_x1:",num_diff_x1,";num_diff_x2:",num_diff_x2)

    diff_overlap_list=[num_same_pert_min,num_same_pert_max, num_same_pert_avg,num_diff_x1, num_diff_x2]
    return diff_overlap_list


def split_string_as_list_by_ngram(input_string,ngram_value):
    #print("input_string0:",input_string)
    input_string="".join([string for string in input_string if string.strip()])
    #print("input_string1:",input_string)
    length = len(input_string)
    result_string=[]
    for i in range(length):
        if i + ngram_value < length + 1:
            result_string.append(input_string[i:i+ngram_value])
    #print("ngram:",ngram_value,"result_string:",result_string)
    return result_string


def compute_blue_ngram(x1_list,x2_list):
    """
    compute blue score use ngram information. x1_list as predict sentence,x2_list as target sentence
    :param x1_list:
    :param x2_list:
    :return:
    """
    count_dict={}
    count_dict_clip={}
    #1. count for each token at predict sentence side.
    for token in x1_list:
        if token not in count_dict:
            count_dict[token]=1
        else:
            count_dict[token]=count_dict[token]+1
    count=np.sum([value for key,value in count_dict.items()])

    #2.count for tokens existing in predict sentence for target sentence side.
    for token in x2_list:
        if token in count_dict:
            if token not in count_dict_clip:
                count_dict_clip[token]=1
            else:
                count_dict_clip[token]=count_dict_clip[token]+1

    #3. clip value to ceiling value for that token
    count_dict_clip={key:(value if value<=count_dict[key] else count_dict[key]) for key,value in count_dict_clip.items()}
    count_clip=np.sum([value for key,value in count_dict_clip.items()])
    result=float(count_clip)/(float(count)+0.00000001)
    return result


def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    #print("matrix:",matrix)
    for i in xrange(1, len(str1) + 1):
        for j in xrange(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


#print ("result:",edit('你好啊我的朋友', '你好朋友在吗啊'))

#test1:load data
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

#test2: token string as list
#string='你好我的朋友'
#result=token_string_as_list(string)
#print("result:",result)

#test3:comput n-gram similiarity featue
#input_x1=u"开通花呗收款后，符合什么条件的商家或买家才可以使用花呗交易"
#input_x1=u"我是商家，我已经申请.开通花呗收款后，符合什么条件的商家或买家才可以使用花呗交易"
#input_x2=u"我是商家，我已经申请了开通蚂蚁花呗和信用卡收款，为什么还是不可以"
#result=blue_score_feature(input_x1,input_x2,n_gram=8)
#print("result:",result)

#test4:compute sentence diff and overlap
input_x1=u"开通花呗后收款，符合什么条件的商家或买家才可以使用花呗交易"
input_x2=u"我是商家，我已经申请.开通花呗收款后，符合什么条件的商家或买家才可以使用花呗交易"
#result=get_sentence_diff_overlap_pert(0,input_x1,input_x2)
#print("result:",result)

#test5: indicator for special start word
#input_x1=u"如何花呗后收款，你好啊符合什么条件的商家或买家才可以使用花呗交易"
#input_x2=u"怎么商家，我已经申请.开通花呗收款后，符合什么条件的商家或买家才可以使用花呗交易是的吗"
#special_start_token=['怎么','如何','为什么','为何']
#result=get_special_start_token(input_x1,input_x2,special_start_token)
#print("result:",result)