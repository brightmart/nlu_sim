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
splitter="&|&"
from data_mining.data_util_tfidf import cos_distance_bag_tfidf

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

    word_vec_fasttext_dict=load_word_vec('data/fasttext_fin_model_50.vec') #word embedding from fasttxt
    word_vec_word2vec_dict = load_word_vec('data/word2vec.txt') #word embedding from word2vec
    tfidf_dict=load_tfidf_dict('data/atec_nl_sim_tfidf.txt')
    BLUE_SCORE=[]
    for i,line in enumerate(fin):
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

        features_vector = data_mining_features(i, sen1, sen2, vocab_word2index, word_vec_fasttext_dict,word_vec_word2vec_dict, tfidf_dict, n_gram=8)
        BLUE_SCORE.append(features_vector)

    test=(lineno_list,X1,X2,BLUE_SCORE)
    return test

####many methods copy from data_util.py. the reason why not import from data_util.py is data_util.py
# use some package not support in test env. it is better to remove those package and import from data_util.py ########################################

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

def load_tfidf_dict(file_path):#今后|||11.357012399387852
    source_object = open(file_path, 'r')
    tfidf_dict={}
    for line in source_object:
        word,tfidf_value=line.strip().split(splitter)
        word=word.decode("utf-8")
        tfidf_dict[word]=float(tfidf_value)
    #print("tfidf_dict:",tfidf_dict)
    return tfidf_dict

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