# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#from sklearn.feature_extraction.text import TfidfVectorizer #TODO
import numpy as np
#from gensim.models import Word2Vec

def cos_distance_bag_tfidf(input_string_x1, input_string_x2,word_vec_dict, tfidf_dict,tfidf_flag=True):
    #print("input_string_x1:",input_string_x1)
    #1.1 get word vec for sentence 1
    sentence_vec1=get_sentence_vector(word_vec_dict,tfidf_dict, input_string_x1,tfidf_flag=tfidf_flag)
    #print("sentence_vec1:",sentence_vec1)
    #1.2 get word vec for sentence 2
    sentence_vec2 = get_sentence_vector(word_vec_dict, tfidf_dict, input_string_x2,tfidf_flag=tfidf_flag)
    #print("sentence_vec2:", sentence_vec2)
    #2 compute cos similiarity
    numerator=np.sum(np.multiply(sentence_vec1,sentence_vec2))
    denominator=np.sqrt(np.sum(np.power(sentence_vec1,2)))*np.sqrt(np.sum(np.power(sentence_vec2,2)))
    cos_distance=float(numerator)/float(denominator+0.000001)

    #print("cos_distance:",cos_distance)
    manhattan_distance=np.sum(np.abs(np.subtract(sentence_vec1,sentence_vec2)))
    #print(manhattan_distance,type(manhattan_distance),np.isnan(manhattan_distance))
    if np.isnan(manhattan_distance): manhattan_distance=300.0
    manhattan_distance=np.log(manhattan_distance+0.000001)/5.0

    canberra_distance=np.sum(np.abs(sentence_vec1-sentence_vec2)/np.abs(sentence_vec1+sentence_vec2))
    if np.isnan(canberra_distance): canberra_distance = 300.0
    canberra_distance=np.log(canberra_distance+0.000001)/5.0

    minkowski_distance=np.power(np.sum(np.power((sentence_vec1-sentence_vec2),3)), 0.33333333)
    if np.isnan(minkowski_distance): minkowski_distance = 300.0
    minkowski_distance=np.log(minkowski_distance+0.000001)/5.0

    euclidean_distance=np.sqrt(np.sum(np.power((sentence_vec1-sentence_vec2),2)))
    if np.isnan(euclidean_distance): euclidean_distance =300.0
    euclidean_distance=np.log(euclidean_distance+0.000001)/5.0

    return cos_distance,manhattan_distance,canberra_distance,minkowski_distance,euclidean_distance


def get_sentence_vector(word_vec_dict,tfidf_dict,word_list,tfidf_flag=True):
    vec_sentence=0.0
    length_vec=len(word_vec_dict[u'花呗'])
    for word in word_list:
        #print("word:",word)
        word_vec=word_vec_dict.get(word,None)
        word_tfidf=tfidf_dict.get(word,None)
        #print("word_vec:",word_vec,";word_tfidf:",word_tfidf)
        if word_vec is None is None or word_tfidf is None:
            continue
        else:
            if tfidf_flag==True:
                vec_sentence+=word_vec*word_tfidf
            else:
                vec_sentence += word_vec * 1.0
    vec_sentence=vec_sentence/(np.sqrt(np.sum(np.power(vec_sentence,2)))+0.000001)
    return vec_sentence

def get_tfidf_score_and_save(source_file,target_file):
    source_object=open(source_file, 'r')
    target_object = open(target_file, 'w')
    corpus=[line.strip() for line in source_object.readlines()]#corpus = ["This is very strange","This is very nice"]
    TfidfVectorizer=None #TODO TODO TODO remove this.
    print("You need to import TfidfVectorizer first, if you want to use tfidif function.")
    vectorizer = TfidfVectorizer(analyzer=lambda x:x.split(' '),min_df=3,use_idf=1,smooth_idf=1,sublinear_tf=1)
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    dict_word_tfidf=dict(zip(vectorizer.get_feature_names(), idf))
    for k,v in dict_word_tfidf.items():
        target_object.write(k+"|||"+str(v)+"\n")
    target_object.close()

source_file='./data/atec_nl_sim_train.txt'
target_file='./data/atec_nl_sim_tfidf.txt'
#get_tfidf_score(source_file,target_file)

#word2vec.word2vec('./data/atec_nl_sim_train.txt', './data/word2vec.vec', size=128,binary=0, verbose=True)