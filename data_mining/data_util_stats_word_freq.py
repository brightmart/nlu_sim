# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import csv
from data_util import *
import json

def get_sentence_embedding(sentence):
    #1.get tfidf value for each word
    #2.get word embedding for each word
    #3.compute sentence embedding using weighted sum from word embeddings.
    pass

def rewrite_data_as_text(file_path,target_file):
    #1.read data
    csvfile = open(file_path, 'r')
    target_object=open(target_file,'a')
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    total_length=0
    count=0
    length_dict={5:0,10:0,15:0,20:0,25:0,30:0,35:0}
    for i, row in enumerate(spamreader):##row:['\ufeff1', '\ufeff怎么更改花呗手机号码', '我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号', '1']
        # 2.tokenize use jieba
        x1_list=token_string_as_list(row[1],tokenize_style='word')
        x2_list=token_string_as_list(row[2],tokenize_style='word')
        #3.write to file system
        total_length=total_length+len(x1_list)+len(x2_list)
        target_object.write(" ".join(x1_list)+"\n")
        target_object.write(" ".join(x2_list)+"\n")
        count=count+1

        x1_list=x2_list
        if len(x1_list)<5:
            length_dict[5]=length_dict[5]+1
        elif len(x1_list)<10:
            length_dict[10] = length_dict[10] + 1
        elif len(x1_list)<15:
            length_dict[15] = length_dict[15] + 1
        elif len(x1_list)<20:
            length_dict[20] = length_dict[20] + 1
        elif len(x1_list)<25:
            length_dict[25] = length_dict[25] + 1
        elif len(x1_list)<30:
            length_dict[30] = length_dict[30] + 1
        else:
            length_dict[35] = length_dict[35] + 1
    print("length_dict1:",length_dict)
    length_dict={k:float(v)/float(count) for k,v in length_dict.items()}
    print("length_dict2:", length_dict)
    target_object.close()

    avg_length=(float(total_length)/2.0)/float(count)
    print("avg length:",avg_length) #8
    #('length_dict2:', {35: 0.003888578254460428, 5: 0.11332791135058201, 10: 0.6525186804249479, 15: 0.17132618309358003, 20: 0.040741117267320694, 25: 0.013190667412189295, 30: 0.005006862196919636})
    #('length_dict2:', { 5: 0.11332791135058201, 10: 0.6525186804249479, 15: 0.17132618309358003, 20: 0.040741117267320694})
    #('length_dict2:', {35: 0.003481929548111625, 5: 0.11388705332181162, 10: 0.6559243633406191, 15: 0.1654043613073756, 20: 0.04325725613785391, 25: 0.013139836323895695, 30: 0.004905200020332436})
    #('length_dict2:', { 5: 0.11388705332181162, 10: 0.6559243633406191, 15: 0.1654043613073756, 20: 0.04325725613785391})

def token_string_as_list(string,tokenize_style='word'):
    string=string.decode("utf-8")
    string=string.replace("***","*")
    length=len(string)
    if tokenize_style=='char':
        listt=[string[i] for i in range(length)]
    elif tokenize_style=='word':
        listt=jieba.lcut(string) #cut_all=True

    listt=[x for x in listt if x.strip()]
    return listt

def token_write_as_text(file_path,target_file,source_file2):
    file_object=open(file_path,'r')
    target_object = open(target_file, 'a')
    i=0
    if i>100:
        for line in file_object:
            line=line.strip()
            json_string=json.loads(line)
            #message
            message=json_string['message']
            message_list=token_string_as_list(message)
            if len(message_list)>0:
                target_object.write(" ".join(message_list)+"\n")
            #response
            response=json_string['response']
            response_list = token_string_as_list(response)
            if len(response_list)>0:
                target_object.write(" ".join(response_list)+"\n")
            #context
            context_list=json_string['context']
            for context in context_list:
                if len(context)>0:
                    context_l=token_string_as_list(context)
                    if len(context_l)>0:
                        target_object.write(" ".join(context_l) + "\n")

    print("start part2")
    file_object.close()

    source_object2=open(source_file2,'r')
    for line in source_object2:
        line=line.strip()
        string_list=token_string_as_list(line)
        target_object.write(" ".join(string_list) + "\n")
    target_object.close()
    source_object2.close()

#file_path='data/atec_nlp_sim_train.csv'
#target_file='data/atec_nl_sim_train.txt'
#rewrite_data_as_text(file_path,target_file)
file_path='data/dialogue-haitong-processed-as-dialogues.json.clean'
target_file='data/atec_additional_cropus.txt'
source_file2='data/train.qa'
#token_write_as_text(file_path,target_file,source_file2)

