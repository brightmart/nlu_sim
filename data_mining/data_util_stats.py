import csv

def count_sentence_frequent(traning_data_path):
    csvfile = open(traning_data_path, 'r')
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    dict_sentence_count={}
    for i, row in enumerate(spamreader):
        x1 = row[1].decode("utf-8")
        x2 = row[2].decode("utf-8")
        if x1 in dict_sentence_count:
            dict_sentence_count[x1]= dict_sentence_count[x1]+1
        else:
            dict_sentence_count[x1] =1
        if x2 in dict_sentence_count:
            dict_sentence_count[x2]= dict_sentence_count[x2]+1
        else:
            dict_sentence_count[x2] =1

    count=0
    for k,v in dict_sentence_count.items():
        if v>1:
            print(k);print(v)
            count=count+1
    print("count:",count)

traning_data_path='/Users/test/PycharmProjects/question_answering_similarity/data/atec_nlp_sim_train.csv'
count_sentence_frequent(traning_data_path)
