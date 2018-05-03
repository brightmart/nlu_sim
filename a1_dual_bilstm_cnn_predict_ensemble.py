# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8') #gb2312
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)

import tensorflow as tf
import numpy as np
from a1_dual_bilstm_cnn_model import DualBilstmCnnModel
from data_util_test import load_vocabulary,load_test_data,get_label_by_logits,write_predict_result
import os
#import word2vec

#configuration
FLAGS=tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string("test_data_path","","test data")
#tf.app.flags.DEFINE_string("target_file","","target file")
#############################below for bilstm_char##########################################################################################
tf.app.flags.DEFINE_string("tokenize_style",'char',"tokenize sentence in char,word,or pinyin.default is char") #to tackle miss typed words
tf.app.flags.DEFINE_string("ckpt_dir","dual_bilstm_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_string("name_scope","bilstm_char","name scope value.")
############################below for bilstm_word###########################################################################################
tf.app.flags.DEFINE_string("tokenize_style_word",'word',"tokenize sentence in char,word,or pinyin.default is char") #to tackle miss typed words
tf.app.flags.DEFINE_string("ckpt_dir_word","dual_bilstm_word_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_string("name_scope_word","bilstm_word","name scope value.")
#######################################################################################################################
tf.app.flags.DEFINE_string("model","dual_bilstm","which model to use:dual_bilstm_cnn,dual_bilstm,dual_cnn.default is:dual_bilstm_cnn")

tf.app.flags.DEFINE_integer("embed_size",128,"embedding size") #128
tf.app.flags.DEFINE_integer("num_filters", 32, "number of filters") #32
tf.app.flags.DEFINE_integer("sentence_len",39,"max sentence length. length should be divide by 3, which is used by k max pooling.") #40
tf.app.flags.DEFINE_string("similiarity_strategy",'additive',"similiarity strategy: additive or multiply. default is additive") #to tackle miss typed words
tf.app.flags.DEFINE_string("max_pooling_style",'chunk_max_pooling',"max_pooling_style:max_pooling,k_max_pooling,chunk_max_pooling. default: chunk_max_pooling") #extract top k feature instead of max feature(max pooling)
tf.app.flags.DEFINE_integer("top_k", 3, "value of top k")
tf.app.flags.DEFINE_string("traning_data_path","./data/atec_nlp_sim_train.csv","path of traning data.")
tf.app.flags.DEFINE_integer("vocab_size",60000,"maximum vocab size.") #80000
tf.app.flags.DEFINE_float("learning_rate",0.0001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 4, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_pretrained_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","word2vec.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_float("dropout_keep_prob", 1.0, "dropout keep probability")


filter_sizes=[3,6,7,8]

#def main(_):
def predict_bilstm(inpath, outpath):
    vocabulary_word2index, vocabulary_index2label= load_vocabulary(FLAGS.traning_data_path,FLAGS.vocab_size,
                                                          name_scope=FLAGS.name_scope,tokenize_style=FLAGS.tokenize_style)
    vocab_size = len(vocabulary_word2index);print("cnn_model.vocab_size:",vocab_size);num_classes=len(vocabulary_index2label);print("num_classes:",num_classes)
    lineno_list, X1, X2=load_test_data(inpath, vocabulary_word2index, FLAGS.sentence_len, tokenize_style=FLAGS.tokenize_style)
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        textCNN=DualBilstmCnnModel(filter_sizes,FLAGS.num_filters,num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training,model=FLAGS.model,
                                   similiarity_strategy=FLAGS.similiarity_strategy,top_k=FLAGS.top_k,max_pooling_style=FLAGS.max_pooling_style)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Not able to find Checkpoint. Going to stop now...")
            iii=0
            iii/0
        #3.feed data & training
        number_of_test_data=len(X1)
        print("number_of_test_data:",number_of_test_data)
        batch_size=FLAGS.batch_size
        iteration=0
        file_object=open(outpath, 'a')
        divide_equally=(number_of_test_data%batch_size==0)
        steps=0
        if divide_equally:
            steps=int(number_of_test_data/batch_size)
        else:
            steps=int(number_of_test_data/batch_size)+1

        print("steps:",steps)
        start=0
        end=0
        for i in range(steps):
            print("i:",i)
            start=i*batch_size
            if i!=steps or divide_equally:
                end=(i+1)*batch_size
                feed_dict = {textCNN.input_x1: X1[start:end],textCNN.input_x2: X2[start:end],
                             textCNN.dropout_keep_prob: FLAGS.dropout_keep_prob,
                             textCNN.iter: iteration,textCNN.tst: not FLAGS.is_training}
                print(i*batch_size,end)
            else:
                end=number_of_test_data-(batch_size*int(number_of_test_data%batch_size))
                feed_dict = {textCNN.input_x1: X1[start:end],textCNN.input_x2: X2[start:end],
                             textCNN.dropout_keep_prob: FLAGS.dropout_keep_prob,
                             textCNN.iter: iteration,textCNN.tst: not FLAGS.is_training}
                print("start:",i*batch_size,";end:",end)
            logits=sess.run(textCNN.logits,feed_dict)
            label_list=get_label_by_logits(logits,vocabulary_index2label)
            write_predict_result(lineno_list[start:end],label_list,file_object)
        file_object.close()


if __name__ == "__main__":
    #tf.app.run()
    pass