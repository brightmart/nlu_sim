# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8') #gb2312
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from a1_dual_bilstm_cnn_model import DualBilstmCnnModel
from data_util import create_vocabulary,load_data
import os
import random
import word2vec
from weight_boosting import compute_labels_weights,get_weights_for_current_batch,get_weights_label_as_standard_dict,init_weights_dict
#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("ckpt_dir","","checkpoint location for the model") #dual_bilstm_char_checkpoint/
tf.app.flags.DEFINE_string("tokenize_style",'',"tokenize sentence in char,word,or pinyin.default is char") #char
tf.app.flags.DEFINE_string("model_name","","which model to use:dual_bilstm_cnn,dual_bilstm,dual_cnn,mix. default is:mix")#dual_bilstm
tf.app.flags.DEFINE_string("name_scope","","name scope value.") #bilstm_char

tf.app.flags.DEFINE_boolean("decay_lr_flag",True,"whether manally decay lr")
tf.app.flags.DEFINE_integer("embed_size",50,"embedding size") #128
tf.app.flags.DEFINE_integer("num_filters",10, "number of filters") #64
tf.app.flags.DEFINE_integer("sentence_len",21,"max sentence length. length should be divide by 3, which is used by k max pooling.") #39
tf.app.flags.DEFINE_string("similiarity_strategy",'additive',"similiarity strategy: additive or multiply. default is additive") #to tackle miss typed words
tf.app.flags.DEFINE_string("max_pooling_style",'chunk_max_pooling',"max_pooling_style:max_pooling,k_max_pooling,chunk_max_pooling. default: chunk_max_pooling") #extract top k feature instead of max feature(max pooling)

tf.app.flags.DEFINE_integer("top_k", 3, "value of top k")
tf.app.flags.DEFINE_string("traning_data_path","./data/atec_nlp_sim_train.csv","path of traning data.")
tf.app.flags.DEFINE_integer("vocab_size",60000,"maximum vocab size.") #80000
tf.app.flags.DEFINE_float("learning_rate",0.0005,"learning rate") #0.001
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",15,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_pretrained_embedding",True,"whether to use embedding or not.")

tf.app.flags.DEFINE_string("word2vec_model_path","data/fasttext_fin_model_50.vec","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.6, "dropout keep probability")


filter_sizes=[2,3,4]

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #if FLAGS.use_pingyin:
    vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label= create_vocabulary(FLAGS.traning_data_path,FLAGS.vocab_size,
                                                                                              name_scope=FLAGS.name_scope,tokenize_style=FLAGS.tokenize_style)
    vocab_size = len(vocabulary_word2index);print("cnn_model.vocab_size:",vocab_size);num_classes=len(vocabulary_index2label);print("num_classes:",num_classes)
    train, valid, test,true_label_percent= load_data(FLAGS.traning_data_path,vocabulary_word2index, vocabulary_label2index,FLAGS.sentence_len,FLAGS.name_scope,tokenize_style=FLAGS.tokenize_style)
    trainX1,trainX2, trainBlueScores,trainY = train
    validX1,validX2,validBlueScores,validY=valid
    testX1,testX2, testBlueScores,testY = test
    length_data_mining_features=len(trainBlueScores[0])
    print("length_data_mining_features:",length_data_mining_features)
    #print some message for debug purpose
    print("model_name:",FLAGS.model_name,";length of training data:",len(trainX1),";length of validation data:",len(testX1),";true_label_percent:",
          true_label_percent,";tokenize_style:",FLAGS.tokenize_style,";vocabulary size:",vocab_size)
    print("train_x1:",trainX1[0],";train_x2:",trainX2[0])
    print("data mining features.length:",len(trainBlueScores[0]),"data_mining_features:",trainBlueScores[0],";train_y:",trainY[0])
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        textCNN=DualBilstmCnnModel(filter_sizes,FLAGS.num_filters,num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training,model=FLAGS.model_name,
                        similiarity_strategy=FLAGS.similiarity_strategy,top_k=FLAGS.top_k,max_pooling_style=FLAGS.max_pooling_style,
                        length_data_mining_features=length_data_mining_features)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            if FLAGS.decay_lr_flag:
                #trainX1, trainX2, trainY = shuffle_data(trainX1, trainX2, trainY)
                for i in range(2):  # decay learning rate if necessary.
                    print(i, "Going to decay learning rate by half.")
                    sess.run(textCNN.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if not os.path.exists(FLAGS.ckpt_dir):
                os.makedirs(FLAGS.ckpt_dir)

            if FLAGS.use_pretrained_embedding: #load pre-trained word embedding
                print("going to use pretrained word embeddings...")
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN,FLAGS.word2vec_model_path)
        curr_epoch=sess.run(textCNN.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX1)
        batch_size=FLAGS.batch_size
        iteration=0
        best_acc=0.60
        best_f1_score=0.20
        weights_dict = init_weights_dict(vocabulary_label2index) #init weights dict.
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            print("Auto.Going to shuffle data")
            trainX1, trainX2, trainBlueScores,trainY = shuffle_data(trainX1, trainX2,trainBlueScores, trainY)
            loss, eval_acc,counter =  0.0,0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                iteration=iteration+1
                input_x1,input_x2,input_bluescores,input_y=generate_batch_training_data(trainX1, trainX2,trainBlueScores, trainY, number_of_training_data, batch_size)
                #input_x1=trainX1[start:end]
                #input_x2=trainX2[start:end]
                #input_bluescores=trainBlueScores[start:end]
                #input_y=trainY[start:end]
                weights = get_weights_for_current_batch(input_y, weights_dict)

                feed_dict = {textCNN.input_x1: input_x1,textCNN.input_x2: input_x2,textCNN.input_bluescores:input_bluescores,textCNN.input_y:input_y,
                             textCNN.weights: np.array(weights),textCNN.dropout_keep_prob: FLAGS.dropout_keep_prob,
                             textCNN.iter: iteration,textCNN.tst: not FLAGS.is_training}
                curr_loss,curr_acc,lr,_,_=sess.run([textCNN.loss_val,textCNN.accuracy,textCNN.learning_rate,textCNN.update_ema,textCNN.train_op],feed_dict)
                loss,eval_acc,counter=loss+curr_loss,eval_acc+curr_acc,counter+1
                if counter %100==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tAcc:%.3f\tLearning rate:%.5f" %(epoch,counter,loss/float(counter),eval_acc/float(counter),lr))
                #middle checkpoint
                #if start!=0 and start%(500*FLAGS.batch_size)==0: # eval every 3000 steps.
                    #eval_loss, acc,f1_score, precision, recall,_ = do_eval(sess, textCNN, validX1, validX2, validY,iteration)
                    #print("【Validation】Epoch %d Loss:%.3f\tAcc:%.3f\tF1 Score:%.3f\tPrecision:%.3f\tRecall:%.3f" % (epoch, acc,eval_loss, f1_score, precision, recall))
                    # save model to checkpoint
                    #save_path = FLAGS.ckpt_dir + "model.ckpt"
                    #saver.save(sess, save_path, global_step=epoch)
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))

            if epoch % FLAGS.validate_every==0:
                eval_loss,eval_accc,f1_scoree,precision,recall,weights_label=do_eval(sess,textCNN,validX1,validX2,validBlueScores,validY,iteration,vocabulary_index2word)
                weights_dict = get_weights_label_as_standard_dict(weights_label)
                print("label accuracy(used for label weight):==========>>>>", weights_dict)
                print("【Validation】Epoch %d\t Loss:%.3f\tAcc %.3f\tF1 Score:%.3f\tPrecision:%.3f\tRecall:%.3f" % (epoch,eval_loss,eval_accc,f1_scoree,precision,recall))
                #save model to checkpoint
                if eval_accc*1.05>best_acc and f1_scoree>best_f1_score:
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    print("going to save model. eval_f1_score:",f1_scoree,";previous best f1 score:",best_f1_score, ";eval_acc",str(eval_accc),";previous best_acc:",str(best_acc))
                    saver.save(sess,save_path,global_step=epoch)
                    best_acc=eval_accc
                    best_f1_score=f1_scoree

                if FLAGS.decay_lr_flag and (epoch!=0 and (epoch==2 or epoch==5 or epoch==9 or epoch==13)):
                    #TODO print("Auto.Restoring Variables from Checkpoint.")
                    #TODO saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

                    for i in range(2):  # decay learning rate if necessary.
                        print(i, "Going to decay learning rate by half.")
                        sess.run(textCNN.learning_rate_decay_half_op)



        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss,acc_t,f1_score_t,precision,recall,weights_label = do_eval(sess, textCNN, testX1,testX2,testBlueScores, testY,iteration,vocabulary_index2word)
        print("Test Loss:%.3f\tAcc:%.3f\tF1 Score:%.3f\tPrecision:%.3f\tRecall:%.3f:" % ( test_loss,acc_t,f1_score_t,precision,recall))
    pass


def shuffle_data(trainX1,trainX2,trainFeatures,trainY):
    c = list(zip(trainX1,trainX2,trainFeatures,trainY))
    random.shuffle(c)
    trainX1[:], trainX2[:], trainFeatures[:],trainY[:]= zip(*c)
    return trainX1, trainX2,trainFeatures, trainY

def generate_batch_training_data(X1,X2,trainBlueScores,Y,num_data,batch_size):
    """
    :param X1:
    :param X2:
    :param y:
    :return:
    """
    index_list_=random.sample(range(0, num_data), batch_size*5)
    #print("length of index_list_",len(index_list_))
    #random select a list of index
    index_list=[]
    countt_true=0
    count_false=0
    for i,index in enumerate(index_list_):
        if Y[index]==1 and countt_true<20:
            #print("i:",i,"index:",index,"going to add index to index_list")
            index_list.append(index)
            countt_true=countt_true+1
            #print("count_true_label:",countt_true)
        if Y[index] == 0 and  count_false < 44:
            #print("i:", i, "index:", index, "going to add index to index_list")
            index_list.append(index)
            count_false=count_false+1
            #print("count_false_label:",count_false,type(count_false))
    #print("length of index_list:",len(index_list),"index_list:",index_list)
    input_x1=[X1[index] for index in index_list]
    input_x2=[X2[index] for index in index_list]
    input_bluescore = [trainBlueScores[index] for index in index_list]
    input_y=[Y[index] for index in index_list]
    return input_x1,input_x2,input_bluescore,input_y

#do eval and report acc, f1 score
small_value=0.00001
file_object=open('data/log_predict_error.txt','a')
def do_eval(sess,textCNN,evalX1,evalX2,evalBlueScores,evalY,iteration,vocabulary_index2word):
    number_examples=len(evalX1)
    print("valid examples:",number_examples)
    eval_loss,eval_accc,eval_counter=0.0,0.0,0
    eval_true_positive, eval_false_positive, eval_true_negative, eval_false_negative=0,0,0,0
    batch_size=1
    weights_label = {}  # weight_label[label_index]=(number,correct)
    weights = np.ones((batch_size))
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {textCNN.input_x1: evalX1[start:end],textCNN.input_x2: evalX2[start:end], textCNN.input_bluescores:evalBlueScores[start:end],textCNN.input_y:evalY[start:end],
                     textCNN.weights:weights,textCNN.dropout_keep_prob: 1.0,textCNN.iter: iteration,textCNN.tst: True}
        curr_eval_loss,curr_accc, logits= sess.run([textCNN.loss_val,textCNN.accuracy,textCNN.logits],feed_dict)#curr_eval_acc--->textCNN.accuracy
        true_positive, false_positive, true_negative, false_negative=compute_confuse_matrix(logits[0], evalY[start:end][0]) #logits:[batch_size,label_size]-->logits[0]:[label_size]
        write_predict_error_to_file(start,file_object,logits[0], evalY[start:end][0],vocabulary_index2word,evalX1[start:end],evalX2[start:end])
        eval_loss,eval_accc,eval_counter=eval_loss+curr_eval_loss,eval_accc+curr_accc,eval_counter+1
        eval_true_positive,eval_false_positive=eval_true_positive+true_positive,eval_false_positive+false_positive
        eval_true_negative,eval_false_negative=eval_true_negative+true_negative,eval_false_negative+false_negative
        weights_label = compute_labels_weights(weights_label, logits, evalY[start:end]) #compute_labels_weights(weights_label,logits,labels)
    print("true_positive:",eval_true_positive,";false_positive:",eval_false_positive,";true_negative:",eval_true_negative,";false_negative:",eval_false_negative)
    p=float(eval_true_positive)/float(eval_true_positive+eval_false_positive+small_value)
    r=float(eval_true_positive)/float(eval_true_positive+eval_false_negative+small_value)
    f1_score=(2*p*r)/(p+r+small_value)
    print("eval_counter:",eval_counter,";eval_acc:",eval_accc)
    return eval_loss/float(eval_counter),eval_accc/float(eval_counter),f1_score,p,r,weights_label

def write_predict_error_to_file(index,file_object,logit,label,vocabulary_index2word,x1_index_list,x2_index_list):
    #1.if label and predict is not same, write x1,x2,label and predict
    #print("x1_index_list:",x1_index_list.shape)
    predict = np.argmax(logit)
    if predict!=label:
        x1=[vocabulary_index2word[x] for x in list(x1_index_list[0])]
        x2 = [vocabulary_index2word[x] for x in list(x2_index_list[0])]
        file_object.write(str(index)+"-------------------------------------------------------\n")
        file_object.write("label:"+str(label)+";predict:"+str(predict)+"\n")
        file_object.write("".join(x1)+"\n")
        file_object.write("".join(x2) + "\n")


def compute_confuse_matrix(logit,predict):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    label=np.argmax(logit)
    true_positive=0  #TP:if label is true('1'), and predict is true('1')
    false_positive=0 #FP:if label is false('0'),but predict is ture('1')
    true_negative=0  #TN:if label is false('0'),and predict is false('0')
    false_negative=0 #FN:if label is false('0'),but predict is true('1')
    if predict==1 and label==1:
        true_positive=1
    elif predict==1 and label==0:
        false_positive=1
    elif predict==0 and label==0:
        true_negative=1
    elif predict==0 and label==1:
        false_negative=1

    return true_positive,false_positive,true_negative,false_negative



def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textCNN,word2vec_model_path):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    word2vec_model = word2vec.load(word2vec_model_path, kind='txt')
    word2vec_dict = {}

    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
        #print("word2vec_model.word:");print(word)
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    word_embedding_2dlist[1] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(1.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(2, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        #print("word:",word)
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
            #print("embedding:",embedding)
        except Exception:
            embedding = None
            #print("word not exists in word2vec_dict:");print(word)
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    #print("word_embedding_final:",word_embedding_final.shape,word_embedding_final) #8267,))
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("################>>>>>>>word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


if __name__ == "__main__":
    tf.app.run()
