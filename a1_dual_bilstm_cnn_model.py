#  -*- coding: utf-8 -*-
# TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
#  print("started...")
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class DualBilstmCnnModel:
    def __init__(self, filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,hidden_size,
                 is_training,initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=3.0,decay_rate_big=0.50,
                 model='dual_bilstm_cnn',similiarity_strategy='additive',top_k=3,max_pooling_style='k_max_pooling',length_data_mining_features=25):
        """init all hyperparameter here"""
        #  set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.is_training=is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")# ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes=filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters=num_filters
        self.initializer=initializer
        self.num_filters_total=self.num_filters * len(filter_sizes) # how many filters totally.
        self.clip_gradients = clip_gradients
        self.model=model
        self.similiarity_strategy=similiarity_strategy
        self.max_pooling_style=max_pooling_style
        self.top_k=top_k
        self.length_data_mining_features=length_data_mining_features

        #  add placeholder (X,label)
        self.input_x1 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x1")  #  X1
        self.input_x2 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x2")  #  X2
        self.input_bluescores= tf.placeholder(tf.float32, [None, self.length_data_mining_features], name="input_bluescores")   #  BLUE SCORES vector
        self.input_y = tf.placeholder(tf.int32, [None,],name="input_y")  #  y:[None,num_classes]
        # self.embedding_trainable_flag = tf.placeholder(tf.bool, name="embedding_trainable_flag")  #  X1

        print("self.input_y:",self.input_y)
        self.weights = tf.placeholder(tf.float32, [None, ], name="weights_label")  #  weights
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32) # training iteration
        self.tst=tf.placeholder(tf.bool)

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.b1_conv1=tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b1_conv2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b1 = tf.Variable(tf.ones([self.hidden_size]) / 10) # embedding_size
        self.b2 = tf.Variable(tf.ones([self.hidden_size]) / 10) # embedding_size
        self.b3 = tf.Variable(tf.ones([self.hidden_size*2]) / 10)  # embedding_size
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        if self.model=='dual_bilstm':
            print("=====>going to start dual bilstm model.")
            self.logits = self.inference_bilstm() # [None, self.label_size]. main computation graph is here.
        elif self.model=='dual_cnn':
            print("=====>going to start dual cnn model.")
            self.logits = self.inference_cnn()
        elif self.model=='dual_bilstm_cnn':
            print("=====>going to start dual_bilstm_cnn model.")
            self.logits=self.inference_bilstm_cnn()
        elif self.model=='shortcut_stacked': # Shortcut-Stacked
            self.logits = self.inference_shortcut_stacked_bilstm()
        elif self.model=='esim':
            print("======>going to use 'enhanced sequential inference model'")
            self.logits = self.inference_esim()
        else:
            print("=====>going to start mix model.")
            self.logits = self.inference_mix()
        # self.possibility=tf.nn.sigmoid(self.logits)
        print("is_training:",is_training)
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")  #  shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) # tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") #  shape=()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): #  embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*2, self.num_classes],initializer=self.initializer) # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       # [label_size] # ADD 2017.06.09

            self.W_LR=tf.get_variable("W_LR",shape=[self.length_data_mining_features,self.num_classes])
            self.b_LR = tf.get_variable("b_LR",shape=[self.num_classes])       # [label_size] # ADD 2017.06.09

            self.W_projection_bilstm = tf.get_variable("W_projection_bilstm", shape=[self.hidden_size, self.num_classes],initializer=self.initializer)  #  [embed_size,label_size]
            self.b_projection_bilstm = tf.get_variable("b_projection_bilstm",shape=[self.num_classes])  #  [label_size] # ADD 2017.06.09

    def inference_mix(self):
        # 1.feature: bilstm features
        x1_rnn=self.bi_lstm(self.input_x1,1) # [batch_size,hidden_size*2]
        x2_rnn=self.bi_lstm(self.input_x2,1,reuse_flag=True) # [batch_size,hidden_size*2]
        x3_rnn=tf.abs(x1_rnn-x2_rnn)
        x4_rnn=tf.multiply(x1_rnn,x2_rnn)
        h_rnn = tf.concat([x1_rnn,x2_rnn,x3_rnn,x4_rnn], axis=1)
        h_rnn= tf.layers.dense(h_rnn, self.hidden_size, use_bias=True,activation=tf.nn.relu)

        # h_rnn = self.additive_attention(x1_rnn, x2_rnn, self.hidden_size/2, "rnn_attention")

        # 2.feature: data mining features
        h_bluescore= tf.layers.dense(self.input_bluescores, self.hidden_size, use_bias=True)
        h_bluescore= tf.nn.relu(h_bluescore)

        # 3.featuere2: cnn features
        # x1=self.conv_layers(self.input_x1, 1)   # [None,num_filters_total]
        # x2= self.conv_layers(self.input_x2, 1,reuse_flag=True) # [None,num_filters_total]
        # h_cnn = self.additive_attention(x1, x2, self.hidden_size/2, "cnn_attention")

        # 4.concat feature
        h = tf.concat([h_rnn, h_bluescore], axis=1)

        # 5.fully connected layer
        h = tf.layers.dense(h, self.hidden_size,activation=tf.nn.relu, use_bias=True)
        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)
        # h,self.update_ema=self.batchnorm(h,self.tst, self.iter, self.b1)

        with tf.name_scope("output"):
            logits=tf.layers.dense(h,self.num_classes, use_bias=False)
        return logits

    def inference_esim(self):
        """
        enhanced sequential inference model. for more check:https://zhuanlan.zhihu.com/p/38256345
        1.input encode with bi-lstm;
        2.local inference modeling-->collected over sequences;
        3.enhance of local information by doing subtract and element-wise multiplication
        4.composition layer(bi-lstm)
        5.max and mean pooling-->concat features
        6.classifier
        :return:
        """
        #  1.input encode with bi-lstm: O.K.
        embedding_x1 = tf.nn.embedding_lookup(self.Embedding,self.input_x1) # shape:[None,sentence_length,embed_size]
        embedding_x2 = tf.nn.embedding_lookup(self.Embedding,self.input_x2) # shape:[None,sentence_length,embed_size]
        a=self.bi_lstm_return_sequences(embedding_x1, 'encoding_input')  # [batch_size,sequence_length,hidden_size*2]
        b=self.bi_lstm_return_sequences(embedding_x2, 'encoding_input', reuse_flag=True)  # [batch_size, sequence_length, hidden_size * 2]

        #  2.local inference model: compute matrix-->weighted sum over axis
        #  2.1 compute matrix. result should be:[batch_size, sequence_length, sequence_length]
        ###below is vinalla implement of compute matrix##########################################
        #b_list = tf.split(b,self.sequence_length,axis=1)  # a list. element is:[batch_size,hidden_size*2]
        #result_list=[]
        #for i,b_sub in enumerate(b_list):  # b_sub is:[batch_size,hidden_size*2]
            #  b_sub:[batch_size,1,hidden_size*2];a:[batch_size,sequence_length,hidden_size*2]
        #    result_sub=tf.multiply(b_sub,a)  # [batch_size,sequence_length,hidden_size*2]
        #    result_sub=tf.reduce_sum(result_sub,axis=-1)  # [batch_size,sequence_length]
        #    result_list.append(result_sub)
        #alignment_matrix=tf.stack(result_list,axis=1)  # [sequence_length,sequence_length].for sequence_length for a
        ###above is vinalla implement of compute matrix###########################################
        alignment_matrix=tf.matmul(b, a, transpose_b=True) #[batch_size,sentence_length,sentence_length]

        #  2.2 collected over sequences
        #  softmax over certain axis
        # get b_bar using weighted sum
        #########below is vinalla implement of collected over sequences for b#############################
        #b_list_new=[]
        #for i, _ in enumerate(b_list):  # b_sub is:[batch_size,hidden_size*2]
        #    attention_score=tf.expand_dims(tf.nn.softmax(alignment_matrix[:,i,:],axis=1),axis=2)  #[batch_size,sequence_length,1]
        #    result_sub=tf.reduce_sum(tf.multiply(attention_score,a),axis=1) # [batch_size,hidden_size*2]<----[batch_size,sequence_length,hidden_size*2]
        #    b_list_new.append(result_sub)
        #b_bar=tf.stack(b_list_new,axis=1)  # [batch_size,sequence_length,hidden_size*2]
        #########above is vinalla implement of collected over sequences for b#############################
        b_bar=tf.matmul(alignment_matrix,a) #matrix:[batch_size,sequence_length,sequence_length];a:[batch_size,sequence_length,hidden_size*2]

        #  get a_bar using weighted_sum
        #a_list_new=[]
        #a_list = tf.split(a, self.sequence_length, axis=1)
        #for i, _ in enumerate(a_list):
        #    attention_score=tf.expand_dims(tf.nn.softmax(alignment_matrix[:,:,i],axis=1),axis=2) #[batch_size,sequence_length,1]
        #    result_sub=tf.reduce_sum(tf.multiply(attention_score,b),axis=1) # [batch_size,hidden_size*2]<----[batch_size,sequence_length,hidden_size*2]
        #    a_list_new.append(result_sub)
        #a_bar=tf.stack(a_list_new,axis=1)  # [batch_size,sequence_length,hidden_size*2]
        a_bar=tf.matmul(alignment_matrix,b) #matrix:[batch_size,sequence_length,sequence_length];a:[batch_size,sequence_length,hidden_size*2]


        # 3.enhance of local information by doing subtract and element-wise multiplication
        ################################################################################################################
        a_minus_a_bar=a-a_bar
        a_multiply_a_bar=tf.multiply(a,a_bar)
        m_a=tf.concat([a,a_bar,a_minus_a_bar,a_multiply_a_bar], axis=-1) #[batch_size,sequence_length,hidden_size*8]

        b_minus_b_ar=b-b_bar
        b_multiply_b_bar=tf.multiply(b,b_bar)
        m_b=tf.concat([b,b_bar,b_minus_b_ar,b_multiply_b_bar],axis=-1) #[batch_size,sequence_length,hidden_size*8]

        # 4.composition layer(bi-lstm): transform & reduce dimensionm-->bi-lstm to encode
        m_a=tf.layers.dense(m_a,self.hidden_size, activation=tf.nn.relu, use_bias=True)  # transform and reduce dimension
        m_a = tf.nn.dropout(m_a, keep_prob=self.dropout_keep_prob)
        m_b=tf.layers.dense(m_b,self.hidden_size, activation=tf.nn.relu, use_bias=True)  # transform and reduce dimension
        m_b = tf.nn.dropout(m_b, keep_prob=self.dropout_keep_prob)
        ################################################################################################################

        m_a_bar=self.bi_lstm_return_sequences(m_a,'composition_layer')  # [batch_size,sequence_length,hidden_size*2]
        m_b_bar=self.bi_lstm_return_sequences(m_b,'composition_layer',reuse_flag=True)  # [batch_size,sequence_length,hidden_size*2]

        # 5.max and mean pooling-->concat features
        m_a_max=tf.reduce_max(m_a_bar, axis=1)  # [batch_size,hidden_size*2]
        m_a_mean=tf.reduce_mean(m_a_bar, axis=1)  # [batch_size,hidden_size*2]
        m_b_max=tf.reduce_max(m_b_bar, axis=1)  # [batch_size, hidden_size*2]
        m_b_mean=tf.reduce_mean(m_b_bar, axis=1)  # [batch_size,hidden_size*2]

        v=tf.concat([m_a_max,m_a_mean,m_b_max,m_b_mean],axis=1)   # [batch_size, hidden_size*8]

        # 6.classifier
        h = tf.layers.dense(v, self.hidden_size,activation=tf.nn.tanh, use_bias=True)
        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)
        # h,self.update_ema=self.batchnorm(h,self.tst, self.iter, self.b1)

        with tf.name_scope("output"):
            logits=tf.layers.dense(h,self.num_classes, use_bias=False)
        return logits

    def inference_shortcut_stacked_bilstm(self):
        """
        shortcut(or residual connected) stacked encoder. check: 'Shortcut-Stacked Sentence Encoders for Multi-Domain Inference, Yixin Nie and Mohit Bansal.'
        #1.multiple layer of bi-lstm as encoder. input of next layer is all previous output and word embedding, or use residual connection between layers.
        #2.max-pooling
        #3. apply three matching methods to the two vectors (i) concatenation (ii) element-wise distance and (iii) element- wise product for these two vectors
            and then concatenate these three match vectors(m)
        #4.feed this final concatenated result m into a MLP layer and use a softmax layer to make final classification.
        :return:
        """
        logits=None
        # 1.multiple layer of bi-lstm as encoder. input of next layer is all previous output and word embedding,
        # or use residual connection between layers.
        x1_embedded = tf.nn.embedding_lookup(self.Embedding, self.input_x1)  # shape:[batch_size,sentence_length,embed_size]
        x2_embedded = tf.nn.embedding_lookup(self.Embedding, self.input_x2)  # shape:[batch_size,sentence_length,embed_size]
        x1_sequences=self.bi_shortcut_stacked_lstm_return_sequences(x1_embedded,'shortcut_stacked')  # [batch_size,sentence_length,hidde_size*2]
        x2_sequences=self.bi_shortcut_stacked_lstm_return_sequences(x2_embedded,'shortcut_stacked',reuse_flag=True)  #[batch_size,sentence_length,hidden_size*2]

        # 2.max-pooling
        x1_rnn=tf.reduce_max(x1_sequences,axis=1)  #[batch_size, hidden_size*2]
        x2_rnn=tf.reduce_max(x2_sequences,axis=1)  #[batch_size, hidden_size*2]

        # 3.apply three matching methods to the two vectors
        # (i) concatenation
        # (ii) element-wise distance and
        # (iii) element- wise product for these two vectors
        # and then concatenate these three match vectors(m)
        x3_rnn=tf.abs(x1_rnn-x2_rnn)
        x4_rnn=tf.multiply(x1_rnn,x2_rnn)
        h_rnn = tf.concat([x1_rnn,x2_rnn,x3_rnn,x4_rnn], axis=1)
        h_rnn= tf.layers.dense(h_rnn, self.hidden_size*2, use_bias=True,activation=tf.nn.relu)
        h = tf.nn.dropout(h_rnn, keep_prob=self.dropout_keep_prob)
        #h = tf.contrib.layers.batch_norm(h_rnn, is_training=self.is_training, scope='shortcut_stacked') #(not self.tst)
        logits = tf.layers.dense(h, self.num_classes, use_bias=False)
        self.update_ema = h  #  TODO need remove

        return logits

    def inference_cnn(self):
        """main computation graph here: 1.get feature of input1 and input2; 2.multiplication; 3.linear classifier"""
        #  1.feature: data mining features
        h_bluescore = tf.layers.dense(self.input_bluescores, self.hidden_size / 2, use_bias=True)
        h_bluescore = tf.nn.relu(h_bluescore)

        #  2.featuere2: cnn features
        x1 = self.conv_layers(self.input_x1, 1)  # [None,num_filters_total]
        x2 = self.conv_layers(self.input_x2, 1, reuse_flag=True)  # [None,num_filters_total]
        h_cnn = self.additive_attention(x1, x2, self.hidden_size / 2, "cnn_attention")

        #  4.concat feature
        h = tf.concat([h_cnn, h_bluescore], axis=1)

        #  5.fully connected layer
        h = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)
        #  h,self.update_ema=self.batchnorm(h,self.tst, self.iter, self.b1)

        with tf.name_scope("output"):
            logits = tf.layers.dense(h, self.num_classes, use_bias=False)
        self.update_ema = h  #  TODO need remove
        return logits


    def inference_bilstm(self):
        # 1.feature:bilstm
        x1=self.bi_lstm(self.input_x1,1) # [batch_size,hidden_size*2]
        x2=self.bi_lstm(self.input_x2,2) # [batch_size,hidden_size*2]

        if self.similiarity_strategy == 'multiply':
            print("similiarity strategy:", self.similiarity_strategy)
            x1=tf.layers.dense(x1,self.hidden_size*2) # [None, hidden_size]
            h_bilstm=tf.multiply(x1,x2) # [None,number_filters_total]
        elif self.similiarity_strategy == 'additive':
            print("similiarity strategy:",self.similiarity_strategy)
            h_bilstm=self.additive_attention(x1,x2,self.hidden_size,"bilstm_attention")

        # 2.feature:data mining feature
        h_bluescore= tf.layers.dense(self.input_bluescores, self.hidden_size/2, use_bias=True)
        h_bluescore= tf.nn.relu(h_bluescore)

        # 3.concat feature
        h = tf.concat([h_bilstm, h_bluescore], axis=1)

        # 4.fully connected layer
        h = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)

        with tf.name_scope("output"):
            logits = tf.layers.dense(h, self.num_classes, use_bias=False)
        return logits

    def inference_bilstm_cnn(self):
        # 1.1 bilstm:get feature of input1 and input2
        x1_bilstm=self.bi_lstm(self.input_x1,1)
        x2_bilstm=self.bi_lstm(self.input_x2,2)

        # 1.2.bilstm:multiplication
        x1_bilstm=tf.layers.dense(x1_bilstm,self.hidden_size*2) # [None, hidden_size]
        h_bilstm=tf.multiply(x1_bilstm,x2_bilstm) # [None,number_filters_total]

        # 2.1:cnn:get feature of input1 and input2
        x1_cnn=self.conv_layers(self.input_x1, 1)   # [None,num_filters_total]
        x2_cnn = self.conv_layers(self.input_x2, 2) # [None,num_filters_total]

        # 2.2 cnn:multiplication
        x1_cnn=tf.layers.dense(x1_cnn,self.num_filters_total) # [None, hidden_size]
        h_cnn=tf.multiply(x1_cnn,x2_cnn) # [None,number_filters_total]

        h=tf.concat([h_bilstm,h_cnn],axis=1)
        print("h concat from bilstm and cnn:",h)
        # 3. fully connected layer
        h = tf.layers.dense(h, self.hidden_size*2, activation=tf.nn.tanh)

        with tf.name_scope("dropout-together"):# TODO TODO TODO TODO TODO
            h=tf.nn.dropout(h,keep_prob=self.dropout_keep_prob) # [None,num_filters_total]TODO TODO TODO TODO TODO

        # 4. linear classifier
        with tf.name_scope("output"):
            logits = tf.matmul(h,self.W_projection_bilstm) + self.b_projection_bilstm  # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits


    def bi_lstm(self,input_x,name_scope,reuse_flag=False):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        # 1.get emebedding of words in the sentence
        embedded_words = tf.nn.embedding_lookup(self.Embedding,input_x) # shape:[None,sentence_length,embed_size]
        # 2. Bi-lstm layer
        #  define lstm cess:get lstm cell output
        with tf.variable_scope("bi_lstm_"+str(name_scope),reuse=reuse_flag):
            lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
            lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) # backward direction cell
            # if self.dropout_keep_prob is not None:
            #     lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            #     lstm_bw_cell=rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
            #  bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
            #                             output: A tuple (outputs, output_states)
            # where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
            outputs,hidden_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,embedded_words,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # creates a dynamic bidirectional recurrent neural network
        # 3. concat output
        feature=tf.concat([hidden_states[0][1],hidden_states[1][1]],axis=1)
        self.update_ema = feature # TODO need remove
        return feature # [batch_size,hidden_size*2]

    def bi_lstm_return_sequences(self,inputs,name_scope,reuse_flag=False):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        # 1.get emebedding of words in the sentence
        # 2. Bi-lstm layer
        #  define lstm cess:get lstm cell output
        with tf.variable_scope("bi_lstm_"+str(name_scope),reuse=reuse_flag):
            lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
            lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) # backward direction cell
            # if self.dropout_keep_prob is not None:
            #lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            #lstm_bw_cell=rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
            #  bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
            #                             output: A tuple (outputs, output_states)
            # where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
            outputs,hidden_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # creates a dynamic bidirectional recurrent neural network
        # 3. concat output. `[batch_size, max_time, cell_fw.output_size]`
        feature=tf.concat([outputs[0],outputs[1]],axis=-1) # [batch_size,max_time*2,cell_fw.output_size]
        self.update_ema = feature # TODO need remove
        return feature # [batch_size,hidden_size*2]

    def bi_shortcut_stacked_lstm_return_sequences_ORIGINAL(self,inputs,name_scope,reuse_flag=False):
        """main computation graph here: 1.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        # 1. Bi-lstm layer
        #  define lstm cell:get lstm cell output
        inputs_copy=inputs
        #layer1
        with tf.variable_scope("bi_lstm_"+str(name_scope)+"1",reuse=reuse_flag):
            lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
            lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) # backward direction cell
            outputs,hidden_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # creates a dynamic bidirectional recurrent neural network
        feature1=tf.concat([outputs[0],outputs[1]],axis=-1) # [batch_size,max_time*2,cell_fw.output_size]

        #layer2
        inputs2=None
        inputs_copy_transform = tf.layers.dense(inputs, self.hidden_size * 2)  # [None, hidden_size]
        with tf.variable_scope("bi_lstm_"+str(name_scope)+"2",reuse=reuse_flag):
            inputs2=inputs_copy_transform+feature1
            lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
            lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) # backward direction cell
            outputs,hidden_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs2,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # creates a dynamic bidirectional recurrent neural network
        feature2=tf.concat([outputs[0],outputs[1]],axis=-1) # [batch_size,max_time*2,cell_fw.output_size]

        #layer3
        with tf.variable_scope("bi_lstm_"+str(name_scope)+"3",reuse=reuse_flag):
            inputs3=inputs2+feature2
            lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
            lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) # backward direction cell
            outputs,hidden_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs3,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # creates a dynamic bidirectional recurrent neural network
        feature=tf.concat([outputs[0],outputs[1]],axis=-1) # [batch_size,max_time*2,cell_fw.output_size]

        self.update_ema = feature # TODO need remove
        return feature # [batch_size,hidden_size*2]

    def bi_shortcut_stacked_lstm_return_sequences(self,inputs,name_scope,reuse_flag=False):
        """main computation graph here: 1.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        # 1. Bi-lstm layer
        #  define lstm cell:get lstm cell output
        inputs_copy=inputs
        #layer1
        #with tf.variable_scope("bi_lstm_"+str(name_scope)+"1",reuse=reuse_flag):
        #    lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
        #    lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) # backward direction cell
        #    outputs,hidden_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # creates a dynamic bidirectional recurrent neural network
        # feature1=tf.concat([outputs[0],outputs[1]],axis=-1) # [batch_size,max_time*2,cell_fw.output_size]
        feature1=self.bi_lstm_unit(inputs, str(name_scope)+"layer_1")

        #layer2
        inputs2=None
        inputs_copy_transform = tf.layers.dense(inputs, self.hidden_size * 2)  # [None, hidden_size]
        inputs2 = tf.concat([inputs_copy, feature1],axis=-1)  # [batch_size,sequence_length, word_embedding+hidden_size*2]
        feature2 = self.bi_lstm_unit(inputs2, str(name_scope) + "layer_2")

        #  with tf.variable_scope("bi_lstm_"+str(name_scope)+"2",reuse=reuse_flag):
        #    lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
        #    lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) # backward direction cell
        #    outputs,hidden_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs2,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # creates a dynamic bidirectional recurrent neural network
        # feature2=tf.concat([outputs[0],outputs[1]],axis=-1) # [batch_size,max_time*2,cell_fw.output_size]

        # layer3
        previous_output = feature2 + feature1
        inputs3 = tf.concat([inputs_copy, previous_output], axis=-1)
        feature = self.bi_lstm_unit(inputs3, str(name_scope) + "layer_3")
        #with tf.variable_scope("bi_lstm_"+str(name_scope)+"3",reuse=reuse_flag):
        #    lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
        #    lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) # backward direction cell
        #    outputs,hidden_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs3,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # creates a dynamic bidirectional recurrent neural network
        #feature=tf.concat([outputs[0],outputs[1]],axis=-1) # [batch_size,max_time*2,cell_fw.output_size]

        self.update_ema = feature # TODO need remove
        return feature # [batch_size,hidden_size*2]

    def bi_lstm_unit(self,inputs,name_scope):
        with tf.variable_scope("bi_lstm_"+str(name_scope),reuse=reuse_flag):
            lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
            lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) # backward direction cell
            outputs,hidden_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # creates a dynamic bidirectional recurrent neural network
        feature=tf.concat([outputs[0],outputs[1]],axis=-1) # [batch_size,max_time*2,cell_fw.output_size]
        return feature

    def bi_lstmX(self,input_x,name_scope,reuse_flag=False):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        # 1.get emebedding of words in the sentence
        embedded_words = tf.nn.embedding_lookup(self.Embedding,input_x) # shape:[None,sentence_length,embed_size]
        # 2. Bi-lstm layer
        #  define lstm cess:get lstm cell output
        with tf.variable_scope("bi_lstm_"+str(name_scope),reuse=reuse_flag):
            lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
            lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) # backward direction cell
            # if self.dropout_keep_prob is not None:
            #     lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            #     lstm_bw_cell=rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
            #  bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
            #                             output: A tuple (outputs, output_states)
            #                                     where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
            outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,embedded_words,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # creates a dynamic bidirectional recurrent neural network
        # 3. concat output
        output_rnn=tf.concat(outputs,axis=2) # [batch_size,sequence_length,hidden_size*2]
        if self.max_pooling_style=='k_max_pooling':
            print("going to use k max_pooling")
            output_rnn=tf.transpose(output_rnn,[0,2,1]) # [batch_size,hidden_size*2,sequence_length]
            output_rnn=tf.nn.top_k(output_rnn,k=self.top_k,sorted=True,name='top_k')[0] # [batch_size,hidden_size*2,self.k]
            feature=tf.reshape(output_rnn,[-1,self.hidden_size*2*self.top_k])
        elif self.max_pooling_style=='max_pooling':
            print("going to use max_pooling")
            feature=tf.reduce_sum(output_rnn,axis=1) # [batch_size,hidden_size*2] # output_rnn_last=output_rnn[:,-1,:] # # [batch_size,hidden_size*2] # TODO
        elif self.max_pooling_style=='chunk_max_pooling':
            print("going to use chunk_max_pooling")
            output_rnn=tf.transpose(output_rnn,[0,2,1]) # [batch_size,hidden_size*2,sequence_length]
            output_rnn=tf.stack(tf.split(output_rnn,self.top_k,axis=-1),axis=2) # [batch_size,hidden_size*2,top_k, seqlence_length/top_k]
            output_rnn = tf.nn.top_k(output_rnn, k=1, name='top_k')[0] # [batch_size,hidden_size*2,top_k, 1]
            feature=tf.reshape(output_rnn,(-1,self.hidden_size*2*self.top_k)) # [batch_size,hidden_size*2*top_k]

        self.update_ema = feature # TODO need remove
        return feature # [batch_size,hidden_size*2]

    def additive_attention(self,x1,x2,dimension_size,vairable_scope):
        with tf.variable_scope(vairable_scope):
            #  v = tf.get_variable("v", shape=[1,self.hidden_size], initializer=tf.random_normal_initializer(stddev=0.1))
            g = tf.get_variable("attention_g", initializer=tf.sqrt(1.0 / self.hidden_size))
            b = tf.get_variable("bias", shape=[dimension_size], initializer=tf.zeros_initializer)
            #  normed_v = g * v * tf.rsqrt(tf.reduce_sum(tf.square(v)))  #  "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks."normed_v=g*v/||v||,
            x1 = tf.layers.dense(x1, dimension_size)  #  [batch_size,hidden_size]
            x2 = tf.layers.dense(x2, dimension_size)  #  [batch_size,hidden_size]
            h = g*tf.nn.relu(x1 + x2 + b)  #  [batch_size,hidden_size]
        return h

    def conv_layers(self,input_x,name_scope,reuse_flag=False):
        """main computation graph here: 1.embedding-->2.CONV-RELU-MAX_POOLING-->3.linear classifier"""
        #  1.=====>get emebedding of words in the sentence
        embedded_words = tf.nn.embedding_lookup(self.Embedding,input_x)# [None,sentence_length,embed_size]
        sentence_embeddings_expanded=tf.expand_dims(embedded_words,-1) # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        #  2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        #  you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope)+"convolution-pooling-%s" %filter_size,reuse=reuse_flag):
                #  ====>a.create filter
                # Layer1:CONV-RELU
                filter=tf.get_variable("filter-%s"%filter_size,[filter_size,self.embed_size,1,self.num_filters],initializer=self.initializer)
                #  ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                # Conv.Returns: A `Tensor`. Has the same type as `input`.
                #          A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                # 1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                # input data format:NHWC:[batch, height, width, channels];output:4-D
                conv=tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1,1,1,1], padding="VALID",name="conv") # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                print("conv1.0:", conv)
                # conv,update_ema_conv1=self.batchnorm(conv,self.tst, self.iter, self.b1_conv1) # TODO TODO TODO TODO TODO
                # print("conv1.1:",conv)
                #  ====>c. apply nolinearity
                b=tf.get_variable("b-%s"%filter_size,[self.num_filters]) # ADD 2017-06-09
                h=tf.nn.relu(tf.nn.bias_add(conv,b),"relu") # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`


                # Layer2:CONV-RELU
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                # TODO h=tf.reshape(h,[-1,self.sequence_length-filter_size+1,self.num_filters,1]) # shape:[batch_size,sequence_length-filter_size+1,num_filters,1]
                # TODO # # filter2 = tf.get_variable("filter2-%s" % filter_size, [1, self.num_filters, 1, self.num_filters],initializer=self.initializer)
                # # # conv2=tf.nn.conv2d(h,filter2,strides=[1,1,1,1],padding="VALID",name="conv2") # shape:[]
                # conv2, update_ema_conv2 = self.batchnorm(conv2, self.tst, self.iter, self.b1_conv2)
                # # # print("conv2:",conv2)
                # # # b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])  #  ADD 2017-06-09
                # # # conv2=conv2+conv
                # # # h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  #  shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                # Layer3:CONV-RELU
                # h = tf.reshape(h, [-1, self.sequence_length - filter_size + 1, self.num_filters, 1]) # shape:[batch_size,sequence_length-filter_size+1,num_filters,1]
                # filter3 = tf.get_variable("filter3-%s" % filter_size, [1, self.num_filters, 1, self.num_filters],initializer=self.initializer)
                # conv3=tf.nn.conv2d(h,filter3,strides=[1,1,1,1],padding="VALID",name="conv3") # shape:[]
                # print("conv3:",conv3)
                # b3 = tf.get_variable("b3-%s" % filter_size, [self.num_filters])  #  ADD 2017-06-09
                # h = tf.nn.relu(tf.nn.bias_add(conv3, b3),"relu3")  #  shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                #  ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #                   ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                #                   strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                # pooled=tf.nn.max_pool(h, ksize=[1,self.sequence_length-filter_size*2+2,1,1], strides=[1,1,1,1], padding='VALID',name="pool")# shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.

                # pooled=tf.nn.max_pool(h, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")# shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input. TODO
                # # # # # max_k_pooling# # # # # # # # # # # # 
                h=tf.reshape(h,[-1,self.sequence_length - filter_size + 1,self.num_filters]) # [batch_size,sequence_length - filter_size + 1,num_filters]
                h=tf.transpose(h, [0, 2, 1]) # [batch_size,num_filters,sequence_length - filter_size + 1]
                h = tf.nn.top_k(h, k=self.top_k, name='top_k')[0]  #  [batch_size,num_filters,self.k]
                h=tf.reshape(h,[-1,self.num_filters*self.top_k]) # TODO [batch_size,num_filters*self.k]
                # # # # # # # # # # # # # # # # # # 
                pooled_outputs.append(h)
        #  3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #          x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #          x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        h_pool=tf.concat(pooled_outputs,1) # shape:[batch_size, num_filters_total*self.k]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        h_pool_flat=tf.reshape(h_pool,[-1,self.num_filters_total*3]) # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        print("h_pool_flat:",h_pool_flat)
        # 4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            h=tf.nn.dropout(h_pool_flat,keep_prob=self.dropout_keep_prob) # [None,num_filters_total]

        return h # ,update_ema_conv1,update_ema_conv2

    def conv_layers_single(self,input_x,name_scope):
        """main computation graph here: 1.embedding-->2.CONV-BN-RELU-POOLING-CONCAT-FC"""
        #  1.=====>get emebedding of words in the sentence
        embedded_words = tf.nn.embedding_lookup(self.Embedding,input_x)# [None,sentence_length,embed_size]
        sentence_embeddings_expanded=tf.expand_dims(embedded_words,-1) # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        #  2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        #  you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope)+"-convolution-pooling-%s" %filter_size):
                #  ====>a.create filter
                filter=tf.get_variable("filter-%s"%filter_size,[filter_size,self.embed_size,1,self.num_filters],initializer=self.initializer)
                #  ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                # Conv.Returns: A `Tensor`. Has the same type as `input`.
                #          A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                # 1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                # input data format:NHWC:[batch, height, width, channels];output:4-D
                conv=tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1,1,1,1], padding="VALID",name="conv") # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # conv,self.update_ema=self.batchnorm(conv,self.tst, self.iter, self.b1) # TODO TODO TODO TODO TODO
                self.update_ema=conv  # NEED REMOVE TODO TODO TODO TODO TODO
                #  ====>c. apply nolinearity
                b=tf.get_variable("b-%s"%filter_size,[self.num_filters]) # ADD 2017-06-09
                h=tf.nn.relu(tf.nn.bias_add(conv,b),"relu") # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                #  ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #                   ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                #                   strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                pooled=tf.nn.max_pool(h, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")# shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)
        #  3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #          x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #          x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        h_pool=tf.concat(pooled_outputs,3) # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        h=tf.reshape(h_pool,[-1,self.num_filters_total]) # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)

        # 4.=====>add dropout: use tf.nn.dropout
        # with tf.name_scope("dropout"):TODO TODO TODO TODO TODO
        #     h=tf.nn.dropout(h,keep_prob=self.dropout_keep_prob) # [None,num_filters_total]TODO TODO TODO TODO TODO
        # feature=tf.layers.dense(h_drop,self.hidden_size,activation=tf.nn.tanh,use_bias=True) # [None,num_filters_total]
        return h # [None,num_filters_total]

    def batchnorm(self,Ylogits, is_test, iteration, offset, convolutional=False): # check:https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.1_batchnorm_five_layers_relu.py# L89
        """
        batch normalization: keep moving average of mean and variance. use it as value for BN when training. when prediction, use value from that batch.
        :param Ylogits:
        :param is_test:
        :param iteration:
        :param offset:
        :param convolutional:
        :return:
        """
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,iteration)  #  adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def loss(self,l2_lambda=0.0003):# 0.0001-->0.0003
        with tf.name_scope("loss"):
            #  input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
                    #  tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=weights)
            # sparse_softmax_cross_entropy
            losses = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits,weights=self.weights);# sigmoid_cross_entropy_with_logits.# losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) #  shape=(?,)
            loss_main=tf.reduce_mean(losses)# print("2.loss.loss:", loss) # shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            # l1_regularizer=tf.contrib.layers.l1_regularizer(l2_lambda*0.3, scope='L1')
            # l1_loss=tf.contrib.layers.apply_regularization(l1_regularizer, weights_list=tf.trainable_variables())

            # loss_cnn=tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits_cnn,weights=self.weights))
            # loss_rnn = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits_rnn, weights=self.weights))
            # loss_bluescore = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits_bluescore, weights=self.weights))

            loss=loss_main+l2_losses # +loss_rnn*0.1+loss_bluescore*0.1 # +l1_loss
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

# test started. toy task: given a sequence of data. compute it's label: sum of its previous element,itself and next element greater than a threshold, it's label is 1,otherwise 0.
# e.g. given inputs:[1,0,1,1,0]; outputs:[0,1,1,1,0].
# invoke test() below to test the model in this toy task.
def test():
    # below is a function test; if you use this for text classifiction, you need to transform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=5
    learning_rate=0.001
    batch_size=8
    decay_steps=1000
    decay_rate=0.95
    sequence_length=5
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=1.0 # 0.5
    filter_sizes=[2,3,4]
    num_filters=128
    multi_label_flag=True
    textRNN=DualBilstmCnnModel(filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training,multi_label_flag=multi_label_flag)
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       for i in range(500):
           input_x=np.random.randn(batch_size,sequence_length) # [None, self.sequence_length]
           input_x[input_x>=0]=1
           input_x[input_x <0] = 0
           input_y_multilabel=get_label_y(input_x)
           loss,possibility,W_projection_value,_=sess.run([textRNN.loss_val,textRNN.possibility,textRNN.W_projection,textRNN.train_op],
                                                    feed_dict={textRNN.input_x:input_x,textRNN.input_y_multilabel:input_y_multilabel,textRNN.dropout_keep_prob:dropout_keep_prob})
           print(i,"loss:",loss,"-------------------------------------------------------")
           print("label:",input_y_multilabel);print("possibility:",possibility)

def get_label_y(input_x):
    length=input_x.shape[0]
    input_y=np.zeros((input_x.shape))
    for i in range(length):
        element=input_x[i,:] # [5,]
        result=compute_single_label(element)
        input_y[i,:]=result
    return input_y

def compute_single_label(listt):
    result=[]
    length=len(listt)
    for i,e in enumerate(listt):
        previous=listt[i-1] if i>0 else 0
        current=listt[i]
        next=listt[i+1] if i<length-1 else 0
        summ=previous+current+next
        if summ>=2:
            summ=1
        else:
            summ=0
        result.append(summ)
    return result


# test()