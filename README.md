NLU SIMILARITY
-------------------------------------------------------------------------
all kinds of baseline models for modeling tasks with pair of sentences: semantic text similarity(STS), natural language inference(NLI), paraphrase identification(PI), question answering(QA).


1.Desc
-------------------------------------------------------------------------
this repository contain models that learn to detect sentence similarity for natural language understanding tasks.

there are two different kinds of models: 

 1) sentence encoding-based models that separate the encoding of the individual sentences,
 
 2) joint methods that allow to use encoding of both sentences( to use cross-features or attention from one sentence to the other)
 
we will try to cover both of these two methods.

find more about task, data or even start AI completation by check here:

 <a href='https://dc.cloud.alipay.com/index#/topic/data?id=3'>https://dc.cloud.alipay.com/index#/topic/data?id=3</a>

<img src="https://github.com/brightmart/nlu_sim/blob/master/data/nlu_similiarity.jpg"  width="60%" height="60%" />



2.Data Processing: data enhancement and word segmentation strategy
-------------------------------------------------------------------------
 length of sentence. 5 stand for less than 5; 10 stand for great than 5 and less than 10

 source data in .csv file.

 data format: line_no,sentence1,sentence2,label. 4 columns are splitted by "\t"

     001\t question1\t question2\t label

 { 5: 0.11388705332181162, 10: 0.6559243633406191, 15: 0.1654043613073756, 20: 0.04325725613785391})

 as you can see that most of sentences in this task is quite short, short less 15 or 20.

a.swap sentence 1 and sentence 2

       if sentence 1 and sentence 2 represent the same meaning, then sentence 2 and sentence 1 also have same meaning.

       check: method get_training_data() at data_util.py

b.randomly change order given a sentence.

       as same key words in the same may contain most important message in a sentence, change order of these key words should also able to send those message;

       however there may exist cases, which it not count that big percentage, that meaning of sentence way changed when we change order of words.

       check: method get_training_data() at data_util.py

  after data enhancement:length of training data: 81922 ;validation data: 1600; test data:800; percent of true label: 0.217

c.tokenize style

     you can train the model use character, or word or pinyin. for example even you train this model in pinyin, it still can get pretty reasonable performance.

     tokenize sentence in pinyin: we will first tokenize sentence into word, then translate it into pinyin. e.g. it now become: ['nihao', 'wo', 'de', 'pengyou']


3.Feature Engineering
-------------------------------------------------------------------------
get data mining features given two sentences as string.

    1)n-gram similiarity(blue score for n-gram=1,2,3...);

    2) get length of questions, difference of length

    3) how many words are same, how many words are unique

    4) question 1,2 start with how/why/when(wei shen me,zenme，ruhe，weihe）

    5）edit distance

    6) cos similiarity using bag of words for sentence representation(combine tfidf with word embedding from word2vec,fasttext)

    7) manhattan_distance,canberra_distance,minkowski_distance,euclidean_distance

  check data_mining_features method under data_util.py


4.Imbalance Classification for Skew Data
-------------------------------------------------------------------------
   20% percent is postive label, 80% is negative label. by predict negative for all test data, you can get 80% acc, but recall is 0%.

   1)if you random guess, what is f1 score for your task?

   by using random number+ feed forward(fully connected) layer, f1 score is around 0.34. so after lots of work and if your model achive less then 0.4,

   obviously it is not a good model.

   2) how to adjust the weight for each label?

   one way is to calculate validate accuracy for each label after a epoch, and use it as indicator to adjust the weight. set high weight for label with low accuracy.

   but for small dataset, validate accuracy may fluctuate(unstable), so you can use move average of accuracy or set a ceiling value for the weight.

   check weight_boosting.py


5.Transfer Learning & Pretrained Word Embedding
-------------------------------------------------------------------------

   since this is small dataset, transfer learning may be helpful.

   option 1):

   download pretrained word embedding(embedding size is 64) at 80g big files, it has around 90% coverage of words used in this task. it boost peformance around 4%.

   choose .bin file, download from  https://pan.baidu.com/s/1o7MWrnc, password: wzqv


   option 2):
   currently we train word embedding on a 1 million dataset for finance online customer, it has

   around 20k words. in total 8000 unique words in this dataset, around 5000 words also exists in external dataset. after merge this task's dataset and

   external dataset, then train in word2vec, this percentage increase(2336 not exist,5930 exist).

   pretrained word embedding in data\asttext_fin_model_50.vec


6.Models
-------------------------------------------------------------------------
1) DualTextCNN:

      each sentence to be compared is pass to a TextCNN to extract feature(e.g.f1,f2), then use mutliply(f1Wf2, where W is a learnable parameter)

      to learn relationship

2) DualBiLSTM:

      each sentence to be compared is pass to BiLSTM to extract feature(e.g.f1,f2), then use mutliply(f1Wf2, where W is a learnable parameter)

      to learn relationship

3) DualBiLSTMCNN:

     features from DualTextCNN and DualBiLSTM are concated, then send to linear classifier.

4) Pure Data Mining:

     features from data mining and deep learning(CNN/RNN) are sent to FC connected layer, and then to linear classifier.

     check inference_mix in xxx_model.py

5)ESIM: Enhanced LSTM for Natural Language Inference

        encode with bi-lstm--->local inference modeling-->enhance of local information-->composition layer-->pooling

<img src="https://github.com/brightmart/nlu_sim/blob/master/data/enhanced_sequential_inference_model.jpg"  width="60%" height="60%" />

        
6) SSE: Shortcut-Stacked Sentence Encoders for Multi-Domain Inference
        
        shortcut(or residual connected) stacked encoder. --->
        
        multiple layer of bi-lstm as encoder with shortcut or  residual connection between layers.--->
        
        max-pooling --->
        
        apply three matching methods to the two vectors then concatenate these three match vectors(m)
        
        feed this final concatenated result m into a MLP layer and use a softmax layer to make final classification.

<img src="https://github.com/brightmart/nlu_sim/blob/master/data/stacked_shortcut_biLSTM.jpg"  width="60%" height="60%" />


7.Performance
-------------------------------------------------------------------------
   performance on validation dataset(seperate from training data):

Model | Epoch|Loss| Accuracy|F1 Score|Precision|Recall|
---         | ---   | ---   | ---   |---    |---         |---|
DualTextCNN |  9 | 0.833	| 0.689	| 0.390 |	0.443	 | 0.349|
DualTextCNN  | test |  0.915| 0.662 | 0.301 |	0.362    | 0.257|
BiLSTM       |  5   | 0.783 |	0.656|	0.453 |	0.668 |  0.342 |
BiLSTM(pinyin)| 8	|0.816	| 0.685	|0.445|0.587 |0.358 |
BiLSTM(word)  |  5   | 0.567 | 0.74	|	0.503 |	0.576 |  0.445 |
BiLSTMCNN(char)    |  3	| 0.696 |	0.767|	0.380 |	0.311	| 0.487|
BiLSTMCNN(char)    |  9	|   1.131| 0.636 |	0.464|	0.712   |	x  |
BiLSTMCNN(word)    | 9	| 0.775	| 0.639	|0.401	|0.547 |0.316 |
BiLSTMCNN(word,noDataEnhance) | 9	| 0.871	| 0.601 | 0.411 | 0.632	| 0.305

【DualTextCNN2.word.Validation】Epoch 5	 Loss:0.539	Acc 0.794	F1 Score:0.575	Precision:0.604	Recall:0.549

【DualTextCNN2.word.Validation】Epoch 8	 Loss:0.528	Acc 0.787	F1 Score:0.550	Precision:0.586	Recall:0.517

【DualTextCNN2.char.Validation】Epoch 6	 Loss:0.557	Acc 0.766	F1 Score:0.524	Precision:0.580	Recall:0.478

 above f1 score on validation may not 100% true since a bug in previous version. below f1 score on validation is true:

 (data mining features + deep learning features(CNN and or RNN)) + feed foward layer ===> f1 score: 0.55

 ESIM[9]===>f1 score: 0.49 (100k training data,epoch5)
 
 SSE[10]:Shortcut-Stacked Sentence Encoder(residual):0.516
 
 SSE[10]:Shortcut-Stacked Sentence Encoder(stacked):0.511



8.Error Analysis
----------------------------------------------------------------

   #1. i already adjust weights of label, and want to fine tuning weights to get best possible performance. what can i do?

   error analysis! print error case(with information:target label,predicted label,inputs) when you are doing validation for each epoch.

   randomly count examples(e.g. 30 cases). for this 30 cases, how many percent the target label is true, how many percent the target label is false.

   if you set weight for true label to very high value, or not use weight at all, you will get two extremes, change your weight when compute loss so that

   percent of target label==true and target label==false in error case has no big difference.

   error case for different weight:

   log1:not use weight at all:                           target label is false: 16; target label is true: 4.

   log2:set weight for true label to a high value(3).    target label is false: 10; target label is true: 20.

   log3:set weight for true label to a middle value(1.3).target label is false: 10; target label is true: 10.

   #2. where to check detail of error cases?

   see data/log_predict_error.txt


9.Usage
-------------------------------------------------------------------------
  to train the model using sample data, run below command:

  try mix model in word:

  python -u a1_dual_bilstm_cnn_train.py --model_name=mix --ckpt_dir=dual_mix_word_checkpoint/ --tokenize_style=word --name_scope=mix_word

  try mix model in char:
  
  python -u a1_dual_bilstm_cnn_train.py --model_name=mix --ckpt_dir=dual_mix_char_checkpoint/ --tokenize_style=char --name_scope=mix_char


  The following arguments are optional:

    --model                  models that supported {mix,shortcut_stacked,esim,dual_bilstm_cnn,dual_bilstm,dual_cnn} [mix]

    --tokenize_style         how to tokenize the data {char,word,pinyin} [char]

    --similiarity_strategy   how to do match two features {additive,multiply} [additive]
    
    --max_pooling_style     how to do max polling. {chunk_max_pooling, max_pooling,k_max_pooling｝ [chunk_max_polling]


  to make a prediction(and save result in file system), run below command:

  ./run.sh data/test.csv data/target_file.csv

  test.csv is the source file you want to make a prediction, target_file.csv is the predicted result save as file.



10.Environment
-------------------------------------------------------------------------
   python 2.7 + tensorflow 1.8

   for people use python3, just comment out three lines below in the begining of file:

      import sys

      reload(sys)

      sys.setdefaultencoding('utf-8')


11.Model Details
-------------------------------------------------------------------------
   1)DualTextCNN buidling blocks:

         a.tokenize sentence in character way-->embedding-->b.multiply Convolution with multiple filters-->c.BN(o)-->d.activation-->

         e.max_pooling-->f.concat features-->g.dropout(o)-->h.fully connectioned layer(o)

   2)DualBiLSTM:

        a.tokenize sentence in character way-->embedding-->b.BiLSTM-->c.k-maxpooling--->d.similiarity strategy(additive or multiply)

   3)DualBiLSTMCNN:

        a.get first part feature using DualTextCNN-->b.get second part feature using DualBiLSTM-->c.concat features--->d.FC(o)--->e.Dropout(o)-->classifier

   4)Mix: method of data mining features,features from RNN and(or) CNN.

       a. get data mining features like cosine similiarity using sum of word embeddings, get features from CNN and or bi-lstm

       b. combine two kinds of features

       c. fully connected layer + linear classifier

   5)ESIM: Enhanced LSTM for Natural Language Inference

        a.input encode with bi-lstm;

        b.local inference modeling-->collected over sequences;

        c.enhance of local information by doing subtract and element-wise multiplication

        d.composition layer(bi-lstm)

        e.max and mean pooling-->concat features

        f.classifier

   check method of inference_esim under xxx_model.py, for more check <a href='https://arxiv.org/pdf/1609.06038.pdf'>here</a>
   
   6) SSE: Shortcut-Stacked Sentence Encoders for Multi-Domain Inference
  
        shortcut(or residual connection) stacked encoder. in original paper, the input of next layer is concatenation of word embedding 
        
        and all previous layers output. similiarly, when use residual connection the input of next layer is summation of previous layers' input and output.
        
        in my implementation, we use 3 layers of bi-lstm with residual connection.
        
        a.multiple layer of bi-lstm as encoder. input of next layer is all previous output and word embedding, or use residual connection between layers.
        
        b.max-pooling
        
        c.apply three matching methods to the two vectors:
        
              (i) concatenation (ii) element-wise distance and (iii) element- wise product for these two vectors
              
          and then concatenate these three match vectors(m)
          
        d.feed this final concatenated result m into a MLP layer and use a softmax layer to make final classification.
      
      
   check method of inference_shortcut_stacked_bilstm under xxx_model.py. for more check <a href='https://arxiv.org/pdf/1708.02312.pdf'>here</a>



12.TODO
-------------------------------------------------------------------------

   1) extract more data mining features

   2) use traditional machin learning like xgboost,random forest

   3) use pingying to tackle miss typed character

   4) try some classic similiarity network


13.Conclusion
-------------------------------------------------------------------------
   1) for small dataset like this which contains only 40k number of data(stage 1), data mining features are crucial.

   2) combine data mining features with other features and sent to neural network

   3) instead of use big network, small network is better for small dataset.

   4) can not rely on deep learning for small dataset

   5) for skew data, that is imbalance data classification, for example here 20% is positive label, use f1 score to evaluate performance.

   adjust weight for each label will significant imporve performance. it will impose model to pay more attention on those label with higher weight.


14.Reference
-------------------------------------------------------------------------
  1) <a href='https://arxiv.org/pdf/1408.5882v2.pdf'>TextCNN:Convolutional Neural Networks for Sentence Classification</a>

  2) A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification

  3) Deep Learning for Chatbots, Part 2 – Implementing a Retrieval-Based Model in Tensorflow, from www.wildml.com

  4) <a href='https://www.kaggle.com/c/quora-question-pairs/discussion/34355'>Quora Question Pairs-Can you identify question pairs that have the same intent?(1st place solution)</a>

  5) <a href='http://www.sohu.com/a/222501203_717210'>Quora Question Pairs.1st place solution.Chinese translation</a>

  6) <a href='http://nbviewer.jupyter.org/github/danielfrg/word2vec/blob/master/examples/word2vec.ipynb'>Word2Vec tutorial</a>

  7) <a href='https://github.com/facebookresearch/fastText'>fastText:Bag of Tricks for Efficient Text Classification</a>

  8) <a href='https://youtu.be/vA1V8A69e9c'>Abhishek Thakur - Is That a Duplicate Quora Question? (Youtube speech)</a>

  9) <a href='https://arxiv.org/pdf/1609.06038.pdf'>ESIM:Enhanced LSTM for Natural Language Inference</a>

  10) <a href='https://arxiv.org/pdf/1708.02312.pdf'>SSE:Shortcut-Stacked Sentence Encoders for Multi-Domain Inference</a>
  
  11) <a href="https://nlp.stanford.edu/projects/snli/">The Stanford Natural Language Inference (SNLI) Corpus and state of art models</a>
  
  12) <a href="https://arxiv.org/abs/1705.02364">Supervised Learning of Universal Sentence Representations from Natural Language Inference Data</a>
  

if you are smart or can contribute new ideas, join with us.

to be continued. for any problem, contact brightmart@hotmail.com
