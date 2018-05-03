NLU SIMILARITY
-------------------------------------------------------------------------
all kinds of baseline models for sentence similarity.

1.Desc
-------------------------------------------------------------------------
this repository contain models that learn to detect sentence similarity.

find more about task, data or even start AI completation by check here:

 <a href='https://dc.cloud.alipay.com/index#/topic/data?id=3'>ATEC蚂蚁开发者大赛人工智能大赛-->金融大脑-金融智能NLP服务</a>


2. Understand your data & data Processing: data enhancement and word segmentation strategy
-------------------------------------------------------------------------
  1).swap sentence 1 and sentence 2

       if sentence 1 and sentence 2 represent the same meaning, then sentence 2 and sentence 1 also have same meaning.

       check: method get_training_data() at data_util.py

  2).randomly change order given a sentence.

       as same key words in the same may contain most important message in a sentence, change order of these key words should also able to send those message;

       however there may exist cases, which it not count that big percentage, that meaning of sentence way changed when we change order of words.

       check: method get_training_data() at data_util.py

  after data enhancement:length of training data: 81922 ;validation data: 1600; test data:800; percent of true label: 0.217

  3).tokenize style

     you can train the model use character, or word or pinyin. for example even you train this model in pinyin, it still can get pretty reasonable performance.

     tokenize sentence in pinyin: we will first tokenize sentence into word, then translate it into pinyin. e.g. it now become: ['nihao', 'wo', 'de', 'pengyou']

3.Models
-------------------------------------------------------------------------
1) DualTextCNN:

      each sentence to be compared is pass to a TextCNN to extract feature(e.g.f1,f2), then use mutliply(f1Wf2, where W is a learnable parameter)

      to learn relationship

2) DualBiLSTM:

      each sentence to be compared is pass to BiLSTM to extract feature(e.g.f1,f2), then use mutliply(f1Wf2, where W is a learnable parameter)

      to learn relationship

3) DualBiLSTMCNN:

     features from DualTextCNN and DualBiLSTM are concated, then send to linear classifier.


4.Performance
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
----------------------------------------------------------------



5.Usage
-------------------------------------------------------------------------
  to train the model using sample data, run below command:

  python -u a1_dual_bilstm_cnn_train.py

  The following arguments are optional:

    --model                  models that supported {dual_bilstm_cnn,dual_bilstm,dual_cnn} [dual_bilstm_cnn]

    --tokenize_style         how to tokenize the data {char,word,pinyin} [char]

    --similiarity_strategy   how to do match two features {additive,multiply} [additive]
    
    --max_pooling_style     how to do max polling. {chunk_max_pooling, max_pooling,k_max_pooling｝ [chunk_max_polling]


  to make a prediction, run below command:

  python -u run.sh data/test.csv data/target_file.csv

  test.csv is the source file you want to make a prediction, target_file.csv is the predicted result save as file.

6.Environment
-------------------------------------------------------------------------
   python 2.7 + tensorflow 1.8

   for people use python3, just comment out three lines below in the begining of file:

      import sys

      reload(sys)

      sys.setdefaultencoding('utf-8')


7.Model Details
-------------------------------------------------------------------------
   1)DualTextCNN buidling blocks:

         a.tokenize sentence in character way-->embedding-->b.multiply Convolution with multiple filters-->c.BN(o)-->d.activation-->

         e.max_pooling-->f.concat features-->g.dropout(o)-->h.fully connectioned layer(o)

   2)DualBiLSTM:

        a.tokenize sentence in character way-->embedding-->b.BiLSTM-->c.k-maxpooling--->d.similiarity strategy(additive or multiply)

   3)DualBiLSTMCNN:

        a.get first part feature using DualTextCNN-->b.get second part feature using DualBiLSTM-->c.concat features--->d.FC(o)--->e.Dropout(o)-->classifier


   Weight enhance:

         as there are only 21.7% true label(1), and 78.3% are false label(0), it is a classic unbalance classification problem.

         we will first compute accuracy for each label, and then use this information to get weight for each label so that we will pay

         more attention to those label with lower accuracy.


         for detail, you can check weight_boosting.py

8.TODO
-------------------------------------------------------------------------
   1) error analysis

   2) understand your data

   3) transfer learning(not limit to pretrained word2vec)

   4) multiple-layer semantic understanding

   5) use pingying to tackle miss typed character


9.Conclusion
-------------------------------------------------------------------------
  this is a placeholder

10.Reference
-------------------------------------------------------------------------
  this is a placeholder

to be continued. for any problem, contact brightmart@hotmail.com
