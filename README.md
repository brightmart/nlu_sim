NLU SIMILARITY
-------------------------------------------------------------------------
all kinds of baseline models for sentence similarity.

1.Desc
-------------------------------------------------------------------------
this repository contain models that learn to detect sentence similarity.

2.Data enhancement
-------------------------------------------------------------------------
  1).swap sentence 1 and sentence 2

       if sentence 1 and sentence 2 represent the same meaning, then sentence 2 and sentence 1 also have same meaning.

       check: method get_training_data() at data_util.py

  2).randomly change order given a sentence.

       as same key words in the same may contain most important message in a sentence, change order of these key words should also able to send those message;

       however there may exist cases, which it not count that big percentage, that meaning of sentence way changed when we change order of words.

       check: method get_training_data() at data_util.py

  after data enhancement:length of training data: 81922 ;validation data: 1600; test data:800; percent of true label: 0.217

  3) tokenize style

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
BiLSTM(pinyin)| 9	|0.876	| 0.635	|0.441|0.669 |0.329 |
BiLSTMCNN(char)    |  3	| 0.696 |	0.767|	0.380 |	0.311	| 0.487|
BiLSTMCNN(char)    |  9	|   1.131| 0.636 |	0.464|	0.712   |	x  |
BiLSTMCNN(word)    | 9	| 0.775	| 0.639	|0.401	|0.547 |0.316 |
BiLSTMCNN(word,noDataEnhance) | 9	| 0.871	| 0.601 | 0.411 | 0.632	| 0.305

----------------------------------------------------------------



5.Usage
-------------------------------------------------------------------------
  python -u a1_dual_cnn_model.py

  The following arguments are optional:

    --model           models that supported {dual_bilstm_cnn,dual_bilstm,dual_cnn} [dual_bilstm_cnn]

    --tokenize_style  how to tokenize the data {char,word,pinyin} [char]


6.Environment
-------------------------------------------------------------------------
   python 3.6 + tensorflow 1.6

   for people use python2.7, add three lines below:
      import sys
      reload(sys)
      sys.setdefaultencoding('utf-8')


7.Model Details
-------------------------------------------------------------------------
   1)DualTextCNN buidling blocks:

         a.tokenize sentence in character way-->embedding-->b.multiply Convolution with multiple filters-->c.BN(o)-->d.activation-->

         e.max_pooling-->f.concat features-->g.dropout(o)-->h.fully connectioned layer(o)

   2)DualBiLSTM:

        a.tokenize sentence in character way-->embedding-->b.BiLSTM--->two features are multiplied with learnable parameter.

   3)DualBiLSTMCNN:

        a.get first part feature using DualTextCNN-->b.get second part feature using DualBiLSTM-->c.concat features--->d.FC(o)--->e.Dropout(o)-->classifier


   Weight enhance:

         as there are only 21.7% true label(1), and 78.3% are false label(0), it is a classic unbalance classification problem.

         we will first compute accuracy for each label, and then use this information to get weight for each label so that we will pay

         more attention to those label with lower accuracy.


         for detail, you can check weight_boosting.py

8.TODO
-------------------------------------------------------------------------
   1) dual_bilstm_model

   2) bilstm_cnn_model

   3) transfer learning(not limit to pretrained word2vec)

   4) multiple-layer semantic understanding

   5) use pingying to tackle miss typed character


9.Conclusion
-------------------------------------------------------------------------
  this is a placeholder

10.Reference
-------------------------------------------------------------------------
  this is a placeholder

to be continued. for any problem, concat brightmart@hotmail.com
