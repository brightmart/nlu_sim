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

  2).randomly change order given a sentence

       as same key words in the same may contain most important message in a sentence, change order of these key words should also able to send those message;

       however there may exist cases, which it not count that big percentage, that meaning of sentence way changed when we change order of words.


3.Models
-------------------------------------------------------------------------
1) TextCNNSim:

      each sentence to be compared is pass to a TextCNN to extract feature(e.g.f1,f2), then use mutliply(f1Wf2, where W is a learnable parameter)

      to learn relationship


4.Performance
-------------------------------------------------------------------------
   performance on validation dataset(seperate from training data):

   【Validation】,Epoch 9, Loss:0.703, Acc 0.742,F1 Score:0.327, Precision:0.300, Recall:0.358



5.Usage
-------------------------------------------------------------------------
  python -u a1_dual_cnn_model.py


6.Environment
-------------------------------------------------------------------------
   python 3.6 + tensorflow 1.6

   for people use python2.7, add three lines below:
      import sys
      reload(sys)
      sys.setdefaultencoding('utf-8')


7.Model Details
-------------------------------------------------------------------------
   1)TextCNN building block:

         a.tokenize sentence in character way-->embedding-->b.multiply Convolution with multiple filters-->c.BN(o)-->d.activation-->

         e.max_pooling-->f.concat features-->g.dropout(o)-->h.fully connectioned layer(o)

   2)Weight enhance:

         as there are only 21.7% true label(1), and 78.3% are false label(0), it is a classic unbalance classification problem.

         we will first compute accuracy for each label, and then use this information to get weight for each label so that we will pay

         more attention to those label with lower accuracy.


         for detail, you can check weight_boosting.py

8.TODO
-------------------------------------------------------------------------
   dual_bilstm_model

   bilstm_cnn_model


9.Conclusion
-------------------------------------------------------------------------
  this is a placeholder

10.Reference
-------------------------------------------------------------------------
  this is a placeholder

to be continued. for any problem, concat brightmart@hotmail.com
