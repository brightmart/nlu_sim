
1.train dual_bilstm in word: #f1 score=0.503

  python -u a1_dual_bilstm_cnn_train.py --model_name=dual_bilstm --ckpt_dir=dual_bilstm_word_checkpoint/ --tokenize_style=word --name_scope=bilstm_word

  python -u a1_dual_bilstm_cnn_train.py --model_name=dual_bilstm --ckpt_dir=dual_bilstm_word_checkpoint2/ --tokenize_style=word --name_scope=bilstm_word

2.train dual_bilstm in char: #f1 score=0.453 ?

  python -u a1_dual_bilstm_cnn_train.py --model_name=dual_bilstm --ckpt_dir=dual_bilstm_char_checkpoint/ --tokenize_style=char --name_scope=bilstm_char


3.train dual_cnn in word:   #f1 score=0.55--0.575

  python -u a1_dual_bilstm_cnn_train.py --model_name=dual_cnn --ckpt_dir=dual_cnn_word_checkpoint/ --tokenize_style=word --name_scope=cnn_word

4.train dual_cnn in char:

  python -u a1_dual_bilstm_cnn_train.py --model_name=dual_cnn --ckpt_dir=dual_cnn_char_checkpoint/ --tokenize_style=char --name_scope=cnn_char


5.ensemble two types of model(4 instances): dual_bilstm(word and char) & dual_cnn(word and char)

  python dual_bilstm_cnn_predict_ensemble.py


6. finally, run following command to make a prediction:

   run.sh data/test.csv data/target_file.csv


