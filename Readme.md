# Next Word Predictor

usage
-------
1. N-gram model

* Train and test the model with the following parameters: n, training_file, test_file, output_file, backoff
```shell script
python3 NGram.py --N [n] --training_file [training_file] --test_file [test_file] --output_file [output_file] --backoff [True/False]
```

2. LSTM model  
* download pretrain Glvoe data  
```shell script
cd data
chmod +x download_glove.sh
./download_glove.sh
```
* building vocabulary with pretrain Glove data(e.g. data/glove.6B.300d.txt) and training corpus(e.g. data/train_all.pkl)
```shell script
python3 vocabulary.py [Glove_file] [corpus_file]
```
* training Neural Network with training corpus(e.g. corpus/train_all.pkl)  
required: vocabulary file (data/vocab)  
optional: window length  
```shell script
python3 LSTM.py --train corpus_file [--length N]
```
* testing model(e.g. data/model_all.pt) with testing corpus  
required: vocabulary file (data/vocab)  
optional: window length 
```shell script
python3 LSTM.py --test model_file [--length N]
```
* Next Word Preditor  
required: vocabulary file (data/vocab), model file(e.g. data/model_all.pt) 
```shell script
python3 app.py
```
