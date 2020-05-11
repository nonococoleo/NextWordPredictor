# Next Word Predictor

usage
-------
1. N-gram model

2. LSTM model  
* building vocabulary with pretrain Glove data(e.g. glove.6B.300d.txt) and training corpus(e.g. data/train_all.pkl)
```shell script
python3 vocabulary.py [Glove_file] [corpus_file]
```
* training Neural Network with training corpus(e.g. data/train_all.pkl)  
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
