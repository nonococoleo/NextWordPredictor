import json
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify

import torch
import numpy as np
from Vocabulary import Vocabulary
from LSTM import LSTMClassifier
from pickle import load

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')

with open("data/vocab", "rb") as f:
    app = load(f)

window_length = 3
model = torch.load("data/model_all.pt", map_location=torch.device('cpu'))
print("model loaded")

server = Flask(__name__)
CORS(server, supports_credentials=True)
example = []

@server.route('/char', methods=['POST'])
def responde():
    string = request.form['test']
    example = string.split()
    if string[-1] != " ":
        cur = example.pop().strip()
    else:
        cur = ""
    if len(example) == 0:
        example.append("<PAD>")
    print(example, cur)
    idx = app.get_idx(example[-window_length:])
    choices = []

    test = torch.from_numpy(np.array([idx]))
    results = model(test.to(device)).detach().cpu().numpy()

    for res in results:
        temp = res.argsort()[::-1]
        words = app.get_word(temp)
        for i in words:
            if i != '<UNK>' and i[:len(cur)] == cur:
                choices.append(i)
    return jsonify(dict(enumerate(choices[:10])))


@server.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    server.run(host="0.0.0.0", port=8000)
