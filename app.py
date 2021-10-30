from flask import Flask, render_template, request
import spacy 
from spacy import displacy
import numpy as np
import joblib
import torch
import torch.nn as nn
from src import config
from src import dataset
from src import engine
from src.model import EntityModel
from transformers import pipeline
Model_path = './src/model.bin'

app = Flask(__name__)

def predict(sentence):

    meta_data = joblib.load("./src/meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    tokenized_sentence = config.TOKENIZER.encode(sentence)


    sentence = sentence.split()
    

    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        pos=[[0] * len(sentence)], 
        tags=[[0] * len(sentence)]
    )

    device = torch.device("cpu")
    MODEL = EntityModel(num_tag=num_tag, num_pos=num_pos)
    MODEL.load_state_dict(torch.load(Model_path, map_location=torch.device("cpu")))
    MODEL.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _ = MODEL(**data)
        print(tag)

        
        out =    enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
    
    return out
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input', methods=['POST', 'GET'])
def input():
    out = []
    tokenized = []
    length = 0
    if request.method == "POST":
        text = request.form['input']
        text = text.strip()

        out = predict(text)
        

        tokenized = []
        t = config.TOKENIZER.tokenize(text)
        print(t)
        for i in config.TOKENIZER.tokenize(text):
            if 'Ġ' in i:
                i = i.replace('Ġ','')
                tokenized.append(i)
            else:
                tokenized.append(i)
        out = out[1:-1]
        length = len(tokenized)
        print(len(out), len(tokenized))
        print(tokenized)
    
    
       
    return render_template('index.html',out= out, tokenized=tokenized, length = length)

if __name__ == '__main__':
    app.run(debug=True)