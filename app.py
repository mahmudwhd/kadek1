# import library
import json
from string import punctuation
import random
import pickle
from tensorflow.keras.models import load_model

import time
from datetime import datetime
import pytz
from flask import Flask, request, render_template, send_from_directory
# from flask_ngrok import run_with_ngrok

with open("Dataset Kampus Merdeka.json") as data_file:
    data = json.load(data_file)
model = load_model('bot_model.tf')
le_filename = open("label_encoder.pickle", "rb")
le = pickle.load(le_filename)
le_filename.close()


def preprocess_string(string):
    string = string.lower()
    exclude = set(punctuation)
    string = ''.join(ch for ch in string if ch not in exclude)
    return string

def chat(model, input_data):
    exit = False
    while not exit:
        inp = input_data
        inp = preprocess_string(inp)
        bot = ""
        prob = model.predict([inp])
        results = le.classes_[prob.argmax()]
        if prob.max() < 0.2:
            bot = "Maaf kak, aku ga ngerti"
        else:
            for tg in data['intents']:
                if tg['tag'] == results:
                    responses = tg['responses']
            if results == 'bye':
                exit = True
                print("END CHAT")
            bot = f"{random.choice(responses)}"
        return bot

# app = Flask(__name__, template_folder='/content/drive/MyDrive/Colab Notebooks/deploy')
app = Flask(__name__, static_url_path='/')
# The absolute path of the directory containing PDF files for users to download
# app.config["CLIENT_PDF"] = "/content/drive/MyDrive/Colab Notebooks/deploy"
resultChat = []
resultBot = []
resultTime = []
now = datetime.now(pytz.timezone('Asia/Jakarta'))
time = now.strftime("%I:%M:%S %p %Z %a, %d %b %Y")
resultTime.append(time)