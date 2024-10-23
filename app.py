from flask import Flask, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

app = Flask(__name__)
app.static_folder = 'static'

nltk.download('popular')
lemmatizer = WordNetLemmatizer()

# Load Model and Data
model = load_model('model/models.h5')
data = json.loads(open('model/data.json').read())
words = pickle.load(open('model/texts.pkl', 'rb'))
classes = pickle.load(open('model/labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1  # Atur indeks yang sesuai dengan kata ke 1
                break  # Hentikan pencarian setelah menemukan kata
    return np.array(bag)



def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    result = "Maaf, saya tidak mengerti pertanyaan Anda saat ini."
    
    for intent in intents_json['data']:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break
    
    return result


def chatbot_response(msg, intents_data):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents_data)
    return res

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText, data)  # Menambahkan data sebagai argumen

if __name__ == "__main__":
    app.run()
