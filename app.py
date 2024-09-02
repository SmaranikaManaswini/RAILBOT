import random
import json
import pickle
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify
from flask_cors import CORS  
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
CORS(app) 
# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load the intents file
intents_file = 'intents.json'
try:
    with open(intents_file) as file:
        intents = json.load(file)
except FileNotFoundError:
    print(f"Error: '{intents_file}' file not found.")
    exit(1)

# Load the words and classes
words_file = 'words.pkl'
classes_file = 'classes.pkl'
try:
    with open(words_file, 'rb') as f:
        words = pickle.load(f)
    with open(classes_file, 'rb') as f:
        classes = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Load the trained model
model_file = 'my_model.keras'
try:
    model = load_model(model_file)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convert sentence into a bag-of-words numpy array."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the class of the input sentence."""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.02 # Lower the threshold to capture more intents
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    if not results:
        return []  # Return an empty list if no intents are above the threshold

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': float(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Retrieve a response from the matched intent or return an error message."""
    if not intents_list:
        return "Sorry, I couldn't find an appropriate response for that."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "Sorry, I couldn't find an appropriate response for that."

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle chat messages.
    Expects JSON input with a 'message' field.
    Returns JSON with a 'response' field.
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': "Invalid input. Please provide a 'message' field."}), 400

    message = data['message']
    ints = predict_class(message)
    res = get_response(ints, intents)
    return jsonify({'response': res})

@app.route('/')
def home():
    return "Chatbot Backend is Running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    
    
    
    
    
  



