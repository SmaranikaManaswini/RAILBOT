import random
import json
import pickle
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load the intents file
try:
    with open('intents.json') as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: 'intents.json' file not found.")
    exit(1)

# Load the words and classes
try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Load the model
try:
    model = load_model('my_model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    #print(f"BOW: {bow}")  # Debugging line to see the bag of words
    res = model.predict(np.array([bow]))[0]
    #print(f"Model output: {res}")  # Debugging line to see model output
    
    ERROR_THRESHOLD = 0.01 # Lower the threshold to capture more intents
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
   #print(f"Results above threshold: {results}")  # Debugging line to see filtered results
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    #print(f"Return list: {return_list}")  # Debugging line to see final return list
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:  # Check if intents_list is empty
        return "Sorry, I didn't understand that. Can you please rephrase?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("GO! Bot is running!")

while True:
    message = input("You: ")
    if message.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(f"Bot: {res}")
