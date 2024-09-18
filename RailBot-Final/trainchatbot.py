import json
import numpy as np
import nltk
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD



# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load your dataset (in JSON format)
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Preprocess the data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))
print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Initialize the bag of words
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Output is a '0' for each tag and '1' for the current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle the features and convert into a numpy array
np.random.shuffle(training)
training = np.array(training, dtype=object)

# Create training lists
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))
# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
#optimizer = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
optimizer =SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
from keras.optimizers import Adam

# Recompile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
#hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
history = model.fit(train_x, train_y, epochs=10,batch_size=5, verbose=1 )
# Save the model
model.save('my_model.keras')