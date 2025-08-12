import json
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import pickle

nltk.download('punkt')
stemmer = LancasterStemmer()

# Load dữ liệu
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

words, labels, docs_x, docs_y = [], [], [], []

# Tiền xử lý patterns
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        docs_x.append(tokens)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# Bag of Words
training, output = [], []
out_empty = [0] * len(labels)

for x, doc in enumerate(docs_x):
    bag = []
    stemmed_words = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        bag.append(1 if w in stemmed_words else 0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Lưu dữ liệu tiền xử lý
with open("data.pkl", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# Xây mô hình
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(len(output[0]), activation="softmax"))

# Compile & Train
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.fit(training, output, epochs=500, batch_size=8, verbose=1)

model.save("chatbot_model.h5")
print("Huấn luyện xong, mô hình đã được lưu!")
