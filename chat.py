import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
from tensorflow.keras.models import load_model

stemmer = LancasterStemmer()

# Load dữ liệu & model
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

with open("data.pkl", "rb") as f:
    words, labels, training, output = pickle.load(f)

model = load_model("chatbot_model.h5")

# Hàm Bag-of-Words
def bag_of_words(sentence, words):
    bag = [0] * len(words)
    token_words = nltk.word_tokenize(sentence)
    token_words = [stemmer.stem(w.lower()) for w in token_words]

    for s in token_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

print("Chatbot sẵn sàng! (Gõ 'quit' để thoát)")

while True:
    inp = input("Bạn: ")
    if inp.lower() == "quit":
        break

    results = model.predict(np.array([bag_of_words(inp, words)]))[0]
    result_index = np.argmax(results)
    tag = labels[result_index]

    if results[result_index] > 0.7:
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        print("Bot:", random.choice(responses))
    else:
        print("Bot: Mình chưa hiểu ý bạn 🤔")
