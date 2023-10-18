import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

'''
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."
'''

import googlemaps

# Remplacez 'YOUR_API_KEY' par votre clé d'API réelle
gmaps = googlemaps.Client(key='AIzaSyC2fnlzGl0NmQLH8uMWoDTQqiC8X4tfjAA')

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == 'geocoding':
                    address = extract_address_from_sentence(sentence)  # Fonction à implémenter pour extraire l'adresse
                    geocode_result = gmaps.geocode(address)
                    if geocode_result:
                        location = geocode_result[0]['geometry']['location']
                        latitude = location['lat']
                        longitude = location['lng']
                        return intent['responses'][0].format(
                            adresse=address,
                            latitude=latitude,
                            longitude=longitude
                        )
                    else:
                        return "Je n'ai pas pu trouver les coordonnées pour cette adresse."
                else:
                    return random.choice(intent['responses'])

    return "Je ne comprends pas..."

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

def extract_address_from_sentence(sentence):
    sentence = ' '.join(sentence)  # Convertir la liste de mots en une chaîne de caractères
    address_pattern = r"(\d+\s+[^,]+,\s*[^,]+,\s*\d+)"
    match = re.search(address_pattern, sentence)
    if match:
        return match.group(1)
    else:
        return None

import re

def is_potential_address(entity):
    address_patterns = [
        r"\d+\s+\w+\s+\w+",
        r"\d+\s+\w+",
        r"\d+\s+\w+,\s+\w+",
        r"\d+,\s+\w+",
        r"\d+\s+\w+,\s+\w+,\s+\w+",
        r"\d+\s+\w+.*"
        # Ajoutez d'autres motifs d'adresse au besoin
    ]

    for pattern in address_patterns:
        if re.search(pattern, entity, re.IGNORECASE):
            return True

    return False


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

