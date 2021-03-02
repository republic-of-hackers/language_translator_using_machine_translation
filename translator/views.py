from django.shortcuts import render
from django.http import HttpResponse
from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model





def home(requests):
    context = {
    'access': 'height:20%;display:none;',
    'access1': 'height:20%;display:none;',
    }
    return render(requests, 'home.html', context)

def etg(requests):
    text = [requests.GET['eng_text']]
    trainX = encode_sequences(eng_tokenizer, eng_length, array(text))
    prediction = model1.predict(trainX, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
    	word = word_for_id(i, ger_tokenizer)
    	if word is None:
    		break
    	target.append(word)
    text = ' '.join(target)
    print(text)
    return render(requests, 'home.html', {
    'result': text,
    'access': 'height:20%;display:block;',
    'access1': 'height:20%;display:none;',
    })

def gte(requests):
    text = [requests.GET['ger_text']]
    trainX = encode_sequences(ger_tokenizer, ger_length, array(text))
    prediction = model.predict(trainX, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
    	word = word_for_id(i, eng_tokenizer)
    	if word is None:
    		break
    	target.append(word)
    text = ' '.join(target)

    print(text)
    return render(requests, 'home.html', {
    'result1': text,
    'access1': 'height:20%;display:block;',
    'access': 'height:20%;display:none;',
    })

#Utility

import os
from django.conf import settings

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(os.path.join(settings.BASE_DIR, filename), 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])

model = load_model('model.h5')
model1 = load_model('eng_to_german_model.h5')
