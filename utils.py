import gensim
import nltk
import string
import os
import datetime

target_words = ['egemonizzare', 'lucciola', 'campanello', 'trasferibile', 'brama', 'polisportiva', 'palmare', 'processare', 'pilotato',
                'cappuccio', 'pacchetto', 'ape', 'unico', 'discriminatorio', 'rampante', 'campionato', 'tac' , 'piovra']

# paths
vertical = r"data/vertical/"
raw_lemma_0 = r"data/flat/raw_lemma/T0.txt"
lemma_0 = r"data/flat/lemma/T0.txt"
lemma_1 = r"data/flat/lemma/T1.txt"
corpus_0 = r"data/processed/corpus_0"
corpus_1 = r"data/processed/corpus_1"

# tokenization and stopwords elimination
from nltk.corpus import stopwords

stop_words = stopwords.words('italian')
stop_words.append('é')
stop_words.append('l\'')
stop_words.append('perchè')
stop_words.append('ii')

punctuation = string.punctuation

patterns = r'''
    (?x)                                            # set flag to allow verbose regexps
    (?:[A-Za-z]\.)+                                 # abbreviations, e.g. U.S.A., u.s.a.
   | \w+(?:-\w+)*                                   # words with optional internal hyphens
   | [\€\$\£]?[ \t]?\d+(?:[ \t]?[\.,][ \t]?\d+)?%?  # currency and percentages, e.g. $12.40, 82%
 '''

from nltk.tokenize import RegexpTokenizer
from gensim.models import phrases
from datetime import datetime

import re
expr = r'''_[a-zA-Z0-9\]\[\.,;"\'\?():-_`\€\$\£«»<>\^•]+''' # matches any word after _

def collect_sentences(word, input):
    f = open(('data/target/'+word+'.txt'), 'w', encoding='utf-8')
    lemma = '_'+word+' '
    for line in open(input, encoding='utf-8'):
        if lemma in line.lower():
            raw_line = re.sub(expr, '', line)
            f.write(raw_line)
    f.close()

def parse_conllu(conllu, vertical):
    expr = r'''^[0-9]'''
    for file_name in os.listdir(conllu):
        conllu_file = open(os.path.join(conllu, file_name), 'r', encoding='utf-8')
        vertical_file = open(os.path.join(vertical, file_name.removesuffix('.conllu') + '.txt'), 'w+', encoding='utf-8')
        for line in conllu_file:
            token = ''
            if re.match(expr, line):
                line = line.split()
                if line[2] != '_' and line[3] != '_':
                    token = line[1] + ' ' + line[3] + ' ' + line[2] + '\n'
            elif line.isspace():
                token = '\n'
            vertical_file.write(token)
        vertical_file.close()
        conllu_file.close()

tokenizer = RegexpTokenizer(patterns, discard_empty=True)

def lowercase(token):
    lower_tokens = [w.lower() for w in token]
    return lower_tokens

def tokenization(path):
    tokens = []
    print("[LOG] {} creting tokens...".format(datetime.now()))
    for line in open(path, encoding='utf-8'):
        tokenized_sentence = tokenizer.tokenize(line.lower())
        #token = lowercase(tokenized_sentence)
        tokens.append(tokenized_sentence)
    print("[LOG] {} tokens created!".format(datetime.now()))
    return tokens

def cleaning(tokens):
    filtered_tokens = []
    print("[LOG] {} cleaning tokens...".format(datetime.now()))
    for t in tokens:
        t = [w for w in t if not w in (stop_words or punctuation)]
        filtered_tokens.append(t)
    print("[LOG] {} tokens clean!".format(datetime.now()))
    return filtered_tokens

def filtering(dictionary, tokens, u):
    t2id = dictionary.token2id
    freq = dictionary.cfs
    clean_tokens = []
    print("[LOG] {} filtering tokens...".format(datetime.now()))
    for t in tokens:
        t = [w for w in t if not freq[t2id[w]] <= u]
        clean_tokens.append(t)

    print("[LOG] {} tokens filtered!".format(datetime.now()))
    return clean_tokens

def ngram(tokens, min, th, v):
    # min_count is the occurrence of a single token in a bi-gram, not of the bi-gram itself
    ngrams = phrases.Phrases(sentences=tokens, min_count=min, threshold=th, max_vocab_size=v)
    print("[LOG] {} n-grams collected".format(datetime.now()))
    return ngrams

def bigram_dictionary(bigram, tokens):
    bigram_dict = {}
    # through statistical computation, without any dictionary
    print("[LOG] {} building bigrams dictionary...".format(datetime.now()))
    for t in tokens:
        for g in bigram[t]:  # list of strings which do not contains bi-grams only, also tokens of two words
            if "_" in g:  # '_' is used to concatenate words into the bigram
                if g in bigram_dict:
                    bigram_dict[g] = bigram_dict[g] + 1
                else:
                    bigram_dict[g] = 1
    print("[LOG] {} bigrams dictionary built!".format(datetime.now()))

    return bigram_dict

# flattening and serialization
import pickle
import vertical2flat
from datetime import datetime
import importlib
importlib.reload(vertical2flat)

def flatten(input, output, m):
    vertical2flat.main(folder_in=input, folder_out=output, mode=m)

def save(object, path):
    file = open(path, 'wb')
    pickle.dump(object, file)
    file.close()
    print("[LOG] {} object successfully serialized in {}!".format(datetime.now(), path))

def load(path):
    file = open(path, 'rb')
    eof = False
    while not eof:
        try:
            object = pickle.load(file)
            print("[LOG] {} object successfully loaded from {}!".format(datetime.now(), path))
        except EOFError:
            eof = True
    file.close()

    return object

# semantic evaluation

from nltk.corpus import wordnet as wn

def path_similarity(target, context):
    target_synset = wn.synsets(target, lang='ita')
    if target_synset != []:
        similarities = []
        for w in context:
            synset = wn.synsets(w[0], lang='ita')
            max_similarity = -1
            for s in synset:
                for t in target_synset:
                    if t.path_similarity(s) > max_similarity:
                        max_similarity = t.path_similarity(s)
            if max_similarity > 0:
                similarities.append(max_similarity)
        s = sum(similarities)/len(similarities)
    else:
        s = 0.0

    return s

def noise(word_vectors, top_words, D):
    noise = {}
    for t in target_words:
        context = word_vectors.most_similar(t, topn=top_words)
        context_lemmas = set([c[0] for c in context])
        noise[t] = len(D.intersection(context_lemmas)) / top_words
    return noise

def vector_to_text(wv, path):
    words = wv.index_to_key
    file = open(path, 'w', encoding='utf-8')
    print("[LOG] {} writing vectors...".format(datetime.now()))
    for w in words:
        new_line = w
        for i in range(0, len(wv[w])):
            new_line = new_line + '\t' + str(wv[w][i])
        file.write((new_line + '\n'))
    file.close()
    print("[LOG] {} vectors successfully written in {}".format(datetime.now(), path))

def text_to_vector(text):
    file = open(text, 'r', encoding='utf-8')
    wv = {}
    print("[LOG] {} writing vectors...".format(datetime.now()))
    for line in file:
        items = line.split('\t')
        values = []
        for i in range(1,len(items)):
            values.append(float(items[i]))
        wv[items[0]] = values
    file.close()
    print("[LOG] {} vectors successfully written".format(datetime.now()))
    return wv

import sklearn.metrics.pairwise as metrics
import numpy as np

def compute_distance_file(aligned_vectors_0, aligned_vectors_1, th, path):
    distances = {}
    for t in target_words:
        distances[t] = metrics.cosine_distances(np.reshape(aligned_vectors_0[t], (1, -1)), np.reshape(aligned_vectors_1[t], (1, -1)))

    submission = open(path, 'w', encoding='utf-8')
    for t in target_words:
        if distances[t] > th:
            submission.write(t + '\t 1\n')
        else:
            submission.write(t + '\t 0\n')
    submission.close()

    return distances

def compute_cosine_distance(aligned_vectors_0, aligned_vectors_1):
    distances = {}
    for t in target_words:
        distances[t] = metrics.cosine_distances(np.reshape(aligned_vectors_0[t], (1, -1)), np.reshape(aligned_vectors_1[t], (1, -1)))
    return distances


def prepare_labels(distances, th):
    truth = {}
    with open(r"data/evaluation/gold.txt", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            truth[line[0]] = int(line[1])

    prediction = {}
    for t in truth.keys():
        if distances[t] > th:
            prediction[t] = 1
        else:
            prediction[t] = 0

    y_true = []
    y_predict = []
    for w in truth.keys():
        y_true.append(truth[w])
        y_predict.append(prediction[w])

    return y_true, y_predict

from sklearn.metrics import accuracy_score

def compute_accuracy(aligned_vector_0, aligned_vector_1):
    accuracy = 0
    for th in np.arange(0.0, 1.0, 0.01):
        distances = compute_cosine_distance(aligned_vector_0, aligned_vector_1)
        y_true, y_predict = prepare_labels(distances, th)
        acc = accuracy_score(y_true, y_predict)
        if acc > accuracy:
            accuracy = acc
            threshold = th

    return accuracy, threshold