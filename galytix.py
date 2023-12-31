# -*- coding: utf-8 -*-
"""Galytix.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HX3-J62JVZmE60VMkEDEBDGECeULqDLv
"""

import pandas as pd
import numpy as np
import gzip
import shutil
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.spatial.distance import euclidean

from google.colab import drive

drive.mount('/content/drive')

from gensim.models import KeyedVectors
import gzip
import numpy as np

from gensim.models import KeyedVectors
from smart_open import open
from gensim.models import word2vec
import gensim.models

model = KeyedVectors.load_word2vec_format('/content/drive/MyDrive/GoogleNews-vectors-negative300.bin.gz', binary=True)

model.save_word2vec_format('vectors.csv')

phrases_df = pd.read_csv('/content/phrases2.csv', header=None, names=['Phrase'])
phrases = phrases_df['Phrase'].tolist()

def calculate_phrase_embedding(phrase):
    words = phrase.split()
    valid_words = [word for word in words if word in model]
    if not valid_words:
        return np.zeros(model.vector_size)
    word_embeddings = [model[word] for word in valid_words]
    normalized_sum = np.sum(word_embeddings, axis=0) / np.linalg.norm(np.sum(word_embeddings, axis=0))
    return normalized_sum

phrases_df['Embedding'] = phrases_df['Phrase'].apply(calculate_phrase_embedding)

phrases_df.drop(0)

def calculate_euclidean_distance(emb1, emb2):
    return euclidean(emb1, emb2)

def calculate_cosine_similarity(emb1, emb2):
    return cosine_similarity([emb1], [emb2])[0][0]

phrases_embeddings = np.vstack(phrases_df['Embedding'].to_numpy())
euclidean_distances_matrix = euclidean_distances(phrases_embeddings)
cosine_similarity_matrix = cosine_similarity(phrases_embeddings)

euclidean_distances_df = pd.DataFrame(euclidean_distances_matrix, index=phrases, columns=phrases)
cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, index=phrases, columns=phrases)

def find_closest_match(input_phrase, distance_metric='cosine'):
    input_embedding = calculate_phrase_embedding(input_phrase)

    if distance_metric == 'cosine':
        distances = [calculate_cosine_similarity(input_embedding, emb) for emb in phrases_embeddings]
    elif distance_metric == 'euclidean':
        distances = [calculate_euclidean_distance(input_embedding, emb) for emb in phrases_embeddings]
    else:
        raise ValueError("Invalid distance metric. Choose 'cosine' or 'euclidean'.")

    closest_index = np.argmin(distances)
    closest_phrase = phrases[closest_index]
    closest_distance = distances[closest_index]

    return closest_phrase, closest_distance

user_input = "How does the forecasted insurance premium penetration in country trend ?"
closest_phrase, distance = find_closest_match(user_input, distance_metric='cosine')
print(f"The closest phrase to '{user_input}' is '{closest_phrase}' with a cosine distance of {distance}")

