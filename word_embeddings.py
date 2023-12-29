from gensim.models import KeyedVectors
import numpy as np

class WordEmbeddings:
    def __init__(self, model_path):
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def get_word_embedding(self, word):
        if word in self.model:
            return self.model[word]
        else:
            return np.zeros(self.model.vector_size)
