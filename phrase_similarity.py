import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

class PhraseSimilarity:
    def __init__(self, word_embeddings, phrases):
        self.word_embeddings = word_embeddings
        self.phrases = phrases

    def calculate_phrase_embedding(self, phrase):
        words = phrase.split()
        valid_words = [word for word in words if word in self.word_embeddings.model]
        if not valid_words:
            return np.zeros(self.word_embeddings.model.vector_size)
        word_embeddings = [self.word_embeddings.model[word] for word in valid_words]
        normalized_sum = np.sum(word_embeddings, axis=0) / np.linalg.norm(np.sum(word_embeddings, axis=0))
        return normalized_sum

    def calculate_similarity_matrix(self, distance_metric='cosine'):
        phrases_embeddings = np.vstack([self.calculate_phrase_embedding(phrase) for phrase in self.phrases])

        if distance_metric == 'cosine':
            return cosine_similarity(phrases_embeddings)
        elif distance_metric == 'euclidean':
            return euclidean_distances(phrases_embeddings)
        else:
            raise ValueError("Invalid distance metric. Choose 'cosine' or 'euclidean'.")
