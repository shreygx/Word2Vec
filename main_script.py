from your_module.word_embeddings import WordEmbeddings
from your_module.phrase_similarity import PhraseSimilarity
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    setup_logging()

    model_path = '/path/to/your/word2vec/model.bin.gz'
    word_embeddings = WordEmbeddings(model_path)

    phrases_path = '/path/to/your/phrases.csv'
    phrases = ['list', 'of', 'phrases']  # Replace with actual phrases

    phrase_similarity = PhraseSimilarity(word_embeddings, phrases)
    try:
        similarity_matrix = phrase_similarity.calculate_similarity_matrix(distance_metric='cosine')
        logging.info("Similarity matrix calculated successfully.")
    except ValueError as e:
        logging.error(f"Error calculating similarity matrix: {e}")
        return

    user_input = "How does the forecasted insurance premium penetration in country trend ?"
    try:
        closest_phrase, distance = phrase_similarity.find_closest_match(user_input, distance_metric='cosine')
        logging.info(f"The closest phrase to '{user_input}' is '{closest_phrase}' with a cosine distance of {distance}")
    except ValueError as e:
        logging.error(f"Error finding closest match: {e}")

if __name__ == "__main__":
    main()
