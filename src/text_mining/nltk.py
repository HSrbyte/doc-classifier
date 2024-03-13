import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def stop_words_filtering(words, language):
    """
    Filter stop words from a list of words.

    This function removes stop words from a list of words based on the specified language.

    Parameters:
        words (list): A list of words to filter.
        language (str): The language of the stop words to use. Supported languages include 'english', 'french', etc.

    Returns:
        list: A list of words with stop words removed.
    """
    tokens = []
    stop_words = set(stopwords.words(language))
    for word in words:
        if word.lower() not in stop_words:
            tokens.append(word.lower())

    return tokens
