import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def stop_words_filtering(words, language):
    """
    Filters out stop words from a list of words and converts the remaining words to lowercase.

    Parameters:
        words (list): A list of words to filter.

    Returns:
        list: A list of filtered words without stop words and converted to lowercase.
    """
    tokens = []
    stop_words = set(stopwords.words(language))
    for word in words:
        if word.lower() not in stop_words:
            tokens.append(word.lower())
    return tokens


# TODO Synchronize the function "stop_words_filtering"
