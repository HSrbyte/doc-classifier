import spacy


def lemmatize_english(words):
    """
    Lemmatizes a list of English words using spaCy's English language model.

    Parameters:
        words (list): A list of English words to lemmatize.

    Returns:
        list: A list of lemmatized words.
    """
    nlp = spacy.load("en_core_web_sm")

    # Lemmatize each word using spaCy French model
    lemmatized_words = []
    for word in words:
        for token in nlp(word):
            lemmatized_words.append(token.lemma_)

    return lemmatized_words


def lemmatize_french(words):
    """
    Lemmatizes a list of French words using spaCy's French language model.

    Parameters:
        words (list): A list of French words to lemmatize.

    Returns:
        list: A list of lemmatized words.
    """
    nlp = spacy.load("fr_core_news_sm")

    # Lemmatize each word using spaCy French model
    lemmatized_words = []
    for word in words:
        for token in nlp(word):
            lemmatized_words.append(token.lemma_)

    return lemmatized_words


# TODO Synchronize the functions and test them
