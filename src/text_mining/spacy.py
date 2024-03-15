import spacy

def check_and_download_model(model_name: str) -> None:
    """Check if the specified SpaCy model is installed. If not, download it.

    Parameters:
        model_name (str): The name of the SpaCy model to check and download.

    Returns:
        None
    """
    if model_name not in spacy.util.get_installed_models():
        print(f"Downloading the {model_name} model...")
        try:
            spacy.cli.download(model_name)
        except Exception as e:
            print(f"Error: Unable to download the {model_name} model.\n{e}")


def lemmatize_english(words):
    """
    Lemmatizes a list of English words using spaCy's English language model.

    Parameters:
        words (list): A list of English words to lemmatize.

    Returns:
        list: A list of lemmatized words.
    """
    check_and_download_model("en_core_web_sm")
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
    check_and_download_model("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

    # Lemmatize each word using spaCy French model
    lemmatized_words = []
    for word in words:
        for token in nlp(word):
            lemmatized_words.append(token.lemma_)

    return lemmatized_words
