import numpy as np
import scipy.sparse as sp

from pathlib import Path
from typing import Union, Tuple, List, Optional
from nltk.tokenize.regexp import RegexpTokenizer

from src import (image_read, read_jsonfile, tesseract_image_process, detect_lang,
                 stop_words_filtering, lemmatize_french, lemmatize_english, rgb2gray, gray2rgb)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from joblib import load


def text_pipeline(image: Union[np.ndarray, str, Path],
                  common_words_data: Union[dict, str, Path],
                  color_mode: str = 'rgb',
                  tokenizer: Optional[RegexpTokenizer] = None,
                  detect_language: bool = True,
                  language: str = "english",
                  keyword_density_lenght: List[int] = [5, 10, 25, 50],
                  return_tesseract_dataframe: bool = False) -> Tuple[str, List[float]]:
    """Performs preprocessing on text extracted from an image.

    This function applies several preprocessing steps on the text extracted from an image,
    including OCR, tokenization, language detection, stop words filtering, lemmatization, and
    keyword density calculation. The resulting structured text data, along with its features,
    is returned for further processing or prediction.

    Args:
        image (Union[np.ndarray, str, Path]): The image from which to extract text.
            It can be a NumPy array representing the image, a string representing the path to
            the image file, or a pathlib.Path object.
        color_mode (str): Desired color mode ('rgb' or 'gray'). Default is 'rgb'.
        common_words_data (Union[dict, str, Path]): Data containing common words for keyword
            density calculation. It can be a dictionary, a string representing the path to a
            JSON file containing the data, or a pathlib.Path object.
        tokenizer (Optional[RegexpTokenizer], optional): A tokenizer object to use for
            tokenization. If not provided, a default tokenizer for English and French is used.
            Defaults to None.
        detect_language (bool, optional): Whether to detect the language of the text. Defaults
            to True.
        language (str, optional): The language to use for preprocessing. If language detection
            is disabled or inconclusive, this language is used. Defaults to "english".
        keyword_density_lenght (List[int], optional): List of lengths for common words to use
            for keyword density calculation. Defaults to [5, 10, 25, 50].

    Returns:
        Tuple[str, List[float]]: A tuple containing the preprocessed text and its structured
            features. The text is represented as a string, and the features are represented
            as a list of floats.
    """
    # Check image
    if isinstance(image, Path):
        image = str(image)
    if isinstance(image, str):
        image = image_read(image, color_mode)

    if len(image.shape) == 3 and color_mode == "gray":
        image = rgb2gray(image)
    elif len(image.shape) == 2 and color_mode == "rgb":
        image = gray2rgb(image)

    # Check common words data
    if isinstance(common_words_data, Path):
        common_words_data = str(common_words_data)
    if isinstance(common_words_data, str):
        common_words_data = read_jsonfile(common_words_data)

    if tokenizer is None:
        tokenizer = RegexpTokenizer("[a-zA-ZéèàïôêûçëüÉÈÀÏÔÊÛÇËÜ]{3,}")

    # Tesseract process
    ocr_result, osd_result = tesseract_image_process(image)

    # Tokenizer
    words = ' '.join(ocr_result['text'].to_list())
    tokens = tokenizer.tokenize(words.lower())

    # Language detection
    if detect_language:
        lang = detect_lang(tokens)
        language = 'french' if lang['language'] == 'fr' and lang['score'] > 0.5 else 'english'

    # Stop Words Filtering
    words_filter = stop_words_filtering(tokens, language)

    # Lemmatize
    if language == 'french':
        lemmatized_tokens = lemmatize_french(words_filter)
    else:
        lemmatized_tokens = lemmatize_english(words_filter)

    # Words Result
    text = " ".join(lemmatized_tokens)

    # Structure words data
    document_lenght = extract_document_length(text)
    lexical_diversity = calculate_lexical_diversity(text)
    text_structure = [document_lenght, lexical_diversity]
    for category, words in common_words_data.items():
        for k in keyword_density_lenght:
            words = [w[0] for w in common_words_data[category][:k]]
            density = calculate_keyword_density(text, words)
            text_structure.append(density)
    if not return_tesseract_dataframe:
        return text, text_structure
    else:
        return text, text_structure, ocr_result, osd_result


def extract_document_length(text: str) -> int:
    """Extracts the length of a document.

    Calculates the number of words in a given text, representing the length
    of the document.

    Args:
        text (str): The input text for which to extract the document length.

    Returns:
        int: The length of the document, measured in words.
    """
    words = text.split(' ')
    return len(words)

def calculate_lexical_diversity(text: str) -> float:
    """Calculate the lexical diversity of a text.

    Lexical diversity is a measure of how diverse the vocabulary is within a
    given text. It is calculated as the ratio of the number of unique words to
    the total number of words.

    Args:
        text (str): The input text for which to calculate lexical diversity.

    Returns:
        float: The lexical diversity score, ranging between 0.0 and 1.0.
            A higher score indicates a more diverse vocabulary.
    """
    words = text.split(' ')
    return len(set(words)) / len(words)

def calculate_keyword_density(text: str, keywords: List[str]) -> float:
    """Calculate the keyword density of a text.

    Keyword density is a measure of the frequency of specified keywords within
    a text. It is calculated as the ratio of the number of occurrences of
    keywords to the total number of words.

    Args:
        text (str): The input text for which to calculate keyword density.
        keywords (List[str]): A list of keywords to search for in the text.

    Returns:
        float: The keyword density score, ranging between 0.0 and 1.0.
            A higher score indicates a higher density of keywords within the
            text.
    """
    words = text.split(' ')
    keyword_count = sum(1 for word in words if word in keywords)
    return keyword_count / len(words)


def text_normalize(tfidf_vectorizer: Optional[Union[str, TfidfVectorizer]] = None,
                   standard_scaler: Optional[Union[str, StandardScaler]] = None,
                   tfidf_data: Optional[str] = None,
                   structure_data: Optional[list] = None,
                   ) -> np.ndarray:
    """Normalize text and structured data using TF-IDF vectorization and standard
    scaling.

    Args:
        tfidf_vectorizer (Optional[Union[str, TfidfVectorizer]]): A trained TF-IDF
            vectorizer or path to the saved vectorizer.
        standard_scaler (Optional[Union[str, StandardScaler]]): A trained
            StandardScaler or path to the saved scaler.
        tfidf_data (Optional[np.ndarray]): TF-IDF transformed data.
        structure_data (Optional[np.ndarray]): Structured data.

    Returns:
        np.ndarray: Concatenated vector containing the normalized data.
    """
    if tfidf_vectorizer is not None:
        if isinstance(tfidf_vectorizer, str):
            tfidf_vectorizer = load(tfidf_vectorizer)
        X_tfidf = tfidf_vectorizer.transform([tfidf_data])
    else:
        X_tfidf = None

    if standard_scaler is not None:
        if isinstance(standard_scaler, str):
            standard_scaler = load(standard_scaler)
        X_structure = standard_scaler.transform([structure_data])
    else:
        X_structure = None

    if X_tfidf is not None and X_structure is not None:
        X_structure = sp.csr_matrix(X_structure)
        vector = sp.hstack([X_tfidf, X_structure])
    elif X_tfidf is not None and X_structure == None:
        vector = X_tfidf
    else:
        vector = X_structure
    return vector
