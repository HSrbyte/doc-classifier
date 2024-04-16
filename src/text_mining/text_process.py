
from typing import List


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