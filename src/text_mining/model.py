from src import text_pipeline, text_normalize, read_jsonfile

from typing import Optional, Union, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from joblib import load
from pathlib import Path
import numpy as np



class TextModel:
    """A class representing a document classification model.

    This class provides methods to preprocess and predict categories of textual data.

    Attributes:
        model_path (str): The path to the trained classification model.
        tfidf_vectorizer_path (str): The path to the TF-IDF vectorizer used for text preprocessing.
        standard_scaler_path (str): The path to the standard scaler used for feature scaling.
        common_words_path (str): The path to the JSON file containing common words data.
        categories (dict): A dictionary mapping category indices to category names.
            Default categories are used if not provided.
    """

    default_categories: Dict[int, str] = {
        0: "email",
        1: "handwritten",
        2: "invoice",
        3: "national_identity_card",
        4: "passport",
        5: "scientific_publication"
    }
    def __init__(self,
                 model_path: str,
                 tfidf_vectorizer_path: str,
                 standard_scaler_path: str,
                 common_words_path: str,
                 categories: Optional[Dict[int, str]] = None):

        self.model_path = model_path
        self.tfidf_vectorizer_path = tfidf_vectorizer_path
        self.standard_scaler_path = standard_scaler_path
        self.common_words_path = common_words_path
        self.load_models()

        if categories is None:
            categories = self.default_categories
        self.categories = categories

    @property
    def model(self):
        return self._model

    @property
    def tfidf(self) -> TfidfVectorizer:
        return self._tfidf

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler

    @property
    def common_words(self) -> dict:
        return self._common_words

    def load_models(self) -> None:
        """Loads the trained model, TF-IDF vectorizer, and standard scaler."""
        self._model = load(self.model_path)
        self._common_words = read_jsonfile(self.common_words_path)

        if self.tfidf_vectorizer_path is not None:
            self._tfidf = load(self.tfidf_vectorizer_path)
        else:
            self._tfidf = None

        if self.standard_scaler_path is not None:
            self._scaler = load(self.standard_scaler_path)
        else:
            self._scaler = None

    def preprocess(self, file: Union[Path, str, np.ndarray]) -> np.ndarray:
        """Preprocesses the input image file or array.

        Args:
            file (Union[Path, str, np.ndarray]): The input image file path or array.

        Returns:
            np.ndarray: The preprocessed text vector.
        """
        self.words, self.words_structure, self.ocr_result, self.osd_result = text_pipeline(file, self.common_words, return_tesseract_dataframe=True)

        if self._tfidf is not None and self.scaler is not None:
            vector = text_normalize(self.tfidf, self.scaler, self.words, self.words_structure)
        elif self._tfidf is not None:
            vector = text_normalize(self.tfidf, None, self.words, None)
        else:
            vector = text_normalize(None, self.scaler, None, self.words_structure)

        return vector

    def predict(self, file: Union[Path, str, np.ndarray]) -> str:
        """Predicts the category of the input image file or array.

        Args:
            file (Union[Path, str, np.ndarray]): The input image file path or array.

        Returns:
            str: The predicted category.
        """
        vector = self.preprocess(file)
        result = self.model.predict(vector)
        return self.categories[result[0]]

    def predict_proba(self, file: Union[Path, str, np.ndarray]) -> List[float]:
        vector = self.preprocess(file)
        result = self.model.predict_proba(vector)
        return result