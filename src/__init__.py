from .utils.misc import (extract_tar_gz, save_jsonfile, read_jsonfile,
                         get_all_files_from_data_folder)

from .dataloader.colorspace import (add_alpha_channel, remove_alpha_channel,
                                    rgb2gray, bgr2gray, gray2rgb, gray2bgr)
from .dataloader.degradation import (jpg_compress, salt_and_pepper_noise, gaussian_noise,
                                     tf_jpg_compress, tf_gaussian_noise, tf_salt_and_pepper_noise)
from .dataloader.formatting import image_normalize, image_denormalize, tf_image_normalize
from .dataloader.io import image_read, image_save
from .dataloader.geometric import (image_rotate, image_flip, image_resize, image_merge,
                                   tf_image_flip)

from .dataloader.dataset import image_process, load_image_and_label, create_train_dataset

from .text_mining.nltk import stop_words_filtering

from .text_mining.spacy import lemmatize_english, lemmatize_french, detect_lang

from .text_mining.tesseract import (tesseract_osd, tesseract_ocr, tesseract_ocr_postprocess,
                                    tesseract_image_process, tesseract_data_process)

from .text_mining.text_process import (extract_document_length, calculate_lexical_diversity,
                                       calculate_keyword_density, text_pipeline, text_normalize)

from .text_mining.xml import img_xml_unicode, img_xml_language, img_xml_dimension

from .text_mining.model import TextModel

from .visualization.visualize import (draw_tesseract_result, plot_image,
                                      image_grid_sample, create_wordcloud,
                                      barplot)

from .visualization.compute_color_histogram import compute_color_histogram
from .visualization.training import plot_history


from .extract_info.bluriness import bluriness

from .extract_info.gp_name import gp_name

from .extract_info.img_info import img_info

from .extract_info.extract_image_info import extract_image_info


__all__ = [
    "extract_tar_gz", "save_jsonfile", "read_jsonfile", "plot_image", "image_grid_sample",
    "draw_tesseract_result", "tesseract_ocr", "tesseract_osd", "image_rotate", "text_pipeline",
    "tesseract_ocr_postprocess", "tesseract_image_process", "tesseract_data_process",
    "lemmatize_english", "lemmatize_french", "stop_words_filtering", "img_xml_unicode", "img_xml_language",
    "img_xml_dimension", "create_wordcloud", "detect_lang", "get_all_files_from_data_folder",
    "barplot", "bluriness", "gp_name", "extract_image_info", "img_info", "compute_color_histogram",
    "extract_document_length", "calculate_lexical_diversity", "calculate_keyword_density", "image_read",
    "image_flip", "jpg_compress", "image_resize", "add_alpha_channel", "remove_alpha_channel",
    "rgb2gray", "bgr2gray", "gray2rgb", "gray2bgr", "image_save", "salt_and_pepper_noise",
    "gaussian_noise", "image_normalize", "image_denormalize", "image_merge", "text_normalize",
    "TextModel", "tf_image_normalize", "tf_gaussian_noise", "tf_salt_and_pepper_noise",
    "image_process", "load_image_and_label", "create_train_dataset", "show_history"
]
