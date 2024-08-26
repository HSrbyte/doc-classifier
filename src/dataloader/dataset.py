import cv2
import random
import numpy as np
import tensorflow as tf

from typing import Tuple, Union, Optional, List
from src import (image_read, image_resize, tf_image_normalize, tf_jpg_compress,
                 tf_image_flip, tf_gaussian_noise, tf_salt_and_pepper_noise)




def image_process_train(file_fath: str) -> tf.Tensor:
    """
    Processes an image by applying various augmentations and normalization.

    Args:
        file_path (str): The path to the image file.

    Returns:
        tf.Tensor: The processed image as a TensorFlow tensor.
        
    The processing steps include:
        1. Reading the image in either 'rgb' or 'gray' mode.
        2. Converting grayscale images to RGB.
        3. Converting the image to a TensorFlow tensor.
        4. Compressing the image using JPEG with a random quality.
        5. Randomly flipping the image.
        6. Adding salt and pepper noise with random densities.
        7. Adding Gaussian noise with random mean and standard deviation.
        8. Resizing the image to 224x224 pixels.
        9. Normalizing the image with specified mean and standard deviation.
    """
    color_mode = random.choice(['rgb', 'gray'])
    image = image_read(file_fath, color_mode=color_mode)
    if color_mode == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf_jpg_compress(image, quality=random.randint(25,100))
    if tf.random.uniform([]) > 0.5:
        image = tf_image_flip(image, tf.random.uniform([], minval=0, maxval=3, dtype=tf.int32))
    image = tf_salt_and_pepper_noise(image,
                                  salt_density=tf.random.uniform([], 0.0, 0.05),
                                  pepper_density=tf.random.uniform([], 0.0, 0.05))
    image = tf_gaussian_noise(image, mean=0, std_dev=tf.random.uniform([], 0.0, 20.0))
    image = image.numpy().astype(np.float32)
    image = image_resize(image, [224, 224])
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf_image_normalize(image,
                            mean = [176.83188398, 177.25901738, 177.25890556],
                            std = [110.50753887, 110.49521524, 110.50979735])
    return image


def image_process_test(file_fath: Union[str, np.ndarray]) -> tf.Tensor:
    """
    Processes an image by applying various augmentations and normalization.

    Args:
        file_path (str): The path to the image file.

    Returns:
        tf.Tensor: The processed image as a TensorFlow tensor.

    The processing steps include:
        1. Reading the image in either 'rgb' or 'gray' mode.
        8. Resizing the image to 224x224 pixels.
        9. Normalizing the image with specified mean and standard deviation.
    """
    if isinstance(file_fath, str):
        image = image_read(file_fath, color_mode='rgb')
    else:
        image = file_fath
    image = image_resize(image, [224, 224])
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf_image_normalize(image,
                            mean = [176.83188398, 177.25901738, 177.25890556],
                            std = [110.50753887, 110.49521524, 110.50979735])
    return image


def load_image_and_label_for_train(image_path: str, label: int) -> Tuple[tf.Tensor, int]:
    """
    Loads an image and its corresponding label.

    Args:
        image_path (str): The path to the image file.
        label (int): The label associated with the image.

    Returns:
        Tuple[tf.Tensor, int]: A tuple containing the processed image as a
            TensorFlow tensor and the label as an integer tensor.

    This function:
        1. Decodes the image path from bytes to a string.
        2. Processes the image using the image_process_train function.
        3. Converts the label to a TensorFlow tensor.
    """
    image_path = image_path.numpy().decode('utf-8')
    image = image_process_train(image_path)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.constant([np.array(label)], tf.int32)
    return image, label


def load_image_and_label_for_test(image_path: str,
                                  label: Optional[int] = None
                                  ) -> Union[Tuple[tf.Tensor, int], tf.Tensor]:
    """
    Loads an image and its corresponding label if it is not None.

    Args:
        image_path (str): The path to the image file.
        label (int): The label associated with the image.

    Returns:
        Union[Tuple[tf.Tensor, int], tf.Tensor]: A tuple containing
            the processed image as a TensorFlow tensor and the label
            as an integer tensor. Or the processed image if lables is None.

    This function:
        1. Decodes the image path from bytes to a string.
        2. Processes the image using the image_process_test function.
        3. Converts the label to a TensorFlow tensor.
    """
    if isinstance(image_path, str):
        image_path = image_path.numpy().decode('utf-8')
    else:
        image_path = image_path.numpy()
    image = image_process_test(image_path)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    if label is not None:
        label = tf.constant([np.array(label)], tf.int32)
        return image, label
    else:
        return image


def create_train_dataset(X, Y, batch_size: int = 32) -> tf.data.Dataset:
    """
    Creates a TensorFlow train dataset from image paths and labels.

    Args:
        X (list): A list of image paths.
        Y (list): A list of labels corresponding to the images.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.

    Returns:
        tf.data.Dataset: A TensorFlow dataset with batches of images and labels.

    This function:
        1. Creates a TensorFlow dataset from the image paths and labels.
        2. Maps the load_function onto the dataset to process the images and labels.
        3. Ensures the shape of the images and labels.
        4. Batches the dataset.
        5. Prefetches batches for efficient loading.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image_and_label_for_train, [x, y], [tf.float32, tf.int32]), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda image, label: (tf.ensure_shape(image, (224, 224, 3)), tf.ensure_shape(label, (1,))))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def create_test_dataset(X, Y, batch_size: int = 32):
    """
    Creates a TensorFlow test dataset from image paths and labels.

    Args:
        X (list): A list of image paths.
        Y (list): A list of labels corresponding to the images.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.

    Returns:
        tf.data.Dataset: A TensorFlow dataset with batches of images and labels.

    This function:
        1. Creates a TensorFlow dataset from the image paths and labels.
        2. Maps the load_function onto the dataset to process the images and labels.
        3. Ensures the shape of the images and labels.
        4. Batches the dataset.
        5. Prefetches batches for efficient loading.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image_and_label_for_test, [x, y], [tf.float32, tf.int32]), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda image, label: (tf.ensure_shape(image, (224, 224, 3)), tf.ensure_shape(label, (1,))))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def create_inference_dataset(X, batch_size: int = 32):
    """
    Creates a TensorFlow test dataset from image paths.

    Args:
        X (list): A list of image paths.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.

    Returns:
        tf.data.Dataset: A TensorFlow dataset with batches of images.

    This function:
        1. Creates a TensorFlow dataset from the image paths and labels.
        2. Maps the load_function onto the dataset to process the images and labels.
        3. Ensures the shape of the images and labels.
        4. Batches the dataset.
        5. Prefetches batches for efficient loading.
    """
    X = [X] if not isinstance(X, list) else X
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.map(lambda x: tf.py_function(load_image_and_label_for_test, [x], [tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda image: (tf.ensure_shape(image, (224, 224, 3))))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def inference(model,
              images: List[Union[str, np.ndarray]],
              batch_size: int = 1,
              return_predictions: bool = False) -> str:
    cat_dict = {
        0: "email",
        1: "handwritten",
        2: "invoice",
        3: "national_identity_card",
        4: "passeport",
        5: "scientific_publication"
    }
    dataset = create_inference_dataset(images, batch_size)
    predict = model.predict(dataset)
    if return_predictions:
        return predict
    result = [cat_dict[pred] for pred in predict.argmax(axis = 1)]
    return result