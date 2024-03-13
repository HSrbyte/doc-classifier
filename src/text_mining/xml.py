import xml.etree.ElementTree as ET
import pandas as pd


def img_xml_unicode(list_filename, filepath):
    """
    Extract Unicode text from XML files.

    This function parses XML files located in the specified filepath and extracts Unicode text
    from the <Unicode> attribute in the XML elements.

    Parameters:
        list_filename (list): A list of XML file names to parse.
        filepath (str): The path to the directory containing the XML files.

    Returns:
        str: A concatenated string of Unicode text extracted from the XML files.
    """
    unicode_text = ''

    # Parsing XML files
    for xmlname in list_filename:
        if xmlname != 'desktop.ini':
            tree = ET.parse(filepath + "/" + xmlname)
            root = tree.getroot()
            url = root.tag.split("}", 1)[0][1:]

            # Accessing the <Unicode> attribute
            unicode_elements = root.findall(f'.//{{{url}}}Unicode')

            # Extracting text from the <Unicode> attribute
            for element in unicode_elements:
                unicode_text += ' ' + element.text

    return unicode_text


def img_xml_language(list_filename, filepath):
    """
    Extract language information from XML files.

    This function parses XML files located in the specified filepath and extracts language
    information from the 'language' attribute of the <Page> element.

    Parameters:
        list_filename (list): A list of XML file names to parse.
        filepath (str): The path to the directory containing the XML files.

    Returns:
        None
    """
    language = []

    # Parsing XML files
    for xmlname in list_filename:
        if xmlname != 'desktop.ini':
            tree = ET.parse(filepath + "/" + xmlname)
            root = tree.getroot()
            url = root.tag.split("}", 1)[0][1:]

            # Accessing the <Page> attributes
            page_element = root.find(f'.//{{{url}}}Page')

            # Accessing the 'language' attribute of <Page>
            property_element = page_element.find(
                f'.//{{{url}}}Property[@key="language"]')

            # Extracting the language
            if property_element is not None:
                language_value = property_element.get('value')
                language.append(language_value)

    # Printing unique language values
    print(pd.Series(language).unique())


def img_xml_dimension(list_filename, file_path):
    """
    Extract image dimensions from XML files.

    This function parses XML files located in the specified file path and extracts image
    dimensions including image filename, height, and width.

    Parameters:
        list_filename (list): A list of XML file names to parse.
        file_path (str): The path to the directory containing the XML files.

    Returns:
        tuple: A tuple containing lists of image filenames, heights, and widths.
    """
    image_filename = []
    image_height = []
    image_width = []

    # Parsing XML files
    for xmlname in list_filename:
        if xmlname != 'desktop.ini':
            tree = ET.parse(file_path + "/" + xmlname)
            root = tree.getroot()
            url = root.tag.split("}", 1)[0][1:]

            # Retrieve image dimensions
            page_element = root.find(f'.//{{{url}}}Page')
            image_filename.append(page_element.get('imageFilename'))
            image_height.append(page_element.get('imageHeight'))
            image_width.append(page_element.get('imageWidth'))

    return image_filename, image_height, image_width
