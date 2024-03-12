import xml.etree.ElementTree as ET
import pandas as pd


def img_xml_unicode(list_filename, filepath):

    unicode = ''

    # Parsing fichier XML
    for xmlname in list_filename:
        if xmlname != 'desktop.ini':
            tree = ET.parse(filepath+"/"+xmlname)
            root = tree.getroot()
            url = root.tag.split("}", 1)[0][1:]

            # Accès aux attributs de <Unicode>
            unicode_elements = root.findall(f'.//{{{url}}}Unicode')

            # Récupérer les texte des images
            for element in unicode_elements:
                unicode = unicode + ' ' + element.text

    return unicode


def img_xml_language(list_filename, filepath):
    language = []

    for xmlname in list_filename:
        if xmlname != 'desktop.ini':
            tree = ET.parse(filepath+"/"+xmlname)
            root = tree.getroot()
            url = root.tag.split("}", 1)[0][1:]

            # Accès aux attributs de <Page>
            page_element = root.find(f'.//{{{url}}}Page')

            # Accès à l'attributs "language de <Property>
            property_element = page_element.find(
                f'.//{{{url}}}Property[@key="language"]')

            # Récupérer les langues
            if property_element is not None:
                language_value = property_element.get('value')
                language.append(language_value)

    print(pd.Series(language).unique())


def img_xml_dimension(list_filename, file_path):

    image_filename = []
    image_height = []
    image_width = []

    for xmlname in list_filename:
        if xmlname != 'desktop.ini':
            tree = ET.parse(filepath+"/"+xmlname)
            root = tree.getroot()
            url = root.tag.split("}", 1)[0][1:]

            # Récupérer les dimensions des images
            image_filename.append(page_element.get('imageFilename'))
            image_height.append(page_element.get('imageHeight'))
            image_width.append(page_element.get('imageWidth'))

    return image_filename, image_height, image_width
