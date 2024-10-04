import os
import numpy as np
import xml.etree.ElementTree as etree

import skimage
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay



# https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection/code


#Parsetja el fitxer xml i recupera la informació necessaria per trobar la cara de l'animal
#
def extract_xml_annotation(filename):
    """Parse the xml file
    :param filename: str
    :return annotation: diccionari
    """
    z = etree.parse(filename)
    objects = z.findall('./object')
    size = (int(float(z.find('.//width').text)), int(float(z.find('.//height').text)))
    dds = []
    for obj in objects:
        dds.append(obj.find('name').text)
        dds.append([int(float(obj.find('bndbox/xmin').text)),
                                      int(float(obj.find('bndbox/ymin').text)),
                                      int(float(obj.find('bndbox/xmax').text)),
                                      int(float(obj.find('bndbox/ymax').text))])

    return {'size': size, 'informacio': dds}

# Selecciona la cara de l'animal i la transforma a la mida indicat al paràmetre mida_desti
def retall_normalitzat(imatge, dades, mida_desti=(64,64)):
    """
    Extreu la regió de la cara (ROI) i retorna una nova imatge de la mida_destí
    :param imatge: imatge que conté un animal
    :param dades: diccionari extret del xml
    :mida_desti: tupla que conté la mida que obtindrà la cara de l'animal
    """
    x, y, ample, alt = dades['informacio'][1]
    retall = np.copy(imatge[y:alt, x:ample])
    return resize(retall, mida_desti)


def obtenir_dades(carpeta_imatges, carpeta_anotacions, mida=(64, 64)):
    """Genera la col·lecció de cares d'animals i les corresponents etiquetes
    :param carpeta_imatges: string amb el path a la carpeta d'imatges
    :param carpeta_anotacions: string amb el path a la carpeta d'anotacions
    :param mida: tupla que conté la mida que obtindrà la cara de l'animal
    :return:
        images: numpy array 3D amb la col·lecció de cares
        etiquetes: llista binaria 0 si l'animal és un moix 1 en cas contrari
    """

    n_elements = len([entry for entry in os.listdir(carpeta_imatges) if os.path.isfile(os.path.join(carpeta_imatges, entry))])
    # Una matriu 3D: mida x mida x nombre d'imatges
    imatges = np.zeros((mida[0], mida[1], n_elements), dtype=np.float16)
    # Una llista d'etiquetes
    etiquetes = [0] * n_elements

    #  Recorre els elements de les dues carpetes: llegeix una imatge i obté la informació interessant del xml
    with os.scandir(carpeta_imatges) as elements:

        for idx, element in enumerate(elements):
            nom = element.name.split(".")
            nom_fitxer = nom[0] + ".xml"
            imatge = imread(carpeta_imatges + os.sep + element.name, as_gray=True)
            anotacions = extract_xml_annotation(carpeta_anotacions + os.sep + nom_fitxer)

            cara_animal = retall_normalitzat(imatge, anotacions, mida)
            tipus_animal = anotacions["informacio"][0]

            imatges[:, :, idx] = cara_animal
            etiquetes[idx] = 0 if tipus_animal == "cat" else 1

    return imatges, etiquetes


def obtenirHoG(imatges):
    caracteristiques = []
    for i in range(imatges.shape[2]):
        imatge = imatges[:, :, i]  # Obtenim la imatge en escala de grisos

        # Calcular HOG
        fd = hog(
            imatge,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True
        )
        caracteristiques.append(fd)

    return np.array(caracteristiques)

def main():
    carpeta_images = "gatigos/images"  # NO ES POT MODIFICAR
    carpeta_anotacions = "gatigos/annotations"  # NO ES POT MODIFICAR
    mida = (64, 64)  # DEFINEIX LA MIDA, ES RECOMANA COMENÇAR AMB 64x64
    imatges, etiquetes = obtenir_dades(carpeta_images, carpeta_anotacions, mida)

    caracteristiques = obtenirHoG(imatges)
    X_train, X_test, y_train, y_test = train_test_split(caracteristiques, etiquetes, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    gamma = 1.0 / (X_train_transformed.shape[1] * X_train_transformed.var())


    def kernel_lineal(x1, x2):
        return x1.dot(x2.T)

    def kernel_gauss(x1, x2):
        return np.exp(-gamma * distance_matrix(x1, x2) ** 2)

    def kernel_poly(x1, x2, degrees=3):
        return (gamma * kernel_lineal(x1, x2)) ** degrees

    kernels = {'linear': kernel_lineal, 'rbf': kernel_gauss, 'poly': kernel_poly}

    for kernel in kernels:
        print(f"\nProbando kernel: {kernel}")

        svm = SVC(C=1.0, kernel=kernels[kernel])
        svm.fit(X_train_transformed, y_train)

        y_predict = svm.predict(X_test_transformed)

        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict)
        cm = confusion_matrix(y_test, y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

        print(f"Accuracy: {accuracy:.4f}\n")
        print(f"Precision: {precision:.4f}\n")
        print(f"Recall: {recall:.4f}\n")
        print(f"F1-Score: {f1:.4f}\n")

'''
CODI PER PROBAR DIFERENTS CONFIGURACIONS DE HOG

    fd, hog_image = hog(
        imatges[:, :, 0],
        orientations=8,
        pixels_per_cell=(3, 3),
        cells_per_block=(4, 4),
        visualize=True

    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(imatges[:, :, 0], cmap=plt.cm.gray)
    ax1.set_title('Input image')


    ax2.axis('off')
    ax2.imshow(hog_image, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    print(f"Caracteristiques HoG de la primera imatge: {caracteristiques[0]}")
'''


if __name__ == "__main__":

    main()









