import streamlit as st
import os
import operator
import pickle
import pygame
import scipy.io.wavfile as waves
from collections import defaultdict
from python_speech_features import mfcc
from PIL import ImageTk, Image
import numpy as np
import tempfile

dataset = []

def loadDataset(filename):
    with open("my.dat", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

loadDataset("my.dat")

def distance(instance1, instance2, k):
    distance = 0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1)) 
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1 
        else:
            classVote[response] = 1 
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]

results = defaultdict(int)
i = 1
for folder in os.listdir("genres_original/"):
    results[i] = folder
    i += 1

def reproducir_cancion(archivo_path):
    pygame.mixer.init()
    pygame.mixer.music.load(archivo_path)
    pygame.mixer.music.play()

def mostrar_imagen(id):
    imagen = Image.open(f"covers/{id}.png")
    # imagen = imagen.resize((300, 300))
    st.image(imagen)

def abrir_archivo():
    global pred
    global archivo
    archivo = st.file_uploader("Seleccionar archivo de audio WAV", type=["wav"])
    
    
    if archivo is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(archivo.read())
            archivo_path = temp.name

        reproducir_cancion(archivo_path)
        (rate, sig) = waves.read(archivo_path)
        duration = len(sig) / rate
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.transpose(mfcc_feat))
        mean_matrix = np.mean(mfcc_feat, axis=0)
        feature = (mean_matrix, covariance, 0)
        pred = nearestClass(getNeighbors(dataset, feature, 5))
        st.subheader(f"**Género Clasificado:** {results[pred]}")
        st.write("Portada del Género")

        mostrar_imagen(pred)
        # generos_musicales = {
        #     1: "Bachata",
        #     2: "Clasical",
        #     3: "Disco",
        #     4: "Hip Hop/Rap",
        #     5: "Metal",
        #     6: "Pop",    
        #     7: "Reggae",
        #     8: "Reggaeton",
        #     9: "Rock",
        #     10: "Salsa",
        # }

        # genre = generos_musicales[pred]
        st.subheader("Información de la canción")
        st.write(f"**Archivo seleccionado:** {archivo.name}")
        st.write(f"**Género:** {results[pred]}")
        st.write(f"**Tasa de muestreo:** {rate} Hz")
        st.write(f"**Duración:** {duration:.2f} segundos")

# Configuración de página
st.set_page_config(page_title="Music Genre Classifier", page_icon=":musical_note:")

# Título de la página
st.title("KNN Music Genre Classifier")

# Introducción
st.write("Bienvenido al Clasificador de Géneros Musicales")
st.write("Este algoritmo utiliza el algoritmo de clasificación K-Nearest Neighbors (KNN) para clasificar archivos de audio en diferentes géneros musicales.")

# Descripción de géneros musicales
st.write("El algoritmo puede clasificar archivos de audio en los siguientes géneros musicales:")
st.write("**Bachata**", "~**Clasical**", "~**Hip Hop/Rap**", "~**Metal**", "~**Pop**", "~**Reggae**", "~**Reggaeton**", "~**Rock**", "~**Salsa**.") 


# Contenido adicional de la aplicación
st.write("Puedes cargar un archivo de audio y el algoritmo clasificará automáticamente el género musical al que pertenece.")

# Resto del código...


# Sección principal
st.sidebar.title("Clasificador de Género Musical")

# Variable de estado para controlar el despliegue de las instrucciones
instrucciones_desplegadas = False

# Botón para desplegar/cerrar las instrucciones
if st.sidebar.button('Instrucciones'):
    instrucciones_desplegadas = not instrucciones_desplegadas

# Despliegue de las instrucciones según el estado
if instrucciones_desplegadas:
    st.sidebar.write('Selecciona el archivo: Haz clic en "Abrir archivo de audio", Se abrirá un cuadro de diálogo que te permitirá seleccionar un archivo de audio en formato WAV.')
    st.sidebar.write('Cargar y reproducir el archivo: Después de seleccionar el archivo de audio, automaticamente se reproducirá y la clasificará. Puedes escuchar la canción para tener una idea de su estilo musical.')
    st.sidebar.write('Clasificar el archivo de audio: La aplicación utilizará el algoritmo KNN para clasificar el archivo de audio en un género musical específico. Una vez que se haya cargado y reproducido el archivo, la aplicación calculará las características del audio y lo clasificará.')
    st.sidebar.write('Resultados: Después de clasificar el archivo de audio, la aplicación mostrará el género musical predicho y una imagen asociada al género. Además, se mostrará información adicional como la tasa de muestreo y la duración de la canción.')
    st.sidebar.write('Clasificar otro archivo: Si deseas clasificar otro archivo de audio, simplemente repite los pasos 3 al 6. Puedes cargar y clasificar varios archivos de audio en la misma sesión de la aplicación.')

else:
    st.sidebar.write('Presiona el botón "Instrucciones" para ver el paso a paso')



# Créditos
st.sidebar.markdown("---")
st.sidebar.markdown("Adaptado por: Daniel Cont")
st.sidebar.markdown("[GitHub Repository](https://github.com/byCont/KNNClasificacionDeGenerosMusicalesLat.git)")

# Footer
st.sidebar.markdown("---")

st.sidebar.markdown("[GitHub Repository Original](https://github.com/Meghashyam-Malur/KNN-Music-Genre-Classification)")
st.sidebar.markdown("ChatGPT")
st.sidebar.markdown("[Bibliografía](https://www.redalyc.org/pdf/5055/505554816003.pdf)")


# Mostrar imagen y resultados en la página principal
etiqueta_imagen = st.empty()

abrir_archivo()

#commads:  streamlit run app.py