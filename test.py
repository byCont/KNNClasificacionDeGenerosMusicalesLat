from python_speech_features import mfcc
import scipy.io.wavfile as waves
import numpy as np
from tempfile import TemporaryFile
import os
import pickle
import operator
import tkinter as tk
from tkinter import PhotoImage, filedialog
import matplotlib.pyplot as plt
import pygame
from tkinter.font import Font


from PIL import ImageTk, Image
import numpy as np
from collections import defaultdict

dataset = []
def loadDataset(filename):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

loadDataset("my.dat")

def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance

def getNeighbors(trainingSet , instance , k):
    distances =[]
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors  

def nearestClass(neighbors):
    classVote ={}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1 
        else:
            classVote[response]=1 
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]


results=defaultdict(int)

i=1
for folder in os.listdir("genres_original/"):
    results[i]=folder
    i+=1

pred=""
#Reproduce la canción al abrir el archivo
def reproducir_cancion():
    pygame.mixer.init()
    pygame.mixer.music.load(archivo)
    pygame.mixer.music.play()

def mostrar_imagen(id):
    imagen = Image.open(f"{id}.jpg")
    imagen = imagen.resize((300, 300))  # Ajusta el tamaño de la imagen según tus necesidades
    imagen_tk = ImageTk.PhotoImage(imagen)
    etiqueta_imagen.configure(image=imagen_tk)
    etiqueta_imagen.image = imagen_tk

def abrir_archivo():
    global pred
    global archivo
    # Abrir ventana de diálogo para seleccionar el archivo de audio
    archivo = filedialog.askopenfilename(filetypes=[("Archivos de audio WAV", "*.wav")])
    
    reproducir_cancion()
    # mostrar_imagen(pred)
    (rate,sig)=waves.read(archivo)

    # Calcular características MFCC
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.transpose(mfcc_feat))
    mean_matrix = np.mean(mfcc_feat, axis=0)
    feature = (mean_matrix, covariance, 0)

    pred = nearestClass(getNeighbors(dataset ,feature , 5))
    mostrar_imagen(pred)
    # Leer el archivo de audio
    muestreo, sonido = waves.read(archivo)
    
    # Calcular la duración del sonido en segundos
    duracion = len(sonido) / muestreo
    
   
    # Crear un arreglo de tiempo para la representación gráfica
    tiempo = np.arange(0, duracion, 1/muestreo)

    #Diccionario de generos
    generos_musicales = {
    1: "Bachata",
    2: "Clasical",
    3: "Disco",
    4: "Hip Hop/Rap",
    5: "Metal",
    6: "Pop",	
    7: "Reggae",
    8: "Reggaeton",
    9: "Rock",
    10: "Salsa",
    }
    genre= generos_musicales[pred]
    
    resultado_label.config(text=f"Archivo seleccionado: {archivo}\nGenero: {genre}\nMuestreo: {muestreo} Hz\nDuración: {duracion} segundos")

    # Mostrar la imagen en una ventana emergente
    plt.show()
    mostrar_imagen(genre)

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("KNN Music Genre Classifier")
ventana.geometry("800x600")  # Establecer las dimensiones de la ventana

# Crear un botón para abrir el archivo de audio
boton_abrir = tk.Button(ventana, text="Abrir archivo de audio", command=abrir_archivo, width=0, bd=1, relief="ridge", font=("Roboto Cn",14))
# boton_abrir=tk.Button(ventana, text="8", width=3, bd=1, relief="ridge", font=("Roboto Cn",14))
boton_abrir.pack()

# Crear una etiqueta para mostrar la imagen
etiqueta_imagen = tk.Label(ventana)
etiqueta_imagen.pack()

# Crear una etiqueta para mostrar el resultado
resultado_label = tk.Label(ventana, text="Resultado:"+ str(results[pred]))
resultado_label.pack()




# Ejecutar la ventana principal
ventana.mainloop()