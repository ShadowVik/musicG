import os
import numpy
from tensorflow.keras.models import load_model
import librosa
import tkinter as tk
from tkinter import filedialog, PhotoImage

model = load_model("music_classifier.h5")

genres = ['hiphop', 'classical', 'rock']

def get_mfcc(y, sr):
    return numpy.array(librosa.feature.mfcc(y=y, sr=sr))

def get_melspectrogram(y, sr):
    return numpy.array(librosa.feature.melspectrogram(y=y, sr=sr))

def get_chroma_vector(y, sr):
    return numpy.array(librosa.feature.chroma_stft(y=y, sr=sr))

def get_tonnetz(y, sr):
    return numpy.array(librosa.feature.tonnetz(y=y, sr=sr))

def get_feature(file_path):
    y, sr = librosa.load(file_path, offset=0, duration=30)
    
    mfcc = get_mfcc(y, sr)
    melspectrogram = get_melspectrogram(y, sr)
    chroma = get_chroma_vector(y, sr)
    tntz = get_tonnetz(y, sr)
    
    features = numpy.concatenate([
        mfcc.mean(axis=1), mfcc.min(axis=1), mfcc.max(axis=1),
        melspectrogram.mean(axis=1), melspectrogram.min(axis=1), melspectrogram.max(axis=1),
        chroma.mean(axis=1), chroma.min(axis=1), chroma.max(axis=1),
        tntz.mean(axis=1), tntz.min(axis=1), tntz.max(axis=1),
    ])
    
    return features

def open_file_dialog():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav")])
    
    if file_path:
        print("Fisier ales: " + file_path[17:])
        
        chosen = tk.Label(text="Fisier ales: " + file_path[17:], font=("Arial", 14), fg="blue", image=photo, compound="right", cursor="hand2")
        chosen.pack()
        chosen.bind("<Button-1>", lambda e: playSong(file_path))

        predictSong(file_path)

def playSong(file_path):
    os.startfile(file_path)

def predictSong(file_path):
    global loading_text
    loading_text = tk.Label(text="Predicting...", font=("Arial", 12))
    loading_text.pack()
    feature = get_feature(file_path)
    y = model.predict(feature.reshape(1, 498))
    ind = numpy.argmax(y)
    print("Genre: " + genres[ind])
    loading_text.destroy()
    
    rez = tk.Label(text="Genul muzicii: " + genres[ind], fg="red", font=("Arial", 20))
    rez.pack()

window = tk.Tk()
window.geometry("500x300")
window.title("Music Genre Classifier")
photo = PhotoImage(file="play.png")
greeting = tk.Label(text="Alege melodia dorită pentru clasificare:", font=("Arial", 16))
greeting.pack()
button = tk.Button(
    text="Browse",
    width=10,
    height=2,
    bg="gold",
    fg="black",
    command=open_file_dialog
)
button.pack()

window.mainloop()
