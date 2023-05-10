import os
import numpy
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, PhotoImage



model = load_model("music_classifier.h5")

def open_file_dialog():
 global file_path 
 file_path = filedialog.askopenfilename(filetypes= [("Audio files", "*.wav")])
 if file_path:
        print("Fisier ales: " + file_path[17:])
        
        chosen = tk.Label(text="Fisier ales: " + file_path[17:], font=("Arial", 14), fg="blue", image=photo, compound="right", cursor="hand2")
        chosen.pack()
        chosen.bind("<Button-1>", lambda e:playSong(file_path))     

        global buttonPr 
        buttonPr= tk.Button(
        text="Predict",
        width=7,
        height=1,
        bg="green",
        fg="yellow",
        command=lambda: [predictSong(file_path)]
        )
        buttonPr.pack()

        global loading_text 
        loading_text = tk.Label(text="Predicting...",font=("Arial", 12))
        loading_text.pack()

def playSong(file_path):
        os.startfile(file_path)

def predictSong(file_path):
    loading_text.destroy()
    buttonPr.destroy()
    from musicG import genres
    from musicG import get_feature
    file_path = file_path[17:]
    feature = get_feature(file_path)
    y = model.predict(feature.reshape(1,498))
    ind = numpy.argmax(y)
    print("Genre: " + genres[ind])
    
    rez = tk.Label(text="Genul muzicii: " + genres[ind], fg="red", font=("Arial", 20))
    rez.pack()
  
window = tk.Tk()
window.geometry("500x300")
window.title("Music Genre Classifier")
photo = PhotoImage(file="play.png")
greeting = tk.Label(text="Alege melodia dorită pentru clasificare:",font=("Arial", 16))
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