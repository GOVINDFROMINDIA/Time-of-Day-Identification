import cv2
from keras.models import load_model
import PIL
import tkinter as tk
from tkinter import filedialog
import numpy as np


def daynight(image):
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    if index ==0 and confidence_score>=0.7:
        print("DAYTIME")
    if index ==1 and confidence_score>=0.7:
        print("NIGHT TIME")
    if index ==2 and confidence_score>=0.7:
        print("SUNRISE")

def choose_file():
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("image files", ("*.jpg", "*.png")), ("all files", "*.*")))
    file_path = root.filename
    capture = cv2.VideoCapture(file_path)
    size = (224, 224)
    image = Image.open(file_path).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    daynight(data[0])

def open_camera():
    capture = cv2.VideoCapture(0)
    ret, image = capture.read()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    daynight(image)

np.set_printoptions(suppress=True)
# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

root = tk.Tk()
root.title("Time of Day")
root.geometry("800x533")
bg_image = tk.PhotoImage(file="bg.png")
bg_label = tk.Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

file_button = tk.Button(root, text="Select file", command=choose_file, font=("Helvetica", 20), width=20, height=3)
file_button.place(relx=0.5, rely=0.4, anchor="center")

camera_button = tk.Button(root, text="Open camera", command=open_camera, font=("Helvetica", 20), width=20, height=3)
camera_button.place(relx=0.5, rely=0.6, anchor="center")
root.mainloop()
