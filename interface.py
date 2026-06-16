import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow
import keras
import matplotlib.pyplot as plt

model = keras.models.load_model("cnn.keras")

canvas_size = 200

img = Image.new("L", (canvas_size, canvas_size), 255)  # white bg
draw_img = ImageDraw.Draw(img)

#tkinter setup
root = tk.Tk()
root.title("Digit Drawer")

canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
canvas.pack()

def draw(event):
    x, y = event.x, event.y
    r = 4

    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")

    draw_img.ellipse([x-r, y-r, x+r, y+r], fill=0)
canvas.bind("<B1-Motion>", draw)

def preprocess(image):
    img = image.copy()
    img = ImageOps.invert(img)
    arr = np.array(img)

    #cropping
    coords = np.column_stack(np.where(arr > 20))
    if len(coords) == 0:
        return None, None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    img = img.crop((x_min, y_min, x_max, y_max))

    # resize 
    img = ImageOps.contain(img, (20, 20))
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img, ((28-img.size[0])//2, (28-img.size[1])//2))
    arr = np.array(new_img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28)

    return arr, new_img

def predict():
    global img
    x, processed = preprocess(img)

    if x is None:
        print("Nothing drawn")
        return

    prediction = model.predict(x)
    digit = np.argmax(prediction)

    print("Predicted digit:", digit)
    print("confidence:", prediction.max()*100, "%")

    plt.imshow(processed, cmap="gray")
    plt.title(f"(pred={digit}, confidence={prediction.max()*100}%)")
    plt.show()

btn = tk.Button(root, text="Predict", command=predict)
btn.pack()

root.mainloop()

print("I am learning git")