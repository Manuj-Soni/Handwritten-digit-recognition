A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network trained on the MNIST dataset. The model predicts digits drawn by the user through a simple interface.


FEATURES:-
1. CNN-based digit classification
2. Real-time prediction from the user input
3. Tkinter-based drawing interface
4. Displays predicted digit with confidence


TECH STACK:-
1. Python
2. Tensorflow / Keras
3. Numpy
4. PIL
5. Tkinter
6. matplotlib


PROJECT STRUCTURE:-
Handwritten-Digit-Recognition/
│
├── model.py              # CNN model training
├── interface.py         # Tkinter drawing app
├── model.h5 / .pt       # Saved trained model
├── requirements.txt
└── README.md


HOW IT WORKS:-
1. User draws a digit in the GUI
2. Image is captured and preprocessed
3. CNN model predicts digit class (0-9)
4. Output is displayed with confidence score


HOW TO RUN:-
1. Clone the repository
2. Install dependencies:
     pip install -r requirements.text
3. Run interface:
     interface.py


MODEL PERFORMANCE:-
1. Dataset : MNIST
2. Accuracy : 99.349999%


AUTHOR:
Manuj Soni


NOTE:-
This project was built to understand CNNs, image preprocessing, and real-time prediction using deep learning.
