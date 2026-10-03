# AI_digit_classifier

# Handwritten Digit Recognition CNN

A Convolutional Neural Network (CNN) project for recognizing handwritten digits from images using TensorFlow and Keras.

## Project Overview

This project started with the MNIST handwritten digit dataset to understand the fundamentals of Convolutional Neural Networks.

The project is now being extended toward recognizing real-world handwritten digits, where images can have different handwriting styles, sizes, positions, lighting conditions, backgrounds, and noise.

```text
Handwritten Image
       ↓
Preprocessing
       ↓
Crop Digit
       ↓
Resize
       ↓
Center
       ↓
Normalize
       ↓
CNN
       ↓
Prediction
       ↓
0 - 9
```
## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Matplotlib

## Dataset

The initial model uses the MNIST handwritten digit dataset.

MNIST contains grayscale images of handwritten digits from "0" to "9".

Each image has the shape:

28 × 28 × 1

where:

- "28" = image height
- "28" = image width
- "1" = grayscale channel

Pixel values are normalized from:

0 - 255

to:

0 - 1

using:

x = x / 255.0

## CNN Architecture

The first version of the model uses the following architecture:

```text
28 × 28 × 1
     ↓
Conv2D
8 filters
3 × 3
     ↓
ReLU
     ↓
MaxPooling
2 × 2
     ↓
14 × 14 × 8
     ↓
Conv2D
16 filters
3 × 3
     ↓
ReLU
     ↓
MaxPooling
2 × 2
     ↓
7 × 7 × 16
     ↓
Flatten
     ↓
784
     ↓
Dense
128 neurons
     ↓
Dense
10 neurons
     ↓
Softmax
     ↓
0 - 9
```
## Model

```code
inputs = tf.keras.Input(shape=(28, 28, 1))

x = tf.keras.layers.Conv2D(
    8,
    (3, 3),
    padding="same"
)(inputs)

x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.MaxPooling2D(
    (2, 2)
)(x)

x = tf.keras.layers.Conv2D(
    16,
    (3, 3),
    padding="same"
)(x)

x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.MaxPooling2D(
    (2, 2)
)(x)

x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

outputs = tf.keras.layers.Dense(
    10,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)
```
## Parameters

The first CNN contains approximately:

103,018 trainable parameters

## Training

The model uses:

- Adam optimizer
- Learning rate: "0.001"
- Sparse categorical cross-entropy
- Sparse categorical accuracy

```code
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy()
    ]
)
```

## Prediction

The trained model produces 10 probabilities:
```
[ P(0), P(1), P(2), ..., P(9) ]
```
The predicted digit is the class with the highest probability.
```code
prediction = model.predict(x)

digit = np.argmax(prediction[0])
confidence = prediction[0][digit]

print("Predicted:", digit)
print("Confidence:", confidence)
```
## Real-World Handwriting

MNIST images are highly standardized.

Real photographs are not.

For example:

## MNIST
```
Black background
      +
White digit
      +
Centered
      +
28 × 28
```
A real photograph can contain:
```
Paper
   +
Shadows
   +
Lighting
   +
Camera noise
   +
Different digit size
   +
Different position
   +
Different handwriting
```

Therefore, an image preprocessing pipeline is required.

Real Image Preprocessing

The current preprocessing pipeline is:
```
Original Image
      ↓
Grayscale
      ↓
Invert
      ↓
Detect Foreground
      ↓
Find Bounding Box
      ↓
Crop Digit
      ↓
Preserve Aspect Ratio
      ↓
Resize
      ↓
Place on 28 × 28 Canvas
      ↓
Center Digit
      ↓
Normalize
      ↓
CNN Input
```
The final CNN input has the shape:

(1, 28, 28, 1)

Why MNIST Alone Is Not Enough

MNIST is excellent for learning CNN fundamentals, but it does not represent every type of real-world handwriting.

Real-world images can differ in:

- Handwriting style
- Stroke thickness
- Digit size
- Digit position
- Rotation
- Slant
- Pen type
- Paper
- Lighting
- Shadows
- Camera quality
- Background
- Noise

The project therefore aims to move beyond MNIST toward more diverse handwriting data.

## Planned CNN V2

The next architecture will be deeper:

```text
28 × 28 × 1
     ↓
Conv2D(32)
     ↓
BatchNormalization
     ↓
ReLU
     ↓
Conv2D(32)
     ↓
BatchNormalization
     ↓
ReLU
     ↓
MaxPooling
     ↓
Dropout
     ↓
Conv2D(64)
     ↓
BatchNormalization
     ↓
ReLU
     ↓
Conv2D(64)
     ↓
BatchNormalization
     ↓
ReLU
     ↓
MaxPooling
     ↓
Dropout
     ↓
Flatten
     ↓
Dense(128)
     ↓
Dropout
     ↓
Dense(10)
     ↓
Softmax
```

## Data Augmentation

To improve robustness, the project will experiment with transformations such as:

- Rotation
- Translation
- Scaling
- Zoom
- Small distortions
- Noise
- Stroke variation

The goal is to expose the CNN to handwriting that is different from the original training examples.

## Project Roadmap

V1 — Basic CNN

- [x] Load MNIST
- [x] Normalize images
- [x] Build CNN
- [x] Train CNN
- [x] Evaluate CNN
- [x] Predict digits

V2 — Deeper CNN

- [ ] Increase convolution filters
- [ ] Add Batch Normalization
- [ ] Add Dropout
- [ ] Improve feature extraction
- [ ] Compare architectures

V3 — Diverse Handwriting

- [ ] Add additional handwriting datasets
- [ ] Create custom handwriting dataset
- [ ] Add data augmentation
- [ ] Test on unseen handwriting

V4 — Real-World Recognition

- [x] Real image preprocessing
- [x] Digit cropping
- [x] Resize and centering
- [ ] Improve camera-image preprocessing
- [ ] Improve real-world accuracy

V5 — Complete System

- [ ] Multi-digit recognition
- [ ] Automatic digit detection
- [ ] Real-time camera recognition
- [ ] Web/API deployment
- [ ] Mobile deployment

## What Are You Learning

This project is also being used to understand the mathematics and internal operation of CNNs.

## Topics include:

- Image tensors
- Channels
- Convolution
- Filters
- Feature maps
- Padding
- Stride
- ReLU
- Pooling
- Receptive fields
- Flattening
- Dense layers
- Softmax
- Cross-entropy
- Backpropagation
- Gradients
- Adam optimizer
- Batch Normalization
- Dropout
- Data augmentation
- Generalization
- Distribution shift

## Core Idea

A major lesson from this project is:

Same input shape
        ≠
Same data distribution

Two images can both be:

28 × 28 × 1

but still look very different to a neural network because their pixel distributions and visual characteristics are different.

Project Structure
```
handwritten-digit-cnn/
│
├── README.md
│
├── train.py
├── predict.py
│
├── models/
│   └── model.keras
│
├── data/
│
└── requirements.txt
```

## Installation

### Install the required packages:

```code
pip install tensorflow numpy pillow matplotlib
```

Running the Project

## Train the model:

```code
python train.py
 ```

## Run prediction:

```code
python predict.py
```

## Future Goal

The long-term goal is to transform this basic MNIST CNN into a robust handwritten digit recognition system capable of handling real-world images rather than only standardized datasets.

---

# Author

Devabratta Yumnam

GitHub: "deva7102bratta"
