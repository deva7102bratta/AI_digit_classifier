# AI_digit_classifier

Handwritten Digit Recognition CNN

A Convolutional Neural Network (CNN) project for recognizing handwritten digits 0–9 from real-world handwritten images.

The project started as an MNIST digit classifier and is being developed toward a more robust real-world handwritten digit recognition system.

---

Project Goal

The initial goal was to train a CNN on the MNIST dataset.

However, a model that performs well on MNIST does not necessarily perform well on photographs of handwritten digits.

Therefore, this project is being upgraded to handle:

- Different handwriting styles
- Different stroke thicknesses
- Different digit sizes
- Different positions
- Different writing instruments
- Different backgrounds
- Real photographs
- Lighting variations
- Noise and image imperfections

The long-term goal is:

Real handwritten image
        ↓
Digit detection
        ↓
Background removal
        ↓
Cropping
        ↓
Resizing
        ↓
Centering
        ↓
Normalization
        ↓
CNN
        ↓
0–9 prediction

---

Current Version

CNN V1

The first CNN architecture was designed primarily for learning the fundamentals of convolutional neural networks.

28×28×1
   ↓
Conv2D(8, 3×3)
   ↓
ReLU
   ↓
MaxPooling(2×2)
   ↓
14×14×8
   ↓
Conv2D(16, 3×3)
   ↓
ReLU
   ↓
MaxPooling(2×2)
   ↓
7×7×16
   ↓
Flatten
   ↓
Dense(128)
   ↓
Dense(10)
   ↓
Softmax

Total trainable parameters:

103,018

V1 performs well on MNIST but is not sufficiently robust for arbitrary real-world handwritten photographs.

---

Why MNIST Is Not Enough

MNIST provides standardized 28×28 grayscale handwritten digits.

A real photograph is much more complicated:

MNIST

28×28
black background
white digit
centered digit
controlled image format

Compared with:

Real photograph

camera image
        ↓
paper
        ↓
lighting
        ↓
shadows
        ↓
pen/marker strokes
        ↓
different digit size
        ↓
different position
        ↓
background
        ↓
noise

Therefore:

«Having the same "28×28×1" shape does not mean that two images belong to the same data distribution.»

This project aims to reduce that gap.

---

Image Preprocessing

A real handwritten photograph is converted into an MNIST-like representation before entering the CNN.

Original image
      ↓
Grayscale
      ↓
Color inversion
      ↓
Digit detection
      ↓
Bounding-box extraction
      ↓
Crop
      ↓
Resize
      ↓
Place on 28×28 canvas
      ↓
Center
      ↓
Normalize
      ↓
(1, 28, 28, 1)

Example preprocessing code:

from PIL import Image
import numpy as np

img = Image.open("digit.jpg")

# Grayscale
img = img.convert("L")

# NumPy array
arr = np.array(img, dtype="float32")

# Invert colors
arr = 255 - arr

# Detect foreground
mask = arr > 40

ys, xs = np.where(mask)

# Bounding box
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()

# Crop
cropped = arr[
    y_min:y_max + 1,
    x_min:x_max + 1
]

# Resize
h, w = cropped.shape

scale = 20 / max(h, w)

new_w = max(1, int(w * scale))
new_h = max(1, int(h * scale))

cropped_img = Image.fromarray(
    cropped.astype("uint8")
)

cropped_img = cropped_img.resize(
    (new_w, new_h),
    Image.Resampling.LANCZOS
)

cropped = np.array(
    cropped_img,
    dtype="float32"
)

# 28×28 canvas
canvas = np.zeros(
    (28, 28),
    dtype="float32"
)

# Center
y_offset = (28 - new_h) // 2
x_offset = (28 - new_w) // 2

canvas[
    y_offset:y_offset + new_h,
    x_offset:x_offset + new_w
] = cropped

# Normalize
canvas = canvas / 255.0

# CNN input
x = canvas.reshape(
    1, 28, 28, 1
)

---

CNN Input

The CNN expects:

(batch, height, width, channels)

Therefore:

(1, 28, 28, 1)

means:

1  → one image
28 → height
28 → width
1  → grayscale channel

The total number of pixel values is:

[
1\times28\times28\times1=784
]

---

Prediction

The final layer contains 10 outputs:

0
1
2
3
4
5
6
7
8
9

The model produces a probability distribution:

[0.01, 0.02, 0.03, 0.01, 0.02,
 0.01, 0.85, 0.01, 0.02, 0.02]

The predicted class is the index with the largest probability.

prediction = model.predict(x, verbose=0)

digit = np.argmax(prediction[0])

confidence = prediction[0][digit]

print("Predicted:", digit)
print("Confidence:", confidence)

---

V2 — Improved CNN

The next architecture is intended to provide a stronger feature representation.

28×28×1
    ↓
Conv2D(32, 3×3)
    ↓
BatchNormalization
    ↓
ReLU
    ↓
Conv2D(32, 3×3)
    ↓
BatchNormalization
    ↓
ReLU
    ↓
MaxPooling(2×2)
    ↓
Dropout
    ↓
Conv2D(64, 3×3)
    ↓
BatchNormalization
    ↓
ReLU
    ↓
Conv2D(64, 3×3)
    ↓
BatchNormalization
    ↓
ReLU
    ↓
MaxPooling(2×2)
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

The purpose of V2 is to investigate whether a deeper feature hierarchy improves generalization beyond the simple V1 network.

---

Dataset Strategy

MNIST will remain useful as a foundation, but it will not be the only source of training data.

The planned dataset strategy is:

                 MNIST
                   │
                   ↓
          Basic digit patterns
                   │
                   ↓
        Additional handwriting
                   │
                   ↓
        Our own handwritten data
                   │
                   ↓
           Data augmentation
                   │
                   ↓
            CNN training

Our own dataset can contain examples such as:

0 → handwritten samples
1 → handwritten samples
2 → handwritten samples
...
9 → handwritten samples

This allows the model to learn handwriting characteristics that are not sufficiently represented by the original MNIST distribution.

---

Data Augmentation

To improve robustness, training images can be modified while preserving their digit identity.

Possible transformations include:

- Small rotations
- Translation
- Scaling
- Zoom
- Stroke variation
- Brightness variation
- Small amounts of noise

Conceptually:

Original 6
   │
   ├── rotate
   ├── shift
   ├── scale
   ├── zoom
   └── add noise
        ↓
Multiple training examples

This increases variation without requiring thousands of completely new handwritten samples.

---

Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Matplotlib

---

Learning Objectives

This project is also being used to understand CNNs from first principles.

Topics covered include:

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
- Gradient descent
- Adam optimizer
- Batch normalization
- Dropout
- Data augmentation
- Generalization
- Distribution shift

---

Project Development

V1
│
├── MNIST
├── Basic CNN
├── 8 filters
├── 16 filters
└── Basic prediction
        ↓
V2
│
├── Deeper CNN
├── More feature maps
├── Batch normalization
└── Dropout
        ↓
V3
│
├── Real handwriting dataset
├── Data augmentation
└── Mixed training data
        ↓
V4
│
├── Real photograph input
├── Automatic digit detection
├── Robust preprocessing
└── Real-world testing

---

Current Status

Completed

- [x] Load MNIST
- [x] Understand image tensors
- [x] Build CNN V1
- [x] Train CNN on MNIST
- [x] Predict MNIST digits
- [x] Process real handwritten images
- [x] Convert photographs to "28×28×1"
- [x] Detect and crop handwritten digits
- [x] Normalize input
- [x] Test the model on personal handwriting

In Progress

- [ ] CNN V2
- [ ] Improve real-image preprocessing
- [ ] Build a real handwriting dataset
- [ ] Data augmentation
- [ ] Train using multiple handwriting sources
- [ ] Evaluate on unseen handwriting
- [ ] Improve real-world robustness

Future

- [ ] Automatic digit detection
- [ ] Multi-digit recognition
- [ ] Real-time camera recognition
- [ ] Web/API deployment
- [ ] Mobile deployment
- [ ] Continuous dataset expansion

---

Key Principle

The objective is not simply to maximize MNIST accuracy.

The objective is to build a model that can generalize:

[
\boxed{
\text{Training data}
\rightarrow
\text{learned representation}
\rightarrow
\text{unseen real-world handwriting}
}
]

A model should ultimately recognize digits it has not seen before, written in different styles and captured under different conditions.

---

Author

Devabratta Yumnam

Machine Learning / AI project focused on understanding neural networks and building practical AI systems.
