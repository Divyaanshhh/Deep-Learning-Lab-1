# Deep Learning Lab 1 - Perceptron using NumPy

A simple implementation of the **Perceptron Algorithm** in Python using **NumPy**. This program demonstrates binary classification by simulating the **AND Logic Gate**.

---

## 📌 Objective
To implement a simple Perceptron model using NumPy and train it to classify the outputs of an AND gate, and to understand the foundational concepts of neural computation that lead into deep learning.

---

## 🧠 What is a Perceptron?

The Perceptron, introduced by **Frank Rosenblatt in 1958**, is the simplest form of an artificial neural network — a single-layer binary linear classifier. It takes multiple inputs, multiplies each by a weight, sums them along with a bias term, and passes the result through a **step activation function** to produce a binary output (0 or 1).

It is considered the foundational building block of modern deep neural networks, including Multi-Layer Perceptrons (MLPs), CNNs, and beyond.

### Mathematical Model
```
z = (w1*x1 + w2*x2 + ... + wn*xn) + b
y = 1   if z >= 0
y = 0   if z <  0
```

### Weight Update Rule (Perceptron Learning Rule)
```
w_new = w_old + learning_rate * (target - predicted) * input
b_new = b_old + learning_rate * (target - predicted)
```

The weights are adjusted only when a misclassification occurs, gradually pushing the decision boundary toward correctly separating the classes.

---

## 🛠️ Technologies Used
- Python 3.x
- NumPy
- Visual Studio Code

---

## 📁 Project Structure
```
Deep-Learning-Lab/
│── Lab-1.py
│── README.md
└── .gitignore
```

---

## 🚀 Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Deep-Learning-Lab.git
```

### 2. Open the Project
```bash
cd Deep-Learning-Lab
```

### 3. Create a Virtual Environment
**Windows**
```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment
**PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```
**Command Prompt**
```cmd
.venv\Scripts\activate
```

### 5. Install Dependencies
```bash
pip install numpy
```

### 6. Run the Program
```bash
python Lab-1.py
```

---

## ⚙️ Program Description

The program performs the following steps:
1. Imports the NumPy library.
2. Creates the input dataset for the AND gate.
3. Initializes weights and bias to zero.
4. Defines a step activation function for binary output.
5. Trains the perceptron over multiple epochs using the Perceptron Learning Rule.
6. Updates weights and bias whenever a prediction error occurs.
7. Tracks and prints the number of misclassifications per epoch.
8. Stops early if the model converges (zero errors in an epoch).
9. Displays the final learned weights, bias, and predictions for all inputs.

---

## 📊 Input Dataset (AND Gate)

| Input 1 | Input 2 | Expected Output |
|:-------:|:-------:|:----------------:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

---

## ✅ Expected Output
```
Epoch 1/20 - Errors: 2
Epoch 2/20 - Errors: 1
Epoch 3/20 - Errors: 0
Converged early!

Training complete!

Final Weights: [0.2 0.1]
Final Bias: -0.2

Predictions:
Input: [0 0] -> Output: 0
Input: [0 1] -> Output: 0
Input: [1 0] -> Output: 0
Input: [1 1] -> Output: 1
```
> *The learned weights may vary slightly depending on initialization, learning rate, and epoch count, but the final predictions should remain the same for a linearly separable problem like AND.*

---

## 🌍 Applications of the Perceptron

| Domain | Application |
|---|---|
| **Logic Gates** | Modeling linearly separable gates like AND, OR, NAND, NOR |
| **Binary Classification** | Spam vs. not-spam email detection, pass/fail prediction |
| **Image Processing** | Basic edge/pattern detection in early computer vision pipelines |
| **Signal Processing** | Simple threshold-based signal classification |
| **Credit Scoring** | Approve/reject decisions based on linearly separable financial features |
| **Medical Diagnosis (basic)** | Presence/absence classification on linearly separable features |
| **Foundational Building Block** | Core computational unit used inside Multi-Layer Perceptrons (MLPs) and modern deep neural networks |
| **Education** | Teaching the fundamentals of supervised learning, weight updates, and neural computation |

---

## ⚠️ Limitations
- Can only solve **linearly separable** problems (fails on the XOR gate).
- Sensitive to input feature scaling and weight initialization.
- Produces strictly binary output — no probability/confidence score.
- Convergence is not guaranteed if the dataset is not linearly separable.

> These limitations are precisely what motivate the next step in deep learning: **Multi-Layer Perceptrons (MLPs)** with hidden layers and non-linear activation functions, covered in later labs of **BDA404-5N (ANN & Deep Learning)**.

---

## 📚 Concepts Covered
- Perceptron
- Binary Classification
- Step Activation Function
- Weight Update Rule (Perceptron Learning Rule)
- Supervised Learning
- Linear Separability
- Epoch-based Training
- NumPy

---

## 🔖 References
- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.* Psychological Review.
- Minsky, M. & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry.* MIT Press.

---

## 👤 Author
**Divyansh**
BSc Data Science and AI, Christ University
Deep Learning Laboratory — BDA404-5N
