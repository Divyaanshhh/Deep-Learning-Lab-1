# Deep Learning Lab 1 - Perceptron using NumPy

A simple implementation of the **Perceptron Algorithm** in Python using **NumPy**. This program demonstrates binary classification by simulating the **AND Logic Gate**, and extends the concept to a real-world **Spam Email Detection** system.

---

## 📌 Objective
To implement a simple Perceptron model using NumPy and train it to classify the outputs of an AND gate, and to understand the foundational concepts of neural computation that lead into deep learning. Additionally, to apply the same Perceptron learning rule to a practical **Spam Email Detection** problem using real-world email features.

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
│── Lab-1.py                          # AND Gate Perceptron
│── perceptron_spam.ipynb             # Spam Email Detection (Jupyter/Colab)
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

### Part A: AND Gate Classification
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

### Part B: Spam Email Detection (Practical Application)
The same Perceptron algorithm is applied to a real-world binary classification problem — detecting whether an email is **Spam (1)** or **Not Spam (0)**.

#### 🎯 Problem Statement
Build a simple neural network classifier that learns to flag spam emails based on three binary features extracted from the email content.

#### 📧 Feature Engineering
We manually extract three key binary features from each email:

| Feature | Description | Value |
|:---|:---|:---:|
| **X1** | Contains Link | 1 = Yes, 0 = No |
| **X2** | Contains FREE keyword | 1 = Yes, 0 = No |
| **X3** | Has Attachment | 1 = Yes, 0 = No |

These features are chosen because they are strong indicators of spam:
- **Links**: Spam emails often contain suspicious or phishing links.
- **"FREE" keyword**: A common spam trigger word used in promotional or fraudulent emails.
- **Attachments**: Spam emails frequently include malicious or unwanted attachments.

#### 📊 Training Dataset

| Contains Link (X1) | Contains FREE (X2) | Has Attachment (X3) | Label (Spam?) |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0 (Not Spam) |
| 1 | 1 | 1 | 1 (Spam) |
| 1 | 0 | 0 | 0 (Not Spam) |
| 1 | 1 | 0 | 1 (Spam) |
| 0 | 1 | 1 | 1 (Spam) |
| 0 | 0 | 1 | 0 (Not Spam) |
| 1 | 0 | 1 | 1 (Spam) |
| 0 | 1 | 0 | 0 (Not Spam) |

#### 🔄 How the Perceptron Learns for Spam Detection
1. **Initialization**: Weights and bias are set to zero.
2. **Forward Pass**: For each email, compute the weighted sum of features plus bias.
3. **Activation**: Apply the step function to get a binary prediction (Spam or Not Spam).
4. **Error Calculation**: Compare the predicted output with the true label.
5. **Weight Update**: If misclassified, adjust weights and bias using the Perceptron Learning Rule. Features that strongly correlate with spam receive higher weights.
6. **Convergence**: Repeat for multiple epochs until the model correctly classifies all training emails.

#### ✅ Expected Output
```
Final Weights: [0.2 0.2 0.1]
Final Bias: -0.1

Predictions:
[0 0 0] -> 0
[1 1 1] -> 1
[1 0 0] -> 0
[1 1 0] -> 1
[0 1 1] -> 1
[0 0 1] -> 0
[1 0 1] -> 1
[0 1 0] -> 0
```
> *Note: The exact weight values may vary slightly depending on the order of training samples and learning rate, but the final predictions should correctly classify all emails.*

#### 🔗 Notebook
Open `perceptron_spam.ipynb` in **Google Colab** or **Jupyter Notebook** to run the Spam Detection experiment interactively.

---

## 📊 Input Dataset (AND Gate)

| Input 1 | Input 2 | Expected Output |
|:-------:|:-------:|:----------------:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

---

## ✅ Expected Output (AND Gate)
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
- Feature Engineering for Spam Detection
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
