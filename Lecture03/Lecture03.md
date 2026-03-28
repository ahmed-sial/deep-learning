# Perceptron

## What is Perceptron?

### 1. Introduction to Perceptron

- **Definition:** A Perceptron is a supervised machine learning algorithm used for binary classification.

- **Significance:** It serves as the basic unit or "building block" for more complex architectures like Multi-Layer Perceptrons (MLP) and Deep Neural Networks.

- **Evolution:** Understanding a single perceptron is crucial before moving to deep learning, as it simplifies the learning of how models are trained.

### 2. Mathematical Design and Model

A Perceptron can be viewed as a mathematical function or model consisting of several key components:

- **A. Inputs ($x_1, x_2, ..., x_n$)**

  - These represent the features of your data (e.g., a student's IQ and CGPA).

  - The number of inputs corresponds to the number of input columns in your dataset.

- **B. Weights ($w_1, w_2, ..., w_n$) and Bias ($b$)**

  - **Weights:** Each input is assigned a weight, representing the "strength" or importance of input in the final decision.

  - **Bias:** An additional parameter (often represented as a connection with input 1) that allows the model to shift the decision boundary.

- **C. Summation Block ($z$)**
  - The model calculates the dot product of inputs and weights and adds the bias:$$z = (w_1 \cdot x_1) + (w_2 \cdot x_2) + ... + (w_n \cdot x_n) + b$$

- **D. Activation Function**

  - The value $z$ is passed through an activation function to map the output into a specific range.
  - **Step Function:** A common activation function for the basic perceptron.

    - If $z \geq 0$, Output = 1

    - If $z < 0$, Output = 0

```text
Inputs:     x1   x2   x3
             │    │    │
             ▼    ▼    ▼

Weighted:  w1x1 w2x2 w3x3
             │    │    │
             └────┴────┘
                  │
                  ▼

            z = w1x1 + w2x2 + w3x3 + b

                  │
                  ▼

        Activation Function (Step)

             z ≥ 0  →  1
             z < 0  →  0
```

### 3. How the Perceptron Works

There are two main stages in using a Perceptron:

**Training**

  - The goal of training is to find the optimal values for weights ($w$) and bias ($b$).

  - The model uses labeled training data to iteratively adjust these parameters until it can correctly classify the inputs.

**Prediction**

- Once $w$ and $b$ are fixed, new data is fed into the model.
- The model calculates $z$ and passes it through the activation function to predict the class (e.g., "Placed" vs. "Not Placed").

### 4. Perceptron vs. Biological Neuron

Deep learning is inspired by the human nervous system, and the Perceptron is an artificial version of a biological neuron.

**Similarities**

- **Inputs:** Dendrites in a neuron are analogous to the input connections in a perceptron.

- **Processing:** The Nucleus (cell body) performs calculations, similar to the summation and activation blocks.

- **Output:** The Axon carries the output signal, similar to the final output of the perceptron.

**Differences**

- **Complexity:** Biological neurons are infinitely more complex and involve electro-chemical reactions, whereas perceptrons use simple math.

- **Processing:** We don't fully understand how biological neurons process information, but perceptron math is transparent.

- **Neuroplasticity:** Biological connections change thickness or disappear over time (learning), while perceptron connections are mathematically fixed once trained.

### 5. Geometric Intuition

Visualizing the perceptron helps understand its behavior:

- **2D Space:** In a 2D plane (two inputs), the perceptron equation ($w_1x_1 + w_2x_2 + b = 0$) represents a straight line.

- **3D Space:** With three inputs, it represents a plane.

- **Higher Dimensions:** It represents a hyperplane.

- **Decision Regions:** The line/plane divides the space into two regions. Points falling on one side are classified as Class A, and points on the other side as Class B.

### 6. Limitations: Linear Separability

- **The Main Constraint:** A Perceptron can only solve problems where the data is linearly separable (meaning a straight line or plane can perfectly divide the classes).

- **Failure on Non-Linear Data:** If the data points are mixed or follow a circular/complex pattern, a single perceptron cannot achieve high accuracy. This limitation led to the development of Multi-Layer Perceptrons (MLPs).

### 7. Practical Interpretation of Weights

- **Feature Importance:** The magnitude of the weights indicates the importance of a feature.

- **Example:** If $w_{CGPA} = 4$ and $w_{IQ} = 2$, the model considers CGPA twice as important as IQ in predicting student placement.

### 8. Practical Implementation (Scikit-Learn)

- You can implement a Perceptron using the `Perceptron` class from the `sklearn.linear_model` library.

- **Key steps in code:**
    1. Import `Perceptron` from `sklearn`.

    2. Create an object: `model = Perceptron()`.

    3. Train the model: `model.fit(X_train, y_train)`.

    4. Inspect weights: `model.coef_` gives weights, and `model.intercept_` gives the bias.

    5. Visualize: Use libraries like `MLxtend` to plot the decision regions.