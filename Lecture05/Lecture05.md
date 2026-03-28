# Perceptron Loss Function

## 1. The Perceptron Architecture (Review)

The Perceptron is the fundamental building block of neural networks. It operates on two main stages:

- **Linear Summation ($z$)**: Calculates the weighted sum of inputs plus a bias:$$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

- **Step Function**: A thresholding mechanism where:

    - If $z \geq 0$, Output = $1$

    - If $z < 0$, Output = $0$

## 2. Limitations of the "Perceptron Trick"

While the Perceptron Trick allows for basic learning by shifting the decision boundary whenever a point is misclassified, it has significant drawbacks:

- Lack of Convergence: On non-linearly separable data, the trick will loop infinitely, constantly shifting the line without ever finding an "optimal" solution.

- No Measure of "Badness": It treats all misclassifications the same. A point that is barely on the wrong side of the line triggers the same update as a point that is very far away.

- Optimization Gap: It is a heuristic, not a mathematically grounded optimization process. To improve this, we need a Loss Function that we can minimize using Gradient Descent.

## 3. Geometric Intuition of Loss

Geometrically, a Perceptron is a hyperplane (a line in 2D) that divides space.

- **Distance as Error**: The mathematical intuition is that the distance of a misclassified point from the decision boundary represents the "error."

- **Correct Classification**: If a point is on the correct side, its contribution to the loss should be $0$.

- **Misclassification**: If a point is on the wrong side, the loss should be proportional to its distance from the line.

## 4. Mathematical Foundation: The Loss Function

To train the model properly, we define a loss function $L$ and minimize it.

### A. Perceptron Loss (Hinge Loss Variant)
The goal is to minimize the sum of distances of misclassified points.

The distance $d$ from a point $(x_i, y_i)$ to the line $w \cdot x + b = 0$ is:$$d = \frac{|w \cdot x_i + b|}{\|w\|}$$

Since we only care about misclassified points, we use the property that for a misclassified point, $y_i (w \cdot x_i + b) < 0$. The Perceptron Loss is defined as:$$L(w, b) = \sum_{i \in \mathcal{M}} -y_i(w \cdot x_i + b)$$

*(Where $\mathcal{M}$ is the set of misclassified points).*

### B. The Sigmoid Function & Probabilistic Intuition

The standard step function is "hard" and non-differentiable at zero, which prevents Gradient Descent from working. To fix this, we replace the Step Function with the Sigmoid Function:$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- **S-Shape**: It squashes any input into a range between $0$ and $1$.

- **Differentiability**: Unlike the step function, Sigmoid is smooth and has a defined derivative everywhere, allowing us to calculate gradients.

## 5. Binary Cross Entropy (Log Loss)

When using the Sigmoid function, we view the output as a probability. If the model predicts $0.8$, it believes there is an 80% chance the point belongs to Class 1.

The Binary Cross Entropy (BCE) loss is used to penalize the model based on the difference between the predicted probability ($\hat{y}$) and the actual label ($y$):$$L = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$$

- **Intuition**: If the true label is $1$ and the model predicts a low probability (e.g., $0.1$), the $\log$ term becomes very large, creating a high penalty.

## 6. Summary of Configurations
The Perceptron model is flexible based on the chosen components:

1. Step Function + Perceptron Loss: Standard Perceptron (Classical approach).

2. Sigmoid + Binary Cross Entropy: Logistic Regression (Probabilistic approach).

3. Linear Activation + Mean Squared Error: Linear Regression (Used for predicting continuous numbers).

## Key Takeaway
Training a Perceptron is not about "guessing" where the line goes. It is about defining a mathematical "surface" (the Loss Function) and using calculus (Gradient Descent) to find the lowest point on that surface, which corresponds to the best possible weights and bias for the model.