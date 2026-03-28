# Perceptron Trick

## Core Objective
The primary goal is to learn how to "train" a Perceptron. Training involves finding the correct values for weights ($w_1, w_2, \dots$) and bias ($b$) so that a linear boundary (a line in 2D or a hyperplane in higher dimensions) can accurately separate data into two classes (e.g., student placement: "Placed" vs. "Not Placed").

### 1. Mathematical Intuition: The Linear Boundary

- **Linear Separability**: The Perceptron works on the assumption that data is linearly separable, meaning a single straight line can distinguish between different classes.

- **The Equation**: In 2D, the boundary is represented as $Ax + By + C = 0$. In deep learning notation, this is often written as $w_1x_1 + w_2x_2 + b = 0$.

- **The Problem**: When we start, we initialize weights and bias randomly. This often results in a "terrible" line that misclassifies many points. The training process is the journey of moving this random line to a position where it correctly classifies the data.

### 2. The Perceptron Trick: Moving the Line

The "trick" involves iteratively adjusting the line based on misclassified points.

- **The Loop**: You run a loop for a set number of iterations (epochs). In each iteration, you pick a random point from the dataset.

- **Feedback Mechanism**:

    - If a point is correctly classified, no changes are made to the weights.

    - If a point is misclassified, the line must be "pulled" toward the point (if it's a positive point in a negative region) or "pushed" away (if it's a negative point in a positive region).

### 3. Identifying Regions (Positive vs. Negative)

To know if a point is misclassified, we must identify which side of the line is positive ($>0$) and which is negative ($<0$).

- By plugging a point's coordinates $(x, y)$ into the line equation $Ax + By + C$, the resulting sign tells us the region.

- Transformation Logic:

    - Changing the constant ($C$) shifts the line parallelly.

    - Changing the coefficients ($A$ and $B$) rotates the line.

    - Combining these changes allows the line to move and rotate toward a specific misclassified point.

### 4. The Update Rule (The Math)

When a point $(x_i, y_i)$ is misclassified, the weights are updated using a small step called the Learning Rate ($\eta$). The learning rate (typically a small value like 0.01 or 0.1) prevents the line from jumping too drastically.

- **Case 1: Positive point in Negative Region**

    - $w_{new} = w_{old} + \eta \cdot x_i$

    - $b_{new} = b_{old} + \eta$

- **Case 2: Negative point in Positive Region**

    - $w_{new} = w_{old} - \eta \cdot x_i$

    - $b_{new} = b_{old} - \eta$

### 5. Unified Programming Logic

To simplify coding, a single rule can be used by incorporating the actual label ($y$) and the predicted label ($\hat{y}$):

- Update Rule: $W_{new} = W_{old} + \eta \cdot (y_i - \hat{y}_i) \cdot X_i$

- If $y_i = \hat{y}_i$ (correct classification), the term $(y_i - \hat{y}_i)$ becomes zero, and no update occurs.

- If they differ, the sign of the difference automatically determines whether to add or subtract, moving the line in the correct direction.

### 6. Epochs and Convergence

A single pass through the data is rarely enough. We run the loop for hundreds or thousands of Epochs.

- In every epoch, the line wiggles and shifts.

- Eventually, the line "converges"—it reaches a position where it no longer needs to move because it has separated the classes as well as possible.

### 7. Implementation Summary
To build this in code, you would:

1. Initialize $W = [1, 1, 1]$ (two weights and one bias).

2. Set $\eta = 0.1$.

3. For $1000$ iterations:
    - Pick a random row $(x_1, x_2, y)$.
    - Calculate $sum = (w_1 \cdot x_1) + (w_2 \cdot x_2) + b$.
    - If $sum \geq 0$, $\hat{y} = 1$; else $\hat{y} = 0$.
    - Update $w_1 = w_1 + \eta \cdot (y - \hat{y}) \cdot x_1$.
    - Update $w_2 = w_2 + \eta \cdot (y - \hat{y}) \cdot x_2$.
    - Update $b = b + \eta \cdot (y - \hat{y})$.
4. Result: The final $w_1, w_2, b$ define the perfect line.

## Key Takeaway
Training a Perceptron is an iterative process of trial and error. By looking at one point at a time and making tiny adjustments to the line's orientation and position, the algorithm eventually "converges" on a line that separates the data classes effectively.