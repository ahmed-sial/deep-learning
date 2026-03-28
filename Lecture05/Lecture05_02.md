# Loss Functions in Perceptron

## 1. Recap: What is a Perceptron?

A **Perceptron** is a mathematical model inspired by a biological neuron in the human brain. It is the fundamental building block of all neural networks.

### Structure

Consider a binary classification problem — predicting whether a student gets placed, based on:
- **x₁** = CGPA
- **x₂** = IQ score

The perceptron computes a **weighted sum** of inputs plus a bias:

$$z = w_1 x_1 + w_2 x_2 + b$$

This is passed through an **activation function** (Step Function):

$$\hat{y} = \begin{cases} +1 & \text{if } z \geq 0 \\ -1 & \text{if } z < 0 \end{cases}$$

### Perceptron Architecture Diagram

```
┌─────────────┐        w₁
│  x₁ (CGPA)  │ ──────────────────┐
└─────────────┘                   │
                                  ▼
┌────────────┐        w₂     ╔══════╗     Step     ╔════════╗
│  x₂  (IQ)  │ ─────────────►║  Σz  ║ ──────────►  ║  ŷ     ║
└────────────┘               ╚══════╝  f(z)        ╚════════╝
                                  ▲       (+1/−1)
┌─────────────┐        b          │
│  1 (bias)   │ ──────────────────┘
└─────────────┘

        ↓  During Training
 ┌──────────────────────────────┐
 │   Loss Function L(w₁,w₂,b)   │  ← quantifies error
 └──────────────────────────────┘
```

### Geometric Intuition

The equation $w_1 x_1 + w_2 x_2 + b = 0$ defines a **straight line** (in 2D) or a **hyperplane** (in higher dimensions). This line divides the feature space into:

- **Positive region** (z > 0) → predicted class +1
- **Negative region** (z < 0) → predicted class −1

This is why the perceptron performs **binary classification** — it is literally drawing a line between two classes.

```
         x₂
          │
    ●  ●  │  ○  ○       ● = Class +1 (placed)
    ●     │     ○       ○ = Class −1 (not placed)
  ────────┼────────── x₁
          │  ← decision boundary
          │    w₁x₁ + w₂x₂ + b = 0
```

---

## 2. Training vs. Prediction

Every machine learning algorithm has two phases:

| Phase | Goal | What changes? |
|---|---|---|
| **Prediction** | Given w₁, w₂, b → compute ŷ | Nothing — fixed weights |
| **Training** | Find the best w₁, w₂, b | Weights update iteratively |

---

## 3. Problems with the Perceptron Trick

The **Perceptron Trick** (previously studied) is a heuristic:
- Start with a random line
- Pick random points; if misclassified, "pull" the line toward that point
- Repeat thousands of times

### Problem 1: No Measure of Quality

After the trick converges, you get *a* line — but you cannot tell if it is the *best* line.

```
Two valid lines — which is better?

      x₂
       │   ╱ Line B (better margin)
  ●  ● │ ╱╱ Line A (poor margin)
  ●    │╱
  ─────┼────── x₁
       │  ○  ○
       │     ○
```

Both lines correctly separate the classes — but Line B has much more margin. The Perceptron Trick **cannot distinguish** between them, because it stops updating as soon as all points are correctly classified.

### Problem 2: Convergence Not Guaranteed

Because points are picked **randomly**, there is a theoretical possibility that only already-correctly-classified points are selected, causing the line to never move.

### The Solution: Loss Functions

---

## 4. What is a Loss Function?

A **Loss Function** (also: cost function, objective function) is a mathematical function that maps model parameters to a **single number** representing how poorly the model performs.

$$L = f(w_1, w_2, b)$$

**Intuition:** Think of the loss as a "penalty score." A bad line → high penalty. A good line → low (ideally zero) penalty.

```
Loss
  L
  │  ╲
  │    ╲
  │      ╲_________
  │               ╲___
  └──────────────────── w₁, w₂, b
                  ↑
           optimal parameters (minimum loss)
```

**Goal of training:**

$$w_1^*, w_2^*, b^* = \arg\min_{w_1, w_2, b} L(w_1, w_2, b)$$

Find the parameter values where the loss is **minimum**.

### Why Loss Functions Solve Both Problems

| Problem | How Loss Functions Solve It |
|---|---|
| No quality measure | Compare any two lines by computing their loss — lower loss = definitively better |
| Convergence issues | Gradient Descent on a smooth loss function has guaranteed convergence (under mild conditions) |

---

## 5. Building a Loss Function for the Perceptron

The instructor walks through progressively better loss functions.

### Attempt 1: Count of Misclassified Points

$$L = \text{Number of misclassified points}$$

**Example:** Line A misclassifies 7 points → L = 7. Line B misclassifies 5 → L = 5. Line B is better.

**Problem:** Treats all misclassified points equally. A point barely on the wrong side gets the same penalty as a point far on the wrong side. This is not accurate — farther mistakes are bigger mistakes.

### Attempt 2: Sum of Distances (Better)

$$L = \sum_{\text{misclassified } i} d_i$$

where $d_i$ is the **perpendicular distance** of misclassified point $i$ from the decision boundary.

```
       ╱ line
  ●   ╱   ← d₂ = 13  (big penalty)
     ╱
●   ╱       ← d₁ = 10
   ╱

Total L = 10 + 13 = 23
```

**Better:** Points farther from the line contribute more to the loss, guiding optimization more accurately.

**Problem:** Computing perpendicular distance requires dividing by $\|w\|$ — expensive. There is a simpler proxy.

### The Perceptron's Proxy for Distance

Rather than computing the actual perpendicular distance, the perceptron plugs the misclassified point's coordinates directly into the line equation:

$$\text{proxy}_i = w_1 x_{1i} + w_2 x_{2i} + b = z_i$$

This quantity is **proportional to the perpendicular distance** (differs only by a constant $\|w\|$), and is far cheaper to compute since it's just a dot product.

**Key insight:** A point far from the line on the wrong side gives a large $|z_i|$; a point close to the boundary gives a small $|z_i|$.

---

## 6. The Perceptron Loss (Hinge Loss) — Full Formula

The official loss function used in scikit-learn's SGD-based perceptron is:

$$\boxed{L(w_1, w_2, b) = \frac{1}{N} \sum_{i=1}^{N} \max\!\left(0,\ -y_i \cdot z_i\right)}$$

where:
- $z_i = w_1 x_{1i} + w_2 x_{2i} + b$ (the raw score for point $i$)
- $y_i \in \{-1, +1\}$ (true label — **must be −1 and +1**, not 0 and 1)
- $N$ = total number of data points
- $\max(0, \cdot)$ = the **ReLU** (Rectified Linear Unit) function / hinge

> **Important:** Labels must be $+1$ and $-1$ for this formula to work correctly. Using 0 and 1 would break the sign-based logic.

Compact form with the proxy $z_i$:

$$L = \frac{1}{N} \sum_{i=1}^{N} \max(0,\ -y_i z_i)$$

Expanded with coordinates substituted:

$$L = \frac{1}{N} \sum_{i=1}^{N} \max\!\left(0,\ -y_i(w_1 x_{1i} + w_2 x_{2i} + b)\right)$$

### What Does max(0, ·) Do?

The `max(0, x)` function is simply:

$$\max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

```
Output
  │      ╱
  │    ╱
  │  ╱
  │╱
──┼────────── x
  0
```

This means the hinge loss "activates" (adds a penalty) only when the inner term is positive — i.e., only when a point is **misclassified**.

---

## 7. Geometric Intuition of the Hinge Loss — Four Cases

For any line and dataset, every point falls into exactly one of four categories. The loss term $\max(0, -y_i z_i)$ behaves differently for each.

```
                Decision Boundary (the line)
                        │
     Negative side (−)  │  Positive side (+)
                        │
```

### Case 1: Green Point (+1) Correctly Classified

- True label: $y_i = +1$
- Point is on positive side: $z_i > 0$
- Compute: $-y_i \cdot z_i = -(+1)(+) = \text{negative}$
- $\max(0, \text{negative}) = 0$

**✓ Zero contribution to loss.**

### Case 2: Red Point (−1) Correctly Classified

- True label: $y_i = -1$
- Point is on negative side: $z_i < 0$
- Compute: $-y_i \cdot z_i = -(-1)(-) = -(+) = \text{negative}$
- $\max(0, \text{negative}) = 0$

**✓ Zero contribution to loss.**

### Case 3: Green Point (+1) Misclassified (on wrong/negative side)

- True label: $y_i = +1$
- Point is on negative side: $z_i < 0$
- Compute: $-y_i \cdot z_i = -(+1)(-) = \text{positive}$
- $\max(0, \text{positive}) = \text{that positive number}$

**✗ Positive contribution to loss — proportional to distance from line.**

### Case 4: Red Point (−1) Misclassified (on wrong/positive side)

- True label: $y_i = -1$
- Point is on positive side: $z_i > 0$
- Compute: $-y_i \cdot z_i = -(-1)(+) = \text{positive}$
- $\max(0, \text{positive}) = \text{that positive number}$

**✗ Positive contribution to loss — proportional to distance from line.**

### Case Diagram

```
┌─────────────────────────────────────────────────────────┐
│              HINGE LOSS — Four Cases                    │
├──────────────────────────┬──────────────────────────────┤
│  Case 1 ✓ Green, correct │  Case 2 ✓ Red, correct       │
│  y=+1, z>0               │  y=−1, z<0                   │
│  −y·z = negative         │  −y·z = negative             │
│  max(0,neg) = 0          │  max(0,neg) = 0              │
│  → Zero loss             │  → Zero loss                 │
├──────────────────────────┼──────────────────────────────┤
│  Case 3 ✗ Green, WRONG   │  Case 4 ✗ Red, WRONG         │
│  y=+1, z<0 (wrong side)  │  y=−1, z>0 (wrong side)      │
│  −y·z = positive         │  −y·z = positive             │
│  max(0,pos) = pos        │  max(0,pos) = pos            │
│  → PENALTY (∝ distance)  │  → PENALTY (∝ distance)     │
└──────────────────────────┴──────────────────────────────┘
```

### Key Geometric Insight

$$\text{loss per point} = \begin{cases} 0 & \text{if correctly classified} \\ |z_i| \propto \text{distance from line} & \text{if misclassified} \end{cases}$$

The sign manipulation with $y_i$ elegantly extracts just this behavior. The sign of $y_i \cdot z_i$ tells you whether a point is correctly classified:
- $y_i \cdot z_i > 0$ → correct (both same sign)
- $y_i \cdot z_i < 0$ → wrong (opposite signs)

---

## 8. The Mathematical Objective

Formally:

$$w_1^*, w_2^*, b^* = \arg\min_{w_1, w_2, b} \frac{1}{N} \sum_{i=1}^{N} \max(0, -y_i z_i)$$

This is the **optimization problem** we need to solve. The solution gives us the best-possible decision boundary for this loss function.

**This solves both perceptron trick problems:**
1. Any two lines can now be compared by computing their loss values.
2. Gradient descent provides a principled, convergent optimization procedure.

---

## 9. Minimizing the Loss — Gradient Descent

To minimize $L(w_1, w_2, b)$, we use **Gradient Descent** (specifically, Stochastic Gradient Descent — SGD):

### Update Rules

$$w_1 \leftarrow w_1 - \eta \cdot \frac{\partial L}{\partial w_1}$$

$$w_2 \leftarrow w_2 - \eta \cdot \frac{\partial L}{\partial w_2}$$

$$b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

where $\eta$ (eta) is the **learning rate** — a small positive constant (e.g., 0.1).

**Intuition:** Like a ball rolling down a hill. The gradient points uphill; subtracting it moves you downhill toward the minimum.

### Training Loop

```
START
  │
  ▼
Initialize w₁, w₂, b (randomly or to 1)
  │
  ▼
┌─────────────────────────────────────┐
│  For epoch = 1 to 1000:            │
│    For each point (xᵢ, yᵢ):        │
│      Compute zᵢ = w₁x₁ᵢ+w₂x₂ᵢ+b  │
│      If yᵢ · zᵢ < 0  (misclass.): │
│        w₁ ← w₁ + η·yᵢ·x₁ᵢ        │
│        w₂ ← w₂ + η·yᵢ·x₂ᵢ        │
│        b  ← b  + η·yᵢ             │
└─────────────────────────────────────┘
  │
  ▼
Return optimal w₁*, w₂*, b*
  │
  ▼
 END
```

---

## 10. Deriving the Update Rules

### Step 1: Write the Loss Function

$$L = \frac{1}{N} \sum_{i=1}^{N} \max(0,\ -y_i z_i), \quad z_i = w_1 x_{1i} + w_2 x_{2i} + b$$

### Step 2: Identify Piecewise Behavior

The $\max(0, \cdot)$ function creates two cases:
- **Case A** (correctly classified, $-y_i z_i \leq 0$): the max gives 0. Derivative = 0.
- **Case B** (misclassified, $-y_i z_i > 0$): the max gives $-y_i z_i$ itself.

For Case B, apply the chain rule:

$$\frac{\partial}{\partial w_1}(-y_i z_i) = -y_i \cdot \frac{\partial z_i}{\partial w_1} = -y_i \cdot x_{1i}$$

Similarly for $w_2$ and $b$:

$$\frac{\partial L}{\partial w_1} = -y_i x_{1i}, \quad \frac{\partial L}{\partial w_2} = -y_i x_{2i}, \quad \frac{\partial L}{\partial b} = -y_i$$

(summed over misclassified points only)

### Step 3: Apply Gradient Descent

$$w_1 \leftarrow w_1 - \eta \cdot (-y_i x_{1i}) = w_1 + \eta \cdot y_i x_{1i}$$

$$w_2 \leftarrow w_2 - \eta \cdot (-y_i x_{2i}) = w_2 + \eta \cdot y_i x_{2i}$$

$$b \leftarrow b - \eta \cdot (-y_i) = b + \eta \cdot y_i$$

The two minus signs cancel, giving a **plus** in the final update rules.

> **Note:** Update is applied **only** for misclassified points where $y_i \cdot z_i < 0$.

---

## 11. Implementation Notes (Python)

```python
import numpy as np

def perceptron(X, y, learning_rate=0.1, epochs=1000):
    """
    X: feature matrix, shape (N, 2) — two input columns
    y: labels array, shape (N,)   — must be +1 and -1
    """
    w1, w2, b = 1.0, 1.0, 1.0   # initialize weights

    for _ in range(epochs):
        for i in range(len(X)):
            x1i = X[i, 0]
            x2i = X[i, 1]
            yi  = y[i]

            # Compute z (dot product + bias)
            zi = w1 * x1i + w2 * x2i + b

            # Check misclassification: yi * zi < 0
            if yi * zi < 0:
                w1 += learning_rate * yi * x1i
                w2 += learning_rate * yi * x2i
                b  += learning_rate * yi

    return w1, w2, b

# IMPORTANT: Labels must be +1 and -1, NOT 0 and 1
# y = np.where(y_raw == 1, 1, -1)
```

**Key notes:**
- `learning_rate = 0.1` is a typical starting value.
- Run for `1000` epochs — enough for most linearly separable datasets.
- The condition `yi * zi < 0` is the misclassification check.
- Labels **must** be $\{-1, +1\}$ for the hinge loss formula to work.

---

## 12. The Flexibility of the Perceptron

> **This is the most important section of the lecture.**

The Perceptron is not just one algorithm — it is a **flexible mathematical framework**. By changing just two components, it becomes entirely different algorithms:

1. The **Activation Function** applied to $z$
2. The **Loss Function** used for training

### The Four Configurations

```
┌──────────────────┬──────────────────────┬────────────────────┬─────────────┐
│ Activation f(z)  │ Loss Function        │ = Algorithm        │ Output      │
├──────────────────┼──────────────────────┼────────────────────┼─────────────┤
│ Step             │ Hinge Loss           │ Perceptron         │ +1 or −1    │
│ z≥0→+1, z<0→−1  │ max(0, −y·z)         │                    │ hard class  │
├──────────────────┼──────────────────────┼────────────────────┼─────────────┤
│ Sigmoid σ(z)     │ Binary Cross Entropy │ Logistic Regression│ P(y=1)∈(0,1)│
│ 1/(1+e⁻ᶻ)       │ −[y·log(ŷ)+          │                    │ probability │
│                  │  (1−y)·log(1−ŷ)]    │                    │             │
├──────────────────┼──────────────────────┼────────────────────┼─────────────┤
│ Softmax          │ Categorical Cross    │ Softmax Regression │ P(class j)  │
│ eᶻʲ/Σeᶻ         │ Entropy              │ (multi-class)      │ for each j  │
│                  │ −Σ yⱼ·log(ŷⱼ)       │                    │             │
├──────────────────┼──────────────────────┼────────────────────┼─────────────┤
│ Linear (identity)│ MSE                  │ Linear Regression  │ any real ŷ  │
│ f(z) = z         │ (1/N)·Σ(ŷ−y)²       │                    │ number      │
└──────────────────┴──────────────────────┴────────────────────┴─────────────┘
```

### How to Apply a New Configuration

```
Same Perceptron Structure:

  x ──► [Σ z = w·x + b] ──► [Activation f(z)] ──► ŷ
                                                     │
                                              [Loss Function]
                                              compares ŷ to y
                                                     │
                                             [Gradient Descent]
                                              updates w, b
```

To switch algorithm, simply replace the activation block and the loss block. The summation layer, gradient descent procedure, and parameter update loop remain **identical**.

---

## 13. Sigmoid Function — Deep Dive

### Formula

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Key Properties

| Property | Value |
|---|---|
| Domain | $(-\infty, +\infty)$ |
| Range | $(0, 1)$ — strictly, never reaches 0 or 1 |
| $\sigma(0)$ | $0.5$ — midpoint |
| As $z \to +\infty$ | $\sigma(z) \to 1$ |
| As $z \to -\infty$ | $\sigma(z) \to 0$ |
| **Derivative** | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ |
| Monotone? | Yes — always increasing |
| Differentiable? | Yes — everywhere |

### Plot: Step vs Sigmoid

```
Step Function                   Sigmoid Function
─────────────────────────       ─────────────────────────
f(z)                            σ(z)
 +1 │         ───────            1 │              ─────────
    │                              │            ╱
    │                           0.5│──────────╱────────────
    │                              │        ╱
 −1 │─────────                   0 │────────
    └──────────────── z            └──────────────────── z
         0                              0
 Hard, not differentiable      Smooth, differentiable everywhere
```

### Why Sigmoid Instead of Step?

The **Step function is not differentiable** at $z = 0$. Since Gradient Descent requires computing $\partial L / \partial w$, the Step function breaks the training procedure. Sigmoid is **smooth and differentiable everywhere**, making it compatible with gradient-based optimization.

### Geometric Intuition of Sigmoid

Sigmoid is a **soft step function**. Instead of abruptly switching from 0 to 1 at $z=0$:
- Points far on the **positive side** → σ(z) close to 1 → model is *confident* it's class +1
- Points far on the **negative side** → σ(z) close to 0 → model is *confident* it's class 0
- Points **near the boundary** → σ(z) ≈ 0.5 → model is *uncertain*

The output is now a **probability**, not a hard decision.

---

## 14. Binary Cross Entropy — Deep Dive

Used with the Sigmoid activation. Labels here are $y \in \{0, 1\}$ (not −1/+1).

### Formula

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right]$$

where $\hat{y}_i = \sigma(z_i) \in (0, 1)$.

### Case Analysis

**Case: $y = 1$ (true positive)**

$$L_i = -\log(\hat{y}_i)$$

| $\hat{y}$ (predicted) | $-\log(\hat{y})$ (loss) |
|---|---|
| 0.99 (correct, confident) | ≈ 0.01 (tiny loss) |
| 0.50 (uncertain) | ≈ 0.69 |
| 0.01 (wrong, confident) | ≈ 4.61 (huge loss!) |

**Case: $y = 0$ (true negative)**

$$L_i = -\log(1 - \hat{y}_i)$$

Same logic applies symmetrically.

### Why Logarithmic Penalty?

The logarithm grows very fast as predictions approach the wrong extreme. This means:
- Being **wrong** and **confident** is penalized **extremely heavily**.
- This strongly discourages the model from making confidently incorrect predictions.
- Linear penalties would not be as aggressive in this regard.

```
Loss
 ∞ │╲
   │  ╲
   │    ╲
 1 │      ╲____
   │            ─────────
 0 └───────────────────── ŷ
   0    0.5              1
   ↑                     ↑
 huge loss            zero loss
(when y=1)            (when y=1)
```

---

## 15. Categorical Cross Entropy & Softmax

Used when there are **more than 2 classes**.

### Softmax Activation

For $K$ classes, produces a probability distribution over all classes:

$$\sigma(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}, \quad \text{for each class } j$$

**Properties:**
- All outputs are in $(0, 1)$
- All outputs **sum to exactly 1** → forms a proper probability distribution
- Class with highest $z_j$ gets the highest probability

### Categorical Cross Entropy Loss

$$L = -\sum_{j=1}^{K} y_j \cdot \log(\hat{y}_j)$$

Since $y_j = 1$ for the true class and $y_j = 0$ for all others (one-hot encoding), this simplifies to:

$$L = -\log(\hat{y}_{\text{true class}})$$

**Intuition:** The loss is simply the **negative log of the predicted probability assigned to the correct class**. The closer that probability is to 1, the smaller the loss.

### Example (3 classes)

Suppose the true class is "A", and the model predicts:

| Prediction | P(A) | P(B) | P(C) | Loss = −log P(A) |
|---|---|---|---|---|
| Confident & correct | 0.90 | 0.05 | 0.05 | 0.105 (small) |
| Uncertain | 0.33 | 0.33 | 0.34 | 1.109 |
| Confident & wrong | 0.05 | 0.90 | 0.05 | 2.996 (large!) |

---

## 16. The Unification Insight

The deepest insight of this lecture:

> **Logistic Regression, Softmax Regression, Linear Regression, and the Perceptron are all the same mathematical model — a perceptron — with different activation and loss functions.**

### Why This Matters

**1. Unified training procedure:** All are trained with gradient descent on the loss function. Only the loss being differentiated changes.

**2. Easy to extend:** New problems may only need a new activation-loss combination, not a new algorithm.

**3. Foundation for deep learning:** Stacking these perceptrons into layers creates **Multi-Layer Perceptrons (MLPs)** and deep neural networks. The same activation/loss flexibility carries over to the entire network.

### Are Perceptron and Logistic Regression the Same?

Many people say they are the same. Technically:

> **Perceptron = Logistic Regression IF AND ONLY IF:**
> - Activation function = Sigmoid
> - Loss function = Binary Cross Entropy

With different activation/loss choices, they are different algorithms — but built on the **same mathematical skeleton**.

---

## 17. Summary Table

| Activation Function | Loss Function | Algorithm | Output | Problem Type |
|---|---|---|---|---|
| Step function | Hinge Loss | Perceptron | $\{+1, -1\}$ | Binary classification |
| Sigmoid $\sigma(z)$ | Binary Cross Entropy | Logistic Regression | $P(y=1) \in (0,1)$ | Binary classification |
| Softmax | Categorical Cross Entropy | Softmax Regression | $P(\text{class}_j)$ for all $j$ | Multi-class classification |
| Linear (identity) | MSE | Linear Regression | Any $\hat{y} \in \mathbb{R}$ | Regression |

---

## 18. Key Takeaways

1. **The Perceptron Trick** is a useful heuristic but has two fundamental weaknesses: no quality measure and potential convergence issues.

2. **Loss functions** solve both problems — they quantify model quality as a single number and provide a differentiable objective for gradient descent to minimize.

3. **The Perceptron (Hinge) Loss** is:
   $$L = \frac{1}{N} \sum_{i} \max(0, -y_i z_i)$$
   Correctly classified points contribute **zero**; misclassified points contribute a **positive penalty proportional to their distance from the boundary**.

4. **The misclassification condition** is $y_i \cdot z_i < 0$ — when the prediction sign and true label sign differ.

5. **Gradient Descent update rules** for misclassified points:
   $$w_1 \leftarrow w_1 + \eta y_i x_{1i}, \quad w_2 \leftarrow w_2 + \eta y_i x_{2i}, \quad b \leftarrow b + \eta y_i$$

6. **Sigmoid** transforms $z \in (-\infty, +\infty)$ into a probability $\in (0,1)$. It is differentiable (unlike Step), enabling gradient-based training.

7. **Binary Cross Entropy** penalizes confidently wrong predictions logarithmically — very heavily. It is paired with Sigmoid for probabilistic binary classification (= Logistic Regression).

8. **Softmax + Categorical Cross Entropy** extends this to multi-class problems (= Softmax Regression).

9. **Linear activation + MSE** gives Linear Regression — the perceptron used for regression.

10. The **Perceptron is a universal and flexible framework** — not a single algorithm but a mathematical template. Swapping activation and loss functions yields entirely different, well-known algorithms.

11. **Next topic:** Why a single perceptron has limitations, and why **Multi-Layer Perceptrons (MLPs)** are needed.

---

## Appendix: Quick Reference Formulas

### Loss Functions

| Name | Formula |
|---|---|
| Hinge Loss | $\max(0, -y \cdot z)$ |
| Binary Cross Entropy | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| Categorical Cross Entropy | $-\sum_j y_j \log \hat{y}_j$ |
| MSE | $\frac{1}{N}\sum_i (\hat{y}_i - y_i)^2$ |

### Activation Functions

| Name | Formula | Range |
|---|---|---|
| Step | $\mathbf{1}[z \geq 0]$ → +1 else −1 | $\{-1, +1\}$ |
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $(0, 1)$ |
| Softmax | $\frac{e^{z_j}}{\sum_k e^{z_k}}$ | $(0,1)$, sums to 1 |
| Linear | $z$ | $(-\infty, +\infty)$ |

### Gradient Descent (General)

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}$$

where $\theta$ represents any parameter ($w_1, w_2, b$) and $\eta$ is the learning rate.

---