---
title: Maths for ML — Multivariable Calculus and Gradient
tags:
  - maths
  - machine-learning
  - calculus
  - gradient
  - backpropagation
date: 2026-08-30
---

# 08 — Multivariable Calculus and Gradient

> **Ek line mein:** Real ML models mein millions parameters hote hain. Gradient ek vector hai jo batata hai har parameter ko kis taraf move karna hai.

---

## 1. Multivariable function

$$f: \mathbb{R}^n \rightarrow \mathbb{R}, \quad z = f(x, y)$$

Example: $f(x,y) = x^2 + y^2$ → ek 3D bowl.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 60)
y = np.linspace(-3, 3, 60)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
ax1.set_title('Surface: f(x,y) = x² + y²')

ax2 = fig.add_subplot(122)
c = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.clabel(c, inline=True, fontsize=8)
ax2.set_title('Contour plot'); ax2.set_aspect('equal')
plt.tight_layout(); plt.show()
```

**ML mein:** Loss function $L(w_1, w_2, \dots, w_n)$ ek multivariable function hai. Hum uska minimum dhoondh rahe hain.

---

## 2. Partial Derivative ⭐⭐

$$\frac{\partial f}{\partial x} = \text{"} x \text{ badlo, baaki sab CONSTANT rakho"}$$

### Example
$f(x,y) = x^2y + 3xy^2 + 5$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \quad\text{(y ko constant maana)}$$
$$\frac{\partial f}{\partial y} = x^2 + 6xy \quad\text{(x ko constant maana)}$$

```python
import sympy as sp

x, y = sp.symbols('x y')
f = x**2*y + 3*x*y**2 + 5

sp.diff(f, x)      # 2*x*y + 3*y**2
sp.diff(f, y)      # x**2 + 6*x*y
```

**Geometric matlab:** Surface ko $y$ constant rakh kar slice karo — us slice ka slope hi $\partial f/\partial x$ hai.

---

## 3. Gradient ⭐⭐⭐

Saare partial derivatives ka **vector**:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### Do properties (ye yaad rakho) ⭐

1. **Gradient hamesha function ke sabse tez badhne ki direction mein point karta hai (steepest ascent)**
2. **Gradient ki magnitude batati hai kitni tezi se badh raha hai**

**Isiliye minimize karne ke liye hum $-\nabla f$ ki direction mein chalte hain:**
$$\mathbf{w}_{new} = \mathbf{w}_{old} - \eta\nabla L$$

Ye **gradient descent** hai.

```python
def gradient(f, point, h=1e-5):
    """Numerical gradient"""
    grad = np.zeros_like(point, dtype=float)
    for i in range(len(point)):
        p_plus = point.copy(); p_plus[i] += h
        p_minus = point.copy(); p_minus[i] -= h
        grad[i] = (f(p_plus) - f(p_minus)) / (2*h)
    return grad

f = lambda p: p[0]**2 + p[1]**2
gradient(f, np.array([3.0, 4.0]))     # [6., 8.]
```

### Gradient field ka visualization
```python
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
U, V = 2*X, 2*Y          # gradient = [2x, 2y]

plt.figure(figsize=(8,8))
plt.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
plt.quiver(X, Y, -U, -V, color='red', alpha=0.6)    # -gradient (descent)
plt.title('Negative gradient — minimum ki taraf ishara')
plt.gca().set_aspect('equal'); plt.show()
```

**Note:** Gradient hamesha contour lines ke **perpendicular** hota hai.

---

## 4. Directional Derivative

Kisi bhi direction $\hat{\mathbf{u}}$ mein rate of change:

$$D_\mathbf{u}f = \nabla f \cdot \hat{\mathbf{u}}$$

**Maximum tab hota hai jab $\hat{\mathbf{u}}$ gradient ki direction mein ho** ($\cos\theta = 1$) — yahi gradient ki property #1 ka proof hai.

---

## 5. Chain Rule — Multivariable ⭐⭐

### Case 1: $z = f(x,y)$, aur $x=g(t)$, $y=h(t)$
$$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}$$

### Case 2: Multiple paths (neural network jaisa)
$$\frac{\partial L}{\partial w} = \sum_{\text{paths}} \prod_{\text{edges in path}} \frac{\partial(\cdot)}{\partial(\cdot)}$$

**Rule:** Multiple paths hain toh **add** karo. Ek path ke andar **multiply** karo.

### Computation graph example
```
w → z = wx → a = σ(z) → L = (a-y)²
```

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w} = 2(a-y)\cdot\sigma'(z)\cdot x$$

---

## 6. Backpropagation ka maths ⭐⭐⭐

### Ek simple 2-layer network

**Forward pass:**
$$z_1 = W_1\mathbf{x} + \mathbf{b}_1, \quad a_1 = \sigma(z_1)$$
$$z_2 = W_2a_1 + \mathbf{b}_2, \quad \hat{y} = \sigma(z_2)$$
$$L = \frac{1}{2}(\hat{y}-y)^2$$

**Backward pass:**
$$\delta_2 = \frac{\partial L}{\partial z_2} = (\hat{y}-y)\odot\sigma'(z_2)$$
$$\frac{\partial L}{\partial W_2} = \delta_2 a_1^T, \quad \frac{\partial L}{\partial \mathbf{b}_2} = \delta_2$$

$$\delta_1 = (W_2^T\delta_2)\odot\sigma'(z_1)$$
$$\frac{\partial L}{\partial W_1} = \delta_1\mathbf{x}^T, \quad \frac{\partial L}{\partial \mathbf{b}_1} = \delta_1$$

### Scratch se implementation ⭐

```python
import numpy as np

def sigmoid(z): return 1/(1+np.exp(-z))
def sigmoid_grad(z):
    s = sigmoid(z); return s*(1-s)

class SimpleNN:
    def __init__(self, n_in, n_hidden, n_out, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.1, (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        self.X  = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, y, lr=0.1):
        m = len(y)
        # Output layer
        d2 = (self.a2 - y) * sigmoid_grad(self.z2)
        dW2 = self.a1.T @ d2 / m
        db2 = d2.mean(axis=0)
        # Hidden layer
        d1 = (d2 @ self.W2.T) * sigmoid_grad(self.z1)
        dW1 = self.X.T @ d1 / m
        db1 = d1.mean(axis=0)
        # Update
        self.W2 -= lr*dW2; self.b2 -= lr*db2
        self.W1 -= lr*dW1; self.b1 -= lr*db1

# XOR problem
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

nn = SimpleNN(2, 8, 1)
for epoch in range(20000):
    out = nn.forward(X)
    nn.backward(y, lr=0.5)
    if epoch % 5000 == 0:
        print(f"epoch {epoch}: loss = {np.mean((out-y)**2):.5f}")

print("\nPredictions:\n", np.round(nn.forward(X), 3))
```

---

## 7. Jacobian Matrix ⭐

Jab function **vector se vector** jaata hai: $\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$

$$J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}_{m\times n}$$

Har row ek output ka gradient hai.

```python
import torch

def f(x):
    return torch.stack([x[0]**2 + x[1], x[0]*x[1], torch.sin(x[0])])

x = torch.tensor([1.0, 2.0], requires_grad=True)
J = torch.autograd.functional.jacobian(f, x)
print(J)
```

**ML mein:** Backprop asal mein Jacobian-vector products ki chain hai. Softmax ki derivative bhi ek Jacobian hai.

---

## 8. Hessian Matrix ⭐

Saare **second-order** partial derivatives:

$$H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1\partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2\partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}$$

**Hessian hamesha symmetric hoti hai** (agar function smooth hai) — Clairaut's theorem.

### Hessian se critical points classify karna ⭐

| Hessian | Point ka type |
|---|---|
| Positive definite (saare $\lambda > 0$) | **Local minimum** |
| Negative definite (saare $\lambda < 0$) | **Local maximum** |
| Mixed signs | **Saddle point** ⚠️ |
| Kuch $\lambda = 0$ | Inconclusive |

```python
def classify_critical_point(H):
    eigvals = np.linalg.eigvals(H)
    if np.all(eigvals > 0):  return "Local Minimum"
    if np.all(eigvals < 0):  return "Local Maximum"
    if np.any(eigvals > 0) and np.any(eigvals < 0): return "Saddle Point"
    return "Inconclusive"

# f(x,y) = x² - y²  → saddle at origin
H = np.array([[2, 0], [0, -2]])
classify_critical_point(H)     # Saddle Point
```

**Deep learning ka bada insight:** High dimensions mein local minima kam problem hain — **saddle points** zyada common hain. Isiliye momentum-based optimizers (Adam, SGD+momentum) kaam aate hain — wo saddle points se nikal aate hain.

### Hessian kyun practically use nahi hoti?
$n$ parameters ke liye Hessian $n \times n$ hoti hai. GPT-scale model mein $n \sim 10^{11}$ → Hessian mein $10^{22}$ entries. Store karna namumkin.

**Isiliye:** first-order methods (SGD, Adam) use hote hain, ya quasi-Newton (L-BFGS) jo Hessian approximate karta hai.

---

## 9. Gradient of common ML functions ⭐

### Linear Regression
$$L = \frac{1}{n}\|X\mathbf{w}-\mathbf{y}\|^2$$
$$\nabla_\mathbf{w}L = \frac{2}{n}X^T(X\mathbf{w}-\mathbf{y})$$

```python
def linreg_gradient(X, y, w):
    n = len(y)
    return (2/n) * X.T @ (X @ w - y)
```

### Logistic Regression
$$L = -\frac{1}{n}\sum[y_i\log\hat{y_i} + (1-y_i)\log(1-\hat{y_i})], \quad \hat{y}=\sigma(X\mathbf{w})$$
$$\nabla_\mathbf{w}L = \frac{1}{n}X^T(\sigma(X\mathbf{w})-\mathbf{y})$$

```python
def logreg_gradient(X, y, w):
    n = len(y)
    return X.T @ (sigmoid(X @ w) - y) / n
```

**Dono ka form same hai!** $X^T(\text{prediction} - \text{actual})$ — ye coincidence nahi, generalized linear models ki property hai.

### Ridge (L2)
$$\nabla_\mathbf{w}L = \frac{2}{n}X^T(X\mathbf{w}-\mathbf{y}) + 2\lambda\mathbf{w}$$

### Lasso (L1)
$$\nabla_\mathbf{w}L = \frac{2}{n}X^T(X\mathbf{w}-\mathbf{y}) + \lambda\,\text{sign}(\mathbf{w})$$

$|w|$ zero pe differentiable nahi hai — isiliye **subgradient** ya coordinate descent use karte hain.

---

## 10. Matrix Calculus — useful identities ⭐

| Expression | Derivative w.r.t. $\mathbf{x}$ |
|---|---|
| $\mathbf{a}^T\mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T\mathbf{a}$ | $\mathbf{a}$ |
| $\mathbf{x}^T\mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^TA\mathbf{x}$ | $(A+A^T)\mathbf{x}$ (agar $A$ symmetric: $2A\mathbf{x}$) |
| $A\mathbf{x}$ | $A^T$ |
| $\|\mathbf{x}\|^2$ | $2\mathbf{x}$ |
| $\|A\mathbf{x}-\mathbf{b}\|^2$ | $2A^T(A\mathbf{x}-\mathbf{b})$ |

**Reference:** "The Matrix Cookbook" (free PDF) — poori list wahan hai.

---

## 11. Gradient Checking ⭐ (debugging ka tool)

Apne analytical gradient ko numerical se compare karo:

```python
def gradient_check(f, grad_f, x, eps=1e-7):
    analytical = grad_f(x)
    numerical = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        numerical[i] = (f(xp) - f(xm)) / (2*eps)

    diff = np.linalg.norm(analytical - numerical) / \
           (np.linalg.norm(analytical) + np.linalg.norm(numerical))
    print(f"Relative difference: {diff:.2e}")
    print("✅ Gradient sahi hai" if diff < 1e-7 else "❌ Bug hai gradient mein")
    return diff

f = lambda x: np.sum(x**2)
grad_f = lambda x: 2*x
gradient_check(f, grad_f, np.array([1.0, 2.0, 3.0]))
```

**Custom layer likhne ke baad hamesha ye chalao.**

---

## 12. Interview Questions

1. **Gradient kya batata hai?** → steepest ascent ki direction aur uski rate.
2. **Gradient descent mein minus sign kyun?** → gradient ascent ki direction batata hai, humein descent chahiye.
3. **Jacobian aur Hessian mein farq?** → Jacobian first-order (vector→vector), Hessian second-order (scalar function ka).
4. **Hessian deep learning mein kyun use nahi hoti?** → $O(n^2)$ memory, $O(n^3)$ inversion — millions parameters ke saath namumkin.
5. **Saddle point kya hai aur kyun problem hai?** → jahan gradient 0 hai lekin minimum nahi; high dimensions mein bahut common.
6. **Backpropagation ko chain rule se explain karo.**
7. **Gradient checking kya hai?** → analytical gradient ko numerical se verify karna, custom implementations debug karne ke liye.
8. **Lasso ka gradient kyun problematic hai?** → $|w|$ zero pe differentiable nahi.

---

## Aage padho
→ [[09 Optimization and Gradient Descent]]
← [[07 Functions and Derivatives]] | [[00 Maths for Machine Learning — MOC]]
