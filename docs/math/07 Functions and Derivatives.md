---
title: Maths for ML — Functions and Derivatives
tags:
  - maths
  - machine-learning
  - calculus
  - derivatives
date: 2026-08-30
---

# 07 — Functions and Derivatives

> **Ek line mein:** Derivative batata hai ki function kis rate se badal raha hai. ML mein har training step ek derivative pe based hota hai.

---

## 1. Function kya hai?

Ek rule jo har input ko exactly ek output se jodta hai.
$$f: X \rightarrow Y, \quad y = f(x)$$

### ML mein
$$\hat{y} = f(\mathbf{x}; \theta)$$
- $\mathbf{x}$ = input features
- $\theta$ = parameters (weights) — **yahi seekhna hai**
- $\hat{y}$ = prediction

### Common functions ML mein ⭐

| Function | Formula | Kahan use |
|---|---|---|
| **Linear** | $f(x)=wx+b$ | Linear regression |
| **Sigmoid** | $\sigma(x)=\frac{1}{1+e^{-x}}$ | Binary classification |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | RNN, hidden layers |
| **ReLU** | $\max(0,x)$ | Deep networks (default) |
| **Leaky ReLU** | $\max(0.01x, x)$ | Dying ReLU fix |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | Multi-class output |
| **Log** | $\ln(x)$ | Log loss, log-likelihood |
| **Exponential** | $e^x$ | Growth, softmax |

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)
def leaky_relu(x, a=0.01): return np.where(x > 0, x, a*x)
def tanh(x): return np.tanh(x)
def softmax(x):
    e = np.exp(x - np.max(x))       # numerical stability
    return e / e.sum()

x = np.linspace(-5, 5, 200)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].plot(x, sigmoid(x)); ax[0,0].set_title('Sigmoid')
ax[0,1].plot(x, tanh(x));    ax[0,1].set_title('Tanh')
ax[1,0].plot(x, relu(x));    ax[1,0].set_title('ReLU')
ax[1,1].plot(x, leaky_relu(x)); ax[1,1].set_title('Leaky ReLU')
for a in ax.ravel(): a.grid(alpha=0.3); a.axhline(0,color='k',lw=0.5); a.axvline(0,color='k',lw=0.5)
plt.tight_layout(); plt.show()
```

---

## 2. Limits

$$\lim_{x \to a} f(x) = L$$

"Jab $x$, $a$ ke bahut kareeb jaata hai, toh $f(x)$ $L$ ke kareeb jaata hai."

### Important limits
$$\lim_{x\to0}\frac{\sin x}{x} = 1, \quad \lim_{x\to\infty}\left(1+\frac{1}{x}\right)^x = e, \quad \lim_{x\to0}\frac{e^x-1}{x}=1$$

### Continuity
$f$ continuous hai $x=a$ pe agar:
1. $f(a)$ defined hai
2. $\lim_{x\to a}f(x)$ exist karti hai
3. Dono barabar hain

**ML mein kyun matter?** Discontinuous functions differentiable nahi hote → gradient descent kaam nahi karega. Isiliye step function ki jagah sigmoid use karte hain.

---

## 3. Derivative ⭐⭐

$$f'(x) = \frac{df}{dx} = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$

### Do interpretations
1. **Geometric:** us point pe tangent line ka **slope**
2. **Physical:** **instantaneous rate of change**

### ML mein interpretation ⭐
$$\frac{\partial L}{\partial w} = \text{"agar main } w \text{ thoda badhaun toh loss kitna badlega?"}$$

- $\frac{\partial L}{\partial w} > 0$ → $w$ badhane se loss badhta hai → $w$ **kam** karo
- $\frac{\partial L}{\partial w} < 0$ → $w$ badhane se loss ghatta hai → $w$ **badhao**
- $\frac{\partial L}{\partial w} = 0$ → flat point (minimum/maximum/saddle)

**Yahi gradient descent ka poora idea hai.**

---

## 4. Differentiation Rules ⭐

| Rule | Formula |
|---|---|
| **Constant** | $\frac{d}{dx}(c) = 0$ |
| **Power** | $\frac{d}{dx}(x^n) = nx^{n-1}$ |
| **Constant multiple** | $\frac{d}{dx}(cf) = cf'$ |
| **Sum** | $(f+g)' = f' + g'$ |
| **Product** ⭐ | $(fg)' = f'g + fg'$ |
| **Quotient** | $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$ |
| **Chain** ⭐⭐ | $\frac{d}{dx}f(g(x)) = f'(g(x))\cdot g'(x)$ |

### Common derivatives

| $f(x)$ | $f'(x)$ |
|---|---|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $a^x$ | $a^x\ln a$ |
| $\ln x$ | $1/x$ |
| $\log_a x$ | $\frac{1}{x\ln a}$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x$ |
| $\sqrt{x}$ | $\frac{1}{2\sqrt{x}}$ |

---

## 5. Chain Rule ⭐⭐⭐ (Deep learning ki jaan)

$$\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}$$

### Example
$f(x) = (3x^2+1)^5$

Let $u = 3x^2+1$, $y = u^5$
$$\frac{dy}{dx} = 5u^4 \cdot 6x = 30x(3x^2+1)^4$$

### Long chain (deep network jaisa)
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_3}\cdot\frac{\partial a_3}{\partial z_3}\cdot\frac{\partial z_3}{\partial a_2}\cdot\frac{\partial a_2}{\partial z_2}\cdot\frac{\partial z_2}{\partial a_1}\cdot\frac{\partial a_1}{\partial z_1}\cdot\frac{\partial z_1}{\partial w_1}$$

**Yahi backpropagation hai** — bas chain rule ko efficiently apply karna.

### Vanishing Gradient ka maths ⭐
Sigmoid ki derivative ka max value **0.25** hai. Agar 10 layers hain:
$$0.25^{10} \approx 0.00000095$$

Gradient practically 0 ho jaata hai → early layers seekhte hi nahi. **Isiliye ReLU aaya** — uski derivative positive side pe exactly 1 hai.

```python
print(f"Sigmoid, 10 layers: {0.25**10:.2e}")     # 9.54e-07
print(f"ReLU, 10 layers:    {1.0**10:.2e}")      # 1.00e+00
```

---

## 6. Activation functions ki derivatives ⭐

| Function | $f(x)$ | $f'(x)$ |
|---|---|---|
| **Sigmoid** | $\sigma(x)$ | $\sigma(x)(1-\sigma(x))$ |
| **Tanh** | $\tanh(x)$ | $1-\tanh^2(x)$ |
| **ReLU** | $\max(0,x)$ | $1$ if $x>0$ else $0$ |
| **Leaky ReLU** | $\max(ax,x)$ | $1$ if $x>0$ else $a$ |

### Sigmoid derivative ka derivation ⭐
$$\sigma(x) = (1+e^{-x})^{-1}$$
$$\sigma'(x) = -(1+e^{-x})^{-2}\cdot(-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}$$
$$= \frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}} = \sigma(x)(1-\sigma(x))$$

**Bahut khoobsurat result** — derivative khud function ke terms mein hai, isliye computation sasta hai.

```python
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x): return (x > 0).astype(float)
def tanh_derivative(x): return 1 - np.tanh(x)**2

x = np.linspace(-6, 6, 200)
plt.plot(x, sigmoid(x), label='σ(x)')
plt.plot(x, sigmoid_derivative(x), label="σ'(x)")
plt.axhline(0.25, color='r', ls='--', label='max = 0.25')
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

---

## 7. Loss functions aur unki derivatives ⭐⭐

### Mean Squared Error (Regression)
$$L = \frac{1}{n}\sum(y_i - \hat{y_i})^2$$
$$\frac{\partial L}{\partial \hat{y}} = \frac{2}{n}(\hat{y}-y)$$

### Binary Cross Entropy (Classification)
$$L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$$
$$\frac{\partial L}{\partial\hat{y}} = \frac{\hat{y}-y}{\hat{y}(1-\hat{y})}$$

### Magic result ⭐
Sigmoid + BCE combine karo, toh chain rule ke baad:
$$\frac{\partial L}{\partial z} = \hat{y} - y$$

**Sab kuch cancel ho jaata hai!** Ye itna simple hai ki isiliye logistic regression mein sigmoid + BCE ki jodi standard hai.

**Derivation:**
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial\hat{y}}\cdot\frac{\partial\hat{y}}{\partial z} = \frac{\hat{y}-y}{\hat{y}(1-\hat{y})}\cdot\hat{y}(1-\hat{y}) = \hat{y}-y$$

```python
def mse(y, y_hat): return np.mean((y - y_hat)**2)
def mse_grad(y, y_hat): return 2*(y_hat - y)/len(y)

def bce(y, y_hat, eps=1e-15):
    y_hat = np.clip(y_hat, eps, 1-eps)
    return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
```

---

## 8. Higher Order Derivatives

$$f''(x) = \frac{d^2f}{dx^2}$$

| $f''(x)$ | Curve ka shape | Point ka type (jab $f'=0$) |
|---|---|---|
| $> 0$ | **Convex** (bowl ⌣) | Minimum |
| $< 0$ | **Concave** (dome ⌢) | Maximum |
| $= 0$ | Inflection point | Test fail — aur check karo |

**Ye "Second Derivative Test" hai.**

---

## 9. Numerical vs Symbolic vs Automatic Differentiation ⭐

### Numerical (approximation)
```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)     # central difference (better)

f = lambda x: x**3 + 2*x
print(numerical_derivative(f, 2))     # ≈ 14.0 (exact: 3(4)+2 = 14)
```
Simple lekin approximate aur slow (har parameter ke liye 2 evaluations).

### Symbolic (SymPy)
```python
import sympy as sp

x = sp.Symbol('x')
f = x**3 + 2*x
sp.diff(f, x)                # 3*x**2 + 2
sp.diff(f, x, 2)             # 6*x  (second derivative)

# Value nikalna
sp.diff(f, x).subs(x, 2)     # 14
```
Exact answer, lekin bade expressions pe slow ho jaata hai.

### Automatic Differentiation ⭐ (PyTorch/TensorFlow yahi use karte hain)
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x
y.backward()
print(x.grad)         # tensor(14.)
```

**Best of both worlds:** exact (numerical nahi), aur efficient (symbolic expression blow-up nahi hota).

---

## 10. Practical: Simple gradient descent ek variable mein

```python
def f(x): return x**2 - 4*x + 5       # minimum x=2 pe
def df(x): return 2*x - 4

x = 10.0
lr = 0.1
history = [x]

for i in range(50):
    x = x - lr * df(x)
    history.append(x)

print(f"Minimum at x = {x:.4f}, f(x) = {f(x):.4f}")

xs = np.linspace(-2, 11, 200)
plt.plot(xs, f(xs), label='f(x)')
plt.scatter(history, [f(h) for h in history], c=range(len(history)),
            cmap='autumn', s=30, zorder=5)
plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend(); plt.grid(alpha=0.3)
plt.title('Gradient Descent ka safar')
plt.show()
```

---

## 11. Interview Questions

1. **Derivative ka ML mein kya matlab hai?** → parameter thoda badalne se loss kitna badlega.
2. **Chain rule backpropagation mein kaise use hota hai?** → output layer se input tak gradients ko multiply karte hue le jaana.
3. **Vanishing gradient problem kya hai?** → sigmoid ki derivative ≤ 0.25, deep networks mein multiply hoke ~0 ho jaati hai.
4. **ReLU sigmoid se better kyun hai?** → derivative 1 hai (vanishing gradient nahi), computationally sasta, sparsity deta hai.
5. **Sigmoid + BCE ki derivative itni simple kyun hai?** → chain rule mein $\hat{y}(1-\hat{y})$ cancel ho jaata hai, bacha $\hat{y}-y$.
6. **ReLU $x=0$ pe differentiable nahi hai — problem?** → practically nahi; frameworks 0 ya 1 assign kar dete hain, exact 0 milne ka chance na ke barabar hai.
7. **Numerical differentiation production mein kyun nahi use karte?** → approximate + har parameter ke liye alag evaluations = bahut slow. Autodiff use hota hai.

---

## Aage padho
→ [[08 Multivariable Calculus and Gradient]]
← [[06 Eigenvalues Eigenvectors PCA SVD]] | [[00 Maths for Machine Learning — MOC]]
