---
title: Maths for ML — Optimization and Gradient Descent
tags:
  - maths
  - machine-learning
  - optimization
  - gradient-descent
date: 2026-08-30
---

# 09 — Optimization and Gradient Descent

> **Ek line mein:** ML training asal mein ek optimization problem hai — loss function ka minimum dhoondhna.

---

## 1. Optimization ka problem statement

$$\theta^* = \arg\min_\theta L(\theta)$$

"Wo parameters dhoondo jahan loss sabse kam ho."

---

## 2. Maxima, Minima, Critical Points

### Critical point
Jahan $\nabla f = 0$ (ya derivative exist na kare).

### Types

| Type | Condition |
|---|---|
| **Local minimum** | Aaspaas mein sabse kam |
| **Global minimum** | Poore domain mein sabse kam ⭐ |
| **Local maximum** | Aaspaas mein sabse zyada |
| **Saddle point** | Ek direction mein min, doosri mein max |
| **Plateau** | Gradient bahut chhota, flat region |

### Second derivative test (1D)
$f'(a)=0$ pe:
- $f''(a) > 0$ → minimum
- $f''(a) < 0$ → maximum
- $f''(a) = 0$ → inconclusive

### Multivariable — Hessian test
Dekho [[08 Multivariable Calculus and Gradient]] section 8.

---

## 3. Convexity ⭐⭐

### Convex function
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

**Simple matlab:** Curve pe koi bhi do points jodo — wo line hamesha curve ke **upar** rahegi.

**Check:** $f''(x) \geq 0$ (1D) ya Hessian positive semi-definite (nD).

### Kyun ye sabse important property hai? ⭐
**Convex function mein har local minimum global minimum hota hai.** Gradient descent guaranteed sahi jawab pe pahunchega.

| Model | Convex? |
|---|---|
| Linear Regression (MSE) | ✅ Haan |
| Logistic Regression (BCE) | ✅ Haan |
| SVM (hinge loss) | ✅ Haan |
| Ridge / Lasso | ✅ Haan |
| **Neural Networks** | ❌ **Nahi** — non-convex |
| K-Means | ❌ Nahi |

**Deep learning non-convex hai** — isiliye initialization, learning rate, aur optimizer ka choice itna matter karta hai. Lekin practice mein zyadatar local minima kaafi acche hote hain.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)
convex = x**2
nonconvex = x**4 - 3*x**2 + x

fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].plot(x, convex); ax[0].set_title('Convex ✅ — ek hi minimum')
ax[1].plot(x, nonconvex); ax[1].set_title('Non-convex ❌ — multiple minima')
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 4. Gradient Descent ⭐⭐⭐

### Algorithm
$$\theta_{t+1} = \theta_t - \eta\nabla L(\theta_t)$$

- $\eta$ = **learning rate** (step size)
- $\nabla L$ = gradient

### Steps
1. Parameters ko randomly initialize karo
2. Gradient calculate karo
3. Gradient ke **ulti** direction mein step lo
4. Convergence tak repeat karo

```python
def gradient_descent(grad_f, x0, lr=0.1, n_iter=100, tol=1e-6):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for i in range(n_iter):
        g = grad_f(x)
        x_new = x - lr * g
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged at iteration {i}")
            break
        x = x_new
        history.append(x.copy())
    return x, np.array(history)

grad = lambda p: np.array([2*p[0], 2*p[1]])
x_min, hist = gradient_descent(grad, [5.0, 4.0], lr=0.1, n_iter=100)
print("Minimum:", x_min)
```

### Visualization
```python
X, Y = np.meshgrid(np.linspace(-6,6,100), np.linspace(-6,6,100))
Z = X**2 + Y**2

plt.figure(figsize=(8,8))
plt.contour(X, Y, Z, levels=25, cmap='viridis', alpha=0.6)
plt.plot(hist[:,0], hist[:,1], 'ro-', ms=4, lw=1)
plt.scatter(0, 0, marker='*', s=300, c='gold', edgecolor='k', zorder=5)
plt.title('Gradient Descent ka path'); plt.gca().set_aspect('equal')
plt.show()
```

---

## 5. Learning Rate ⭐⭐

Sabse important hyperparameter.

| $\eta$ | Kya hoga |
|---|---|
| Bahut chhota | Convergence bahut slow ⏳ |
| Sahi | Smooth aur fast convergence ✅ |
| Bahut bada | Oscillation, ya diverge ho jayega 💥 |

```python
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, lr in zip(ax, [0.01, 0.1, 0.5, 1.01]):
    _, h = gradient_descent(lambda p: np.array([2*p[0], 2*p[1]]),
                            [5.0, 4.0], lr=lr, n_iter=30)
    a.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
    a.plot(h[:,0], h[:,1], 'ro-', ms=3)
    a.set_title(f'lr = {lr}'); a.set_xlim(-8,8); a.set_ylim(-8,8)
plt.tight_layout(); plt.show()
```

### Learning Rate Scheduling ⭐

```python
# Step decay
def step_decay(lr0, epoch, drop=0.5, epochs_drop=10):
    return lr0 * (drop ** (epoch // epochs_drop))

# Exponential decay
def exp_decay(lr0, epoch, k=0.01):
    return lr0 * np.exp(-k * epoch)

# Cosine annealing (modern favourite)
def cosine_annealing(lr0, epoch, total):
    return lr0 * 0.5 * (1 + np.cos(np.pi * epoch / total))

# 1/t decay
def inverse_decay(lr0, epoch, k=0.01):
    return lr0 / (1 + k*epoch)
```

**Warmup:** Shuru mein learning rate dheere badhao, phir decay karo. Transformers ki training mein standard.

---

## 6. Gradient Descent ke Variants ⭐⭐

| Variant | Har step mein kitna data | Speed | Stability |
|---|---|---|---|
| **Batch GD** | Poora dataset | Slow | Smooth ✅ |
| **Stochastic GD (SGD)** | 1 sample | Fast | Bahut noisy |
| **Mini-batch GD** ⭐ | 32/64/128 samples | Balanced | Balanced ✅ |

**Practice mein hamesha mini-batch use hota hai.**

```python
def minibatch_gd(X, y, grad_fn, w0, lr=0.01, batch_size=32, epochs=100):
    w = w0.copy()
    n = len(y)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            w -= lr * grad_fn(X[b], y[b], w)
    return w
```

### SGD ka chhupa hua fayda
Noise actually **help** karta hai — network ko shallow local minima se bahar nikaal deta hai, aur regularization ka kaam bhi karta hai.

---

## 7. Advanced Optimizers ⭐⭐

### Momentum
Physics ka concept — ball ko "momentum" do.
$$v_t = \beta v_{t-1} + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

Typical $\beta = 0.9$. Oscillations kam karta hai, ravines mein tez chalata hai.

### Nesterov Accelerated Gradient (NAG)
Pehle "look ahead" karo, phir gradient nikaalo — thoda smarter momentum.

### AdaGrad
Har parameter ki apni learning rate. Jo parameter zyada update hua, uska LR kam kar do.
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t+\epsilon}}\odot\nabla L$$

**Problem:** $G_t$ hamesha badhta hai → LR eventually 0 ho jaata hai.

### RMSProp
AdaGrad ka fix — exponential moving average use karta hai.
$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2$$

### Adam ⭐⭐ (default choice)
Momentum + RMSProp ka combination.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{(first moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(second moment)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 0.001$

```python
class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grads
        self.v = self.b2*self.v + (1-self.b2)*grads**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### AdamW
Adam + **decoupled** weight decay. Modern transformers ka default.

### Optimizer choose kaise karein?

| Situation | Optimizer |
|---|---|
| Default / kuch pata nahi | **Adam / AdamW** |
| Computer vision (CNN) | SGD + momentum (better generalization) |
| NLP / Transformers | AdamW + warmup |
| Sparse data | AdaGrad |
| Chhota model, convex loss | Batch GD ya L-BFGS |

---

## 8. Constrained Optimization — Lagrange Multipliers ⭐

Problem: $f(x)$ minimize karo **subject to** $g(x)=0$.

**Lagrangian:**
$$\mathcal{L}(x, \lambda) = f(x) - \lambda g(x)$$

Solve: $\nabla_x\mathcal{L}=0$ aur $\nabla_\lambda\mathcal{L}=0$

**ML mein kahan?**
- **SVM** — margin maximize karo constraints ke saath (dual formulation)
- **PCA** — variance maximize karo $\|w\|=1$ ke constraint ke saath
- **Regularization** — constrained problem ka penalized version hai

### KKT Conditions
Inequality constraints ke liye Lagrange ka extension. SVM ki dual derivation ka base.

---

## 9. Regularization ka optimization view ⭐

$$L_{total} = L_{data} + \lambda R(\theta)$$

| Type | $R(\theta)$ | Effect |
|---|---|---|
| **L2 (Ridge)** | $\|\theta\|_2^2$ | Weights shrink karte hain, zero nahi hote |
| **L1 (Lasso)** | $\|\theta\|_1$ | **Sparsity** — kuch weights exactly 0 |
| **Elastic Net** | $\alpha\|\theta\|_1 + (1-\alpha)\|\theta\|_2^2$ | Dono ka mix |

**L1 sparsity kyun deta hai?** L1 ka constraint region diamond-shaped hai (corners hain). Optimization ka solution aksar corner pe milta hai, jahan koi coordinate exactly 0 hota hai. L2 ka region circle hai — koi corner nahi.

```python
def ridge_gd(X, y, lr=0.01, lam=0.1, epochs=1000):
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(epochs):
        grad = (2/n) * X.T @ (X @ w - y) + 2*lam*w
        w -= lr * grad
    return w
```

---

## 10. Convergence Criteria

Kab rukein?

```python
# 1. Gradient chhota ho gaya
if np.linalg.norm(grad) < 1e-6: break

# 2. Parameters barely badal rahe
if np.linalg.norm(w_new - w_old) < 1e-8: break

# 3. Loss barely badal raha
if abs(loss_new - loss_old) < 1e-8: break

# 4. Max iterations
if epoch >= max_epochs: break

# 5. Early stopping (validation loss badhne lage) ⭐ best
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= patience: break
```

---

## 11. Complete Example — Linear Regression scratch se ⭐

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = 2 * np.random.rand(200, 1)
y = 4 + 3*X.ravel() + np.random.randn(200)
X_b = np.c_[np.ones((200,1)), X]

def train(X, y, lr=0.1, epochs=200):
    w = np.zeros(X.shape[1])
    n = len(y)
    losses = []
    for _ in range(epochs):
        pred = X @ w
        loss = np.mean((pred - y)**2)
        grad = (2/n) * X.T @ (pred - y)
        w -= lr * grad
        losses.append(loss)
    return w, losses

w, losses = train(X_b, y)
print("GD solution      :", w)
print("Closed-form      :", np.linalg.lstsq(X_b, y, rcond=None)[0])

fig, ax = plt.subplots(1, 2, figsize=(13,5))
ax[0].plot(losses); ax[0].set_title('Loss curve')
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('MSE'); ax[0].grid(alpha=0.3)
ax[1].scatter(X, y, alpha=0.5)
ax[1].plot(X, X_b @ w, 'r-', lw=2, label='Fitted line')
ax[1].legend(); ax[1].set_title('Fit')
plt.tight_layout(); plt.show()
```

---

## 12. Common Problems & Fixes ⚠️

| Problem | Symptom | Fix |
|---|---|---|
| **Exploding gradients** | Loss = NaN/inf | Gradient clipping, LR kam karo, better init |
| **Vanishing gradients** | Early layers seekhte nahi | ReLU, BatchNorm, residual connections |
| **Slow convergence** | Loss dheere ghat raha | LR badhao, Adam use karo, features scale karo |
| **Oscillation** | Loss upar-neeche | LR kam karo, momentum add karo |
| **Local minima / saddle** | Loss stuck | Momentum, random restarts, better init |
| **Overfitting** | Train loss ↓ val loss ↑ | Regularization, dropout, early stopping |
| **Feature scaling issue** | Elongated contours, slow | StandardScaler use karo ⭐ |

### Gradient clipping
```python
def clip_gradients(grads, max_norm=1.0):
    norm = np.linalg.norm(grads)
    return grads * (max_norm / norm) if norm > max_norm else grads
```

### Feature scaling kyun zaroori hai ⭐
Agar ek feature 0–1 range mein hai aur doosra 0–100000, toh loss surface bahut elongated ho jaati hai — gradient descent zigzag karta hua bahut slow chalega. Scaling se contours gol ho jaate hain aur convergence tez.

---

## 13. Interview Questions

1. **Gradient descent kaise kaam karta hai?**
2. **Batch, SGD, aur mini-batch mein farq?**
3. **Learning rate bahut bada ho toh kya hoga?** → oscillation ya divergence.
4. **Adam SGD se better kyun mana jaata hai?** → adaptive per-parameter LR + momentum, kam tuning chahiye.
5. **Convex function kya hai aur ML mein kyun matter karta hai?** → local min = global min, convergence guaranteed.
6. **Neural networks non-convex hain — phir bhi kaam kyun karte hain?** → high dimensions mein zyadatar local minima quality mein similar hote hain; saddle points se momentum nikaal deta hai.
7. **Momentum kya karta hai?** → previous gradients ka moving average, oscillations kam karta hai aur convergence tez.
8. **Gradient clipping kab use karte hain?** → RNN/transformers mein exploding gradients rokne ke liye.
9. **L1 sparsity kyun deta hai, L2 nahi?** → L1 constraint region ke corners hote hain.
10. **Early stopping regularization kaise hai?** → model ko overfit hone se pehle rok deta hai, effective capacity limit karta hai.

---

## Aage padho
→ [[10 Descriptive Statistics]]
← [[08 Multivariable Calculus and Gradient]] | [[00 Maths for Machine Learning — MOC]]
