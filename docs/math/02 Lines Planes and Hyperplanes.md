---
title: Maths for ML — Lines, Planes and Hyperplanes
tags:
  - maths
  - machine-learning
  - linear-algebra
  - geometry
date: 2026-08-30
---

# 02 — Lines, Planes and Hyperplanes

> **Ek line mein:** Har classification algorithm asal mein ek **hyperplane** dhoondh raha hota hai jo classes ko alag kar de.

---

## 1. 2D mein Line

### School wala formula
$$y = mx + c$$
- $m$ = slope
- $c$ = y-intercept

### General form ⭐ (ML mein yahi use hota hai)
$$Ax + By + C = 0$$

**Kyun general form better hai?** Vertical line ($x = 5$) ko $y=mx+c$ se likh hi nahi sakte (slope infinite ho jaata hai). General form har line handle kar leta hai.

### ML notation
$$w_1x_1 + w_2x_2 + b = 0$$

Ya vector notation mein:
$$\mathbf{w}^T\mathbf{x} + b = 0$$

Yahan $\mathbf{w} = [w_1, w_2]$ **weight vector** hai aur $b$ **bias**.

---

## 2. Weight vector `w` ka geometric matlab ⭐⭐

**Sabse important insight:** $\mathbf{w}$ hamesha line/plane ke **perpendicular (normal)** hota hai.

**Proof intuition:** Agar $\mathbf{x_1}$ aur $\mathbf{x_2}$ dono line pe hain:
$$\mathbf{w}^T\mathbf{x_1} + b = 0, \quad \mathbf{w}^T\mathbf{x_2} + b = 0$$
Subtract karo:
$$\mathbf{w}^T(\mathbf{x_1} - \mathbf{x_2}) = 0$$

$(\mathbf{x_1}-\mathbf{x_2})$ line ke **along** hai, aur uska $\mathbf{w}$ ke saath dot product 0 hai → **$\mathbf{w}$ perpendicular hai**. ✅

```python
import numpy as np
import matplotlib.pyplot as plt

w = np.array([2, 3])
b = -6

x = np.linspace(-2, 5, 100)
y = (-w[0]*x - b) / w[1]          # w1*x + w2*y + b = 0 se y nikalo

plt.plot(x, y, label='w·x + b = 0')
plt.quiver(0, 2, w[0], w[1], angles='xy', scale_units='xy',
           scale=1, color='red', label='w (normal)')
plt.grid(alpha=0.3); plt.legend(); plt.gca().set_aspect('equal')
plt.show()
```

---

## 3. Point line ke kis taraf hai? ⭐⭐ (Classification ka core)

Kisi point $\mathbf{x_0}$ ko equation mein daal do:

$$f(\mathbf{x_0}) = \mathbf{w}^T\mathbf{x_0} + b$$

| Value | Matlab |
|---|---|
| $f(\mathbf{x_0}) > 0$ | Point $\mathbf{w}$ ki taraf (positive side) |
| $f(\mathbf{x_0}) = 0$ | Point line PE hai |
| $f(\mathbf{x_0}) < 0$ | Point ulti taraf (negative side) |

**Yahi Perceptron, SVM, aur Logistic Regression ka basic idea hai:**
$$\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b)$$

```python
def side(w, b, point):
    val = np.dot(w, point) + b
    return 'positive' if val > 0 else ('negative' if val < 0 else 'on the line')

w = np.array([2, 3]); b = -6
side(w, b, np.array([3, 3]))     # positive
side(w, b, np.array([0, 0]))     # negative
```

---

## 4. Point se line ki distance ⭐⭐

$$d = \frac{|\mathbf{w}^T\mathbf{x_0} + b|}{\|\mathbf{w}\|}$$

2D mein: $d = \dfrac{|Ax_0 + By_0 + C|}{\sqrt{A^2+B^2}}$

```python
def distance_to_hyperplane(w, b, point):
    return abs(np.dot(w, point) + b) / np.linalg.norm(w)

distance_to_hyperplane(np.array([2,3]), -6, np.array([3,3]))
```

### **Signed distance** (sign ke saath, absolute value nahi):
$$d_{signed} = \frac{\mathbf{w}^T\mathbf{x_0}+b}{\|\mathbf{w}\|}$$

Ye batata hai kitna door **aur** kis taraf.

**SVM mein iska role:** SVM aisa hyperplane dhoondhta hai jahan closest points ki distance (**margin**) maximum ho:
$$\text{margin} = \frac{2}{\|\mathbf{w}\|}$$
Margin maximize karna = $\|\mathbf{w}\|$ minimize karna. Isiliye SVM ka objective mein $\frac{1}{2}\|\mathbf{w}\|^2$ hota hai.

---

## 5. 3D mein Plane

$$Ax + By + Cz + D = 0$$

Ya: $\mathbf{w}^T\mathbf{x} + b = 0$ jahan $\mathbf{w}=[A,B,C]$ normal vector hai.

```python
from mpl_toolkits.mplot3d import Axes3D

w = np.array([1, 1, 1]); b = -3
xx, yy = np.meshgrid(np.linspace(-2,4,20), np.linspace(-2,4,20))
zz = (-w[0]*xx - w[1]*yy - b) / w[2]

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='viridis')
ax.quiver(1,1,1, w[0], w[1], w[2], color='red', length=1)
plt.show()
```

---

## 6. Hyperplane — n dimensions mein ⭐

$$w_1x_1 + w_2x_2 + \dots + w_nx_n + b = 0$$
$$\boxed{\mathbf{w}^T\mathbf{x} + b = 0}$$

**Key rule:** n-dimensional space mein hyperplane ka dimension **(n−1)** hota hai.

| Space | Hyperplane |
|---|---|
| 1D (line) | Point (0D) |
| 2D (plane) | Line (1D) |
| 3D | Plane (2D) |
| 4D | 3D "volume" |
| nD | (n−1)-D hyperplane |

**Ye kyun matter karta hai?** Agar tumhare paas 100 features hain, toh tumhara data 100-D space mein hai, aur classifier ek 99-D hyperplane dhoondh raha hai. Visualize nahi kar sakte, lekin maths bilkul wahi hai.

---

## 7. Homogeneous coordinates (bias trick) ⭐

Bias $b$ ko alag rakhne ki jagah $\mathbf{w}$ mein hi mila do:

$$\tilde{\mathbf{w}} = [b, w_1, w_2, \dots, w_n], \quad \tilde{\mathbf{x}} = [1, x_1, x_2, \dots, x_n]$$

Ab equation simple ho gayi:
$$\tilde{\mathbf{w}}^T\tilde{\mathbf{x}} = 0$$

```python
X = np.array([[2, 3], [4, 5], [6, 7]])
X_aug = np.hstack([np.ones((X.shape[0], 1)), X])     # bias column add
```

**Ye trick har ML library andar se use karti hai** — code simple ho jaata hai, sab kuch ek matrix multiplication ban jaata hai.

---

## 8. ML Connection — Decision Boundaries

### Perceptron
$$\hat{y} = \text{sign}(\mathbf{w}^T\mathbf{x} + b)$$
Simple: kis taraf ho, wahi class.

### Logistic Regression
$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x}+b) = \frac{1}{1+e^{-(\mathbf{w}^T\mathbf{x}+b)}}$$
Sign ki jagah **probability** deta hai. Decision boundary wahi hyperplane hai jahan $\mathbf{w}^T\mathbf{x}+b = 0$ (yaani probability = 0.5).

### SVM
$$\min \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x_i}+b) \geq 1$$
Sirf koi bhi separating hyperplane nahi — **maximum margin** wala.

### Linear Regression
$$\hat{y} = \mathbf{w}^T\mathbf{x} + b$$
Yahan hyperplane classify nahi karta, **fit** karta hai.

---

## 9. Practical Example — Decision boundary plot karna

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
model = LogisticRegression().fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]

x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
y_vals = -(w[0]*x_vals + b) / w[1]

plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolor='k')
plt.plot(x_vals, y_vals, 'k--', lw=2, label='Decision Boundary')
plt.legend(); plt.show()

print("w =", w, " b =", b)
print("Distance of first point:", abs(np.dot(w, X[0]) + b) / np.linalg.norm(w))
```

---

## 10. Linearly Separable vs Non-separable

**Linearly separable:** ek straight hyperplane classes ko poori tarah alag kar deta hai.

**Non-separable (XOR problem):** koi straight line kaam nahi karti.

**Solutions:**
1. **Kernel trick (SVM)** — data ko higher dimension mein bhejo jahan wo separable ho jaaye
2. **Neural networks** — multiple layers se non-linear boundary banao
3. **Soft margin** — kuch galtiyan allow karo

```python
# XOR — kabhi bhi linearly separable nahi
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])
```

---

## 11. Formulas Summary

| Cheez | Formula |
|---|---|
| Line (2D general) | $Ax+By+C=0$ |
| Hyperplane (nD) | $\mathbf{w}^T\mathbf{x}+b=0$ |
| Normal vector | $\mathbf{w}$ |
| Point ka side | $\text{sign}(\mathbf{w}^T\mathbf{x_0}+b)$ |
| Distance | $\dfrac{\|\mathbf{w}^T\mathbf{x_0}+b\|}{\|\mathbf{w}\|}$ |
| SVM margin | $\dfrac{2}{\|\mathbf{w}\|}$ |
| Hyperplane dimension | $n-1$ |

---

## 12. Interview Questions

1. **Hyperplane kya hai aur ML mein kyun matter karta hai?**
2. **`w` vector ka geometric matlab kya hai?** → hyperplane ka normal.
3. **SVM margin kyun maximize karta hai?** → better generalization, unseen data pe zyada robust.
4. **`||w||` minimize karne aur margin maximize karne ka kya relation hai?** → margin $= 2/\|w\|$, toh ek ko maximize karna doosre ko minimize karna hai.
5. **Bias term kya karta hai?** → hyperplane ko origin se shift karta hai. Bias ke bina hyperplane hamesha origin se guzarega.
6. **XOR linearly separable kyun nahi?**

---

## Aage padho
→ [[03 Matrices]]
← [[01 Vectors]] | [[00 Maths for Machine Learning — MOC]]
