---
title: Maths for ML — Vectors
tags:
  - maths
  - machine-learning
  - linear-algebra
  - vectors
date: 2026-08-30
---

# 01 — Vectors

> **Ek line mein:** Vector ek aisi cheez hai jiske paas **magnitude (size)** aur **direction** dono hoti hai. ML mein har data point ek vector hota hai.

---

## 1. Vector kya hai?

### Teen nazariye

| Nazariya | Kya kehta hai |
|---|---|
| **Physics wala** | Ek arrow — jiski lambai magnitude aur jiska rukh direction |
| **Computer Science wala** | Numbers ki ek ordered list |
| **Mathematician wala** | Koi bhi cheez jo addition aur scalar multiplication ke rules follow kare |

**ML mein CS wala nazariya sabse zyada kaam aata hai.**

### ML mein vector ka matlab

Ek student ka data:
$$\mathbf{x} = \begin{bmatrix} 21 \\ 85 \\ 12 \end{bmatrix} \quad \text{(age, marks, package)}$$

Ye ek **3-dimensional feature vector** hai. Dataset ki har row ek vector hai.

```python
import numpy as np
x = np.array([21, 85, 12])
```

---

## 2. Row vs Column Vector

$$\text{Row: } \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}_{1\times3} \qquad \text{Column: } \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}_{3\times1}$$

**Convention:** Maths aur ML papers mein by default **column vector** maana jaata hai. Row vector ko $\mathbf{x}^T$ likhte hain.

```python
row = np.array([[1, 2, 3]])        # shape (1,3)
col = np.array([[1], [2], [3]])    # shape (3,1)
col = np.array([1,2,3]).reshape(-1, 1)   # aasan tareeqa
```

---

## 3. Zero Vector aur Unit Vector

**Zero vector:** $\mathbf{0} = [0, 0, ..., 0]$ — koi direction nahi, magnitude 0.

**Unit vector:** Jiski magnitude exactly 1 ho.
$$\hat{\mathbf{u}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

```python
v = np.array([3, 4])
unit_v = v / np.linalg.norm(v)     # [0.6, 0.8]
np.linalg.norm(unit_v)              # 1.0
```

**Kyun important?** Unit vector sirf **direction** batata hai, size hata deta hai. Cosine similarity mein yahi hota hai.

---

## 4. Magnitude / Norm ⭐

Vector ki "lambai".

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}$$

```python
v = np.array([3, 4])
np.linalg.norm(v)     # 5.0
```

### Norms ke types ⭐ (ML mein regularization)

| Norm | Formula | Naam | ML mein |
|---|---|---|---|
| **L1** | $\sum \|v_i\|$ | Manhattan / Taxicab | **Lasso** regularization |
| **L2** | $\sqrt{\sum v_i^2}$ | Euclidean | **Ridge** regularization |
| **L∞** | $\max \|v_i\|$ | Max norm | Adversarial robustness |
| **Lp** | $(\sum \|v_i\|^p)^{1/p}$ | General | — |

```python
v = np.array([3, -4, 5])
np.linalg.norm(v, 1)        # L1 = 12
np.linalg.norm(v, 2)        # L2 = 7.07
np.linalg.norm(v, np.inf)   # L∞ = 5
```

**Kyun L1 sparsity deta hai?** L1 ka "ball" corners wala hota hai (diamond shape), aur optimization ka solution aksar corner pe aata hai jahan koi coordinate exactly 0 hota hai. L2 ka ball gol hai, isliye values chhoti hoti hain lekin exactly 0 nahi.

---

## 5. Vector Addition & Subtraction

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}$$

**Geometric intuition:**
- **Addition** → *tip-to-tail*: `a` ke end se `b` shuru karo, origin se naye end tak ka arrow hi result hai (parallelogram law)
- **Subtraction** → `a - b` wo vector hai jo **b ke tip se a ke tip** tak jaata hai

```python
a = np.array([2, 3])
b = np.array([1, 1])
a + b     # [3, 4]
a - b     # [1, 2]
```

**Properties:**
- Commutative: $\mathbf{a}+\mathbf{b} = \mathbf{b}+\mathbf{a}$
- Associative: $(\mathbf{a}+\mathbf{b})+\mathbf{c} = \mathbf{a}+(\mathbf{b}+\mathbf{c})$

---

## 6. Scalar Multiplication

$$k\mathbf{v} = \begin{bmatrix} k v_1 \\ k v_2 \end{bmatrix}$$

- $k > 1$ → vector lamba ho jaata hai
- $0 < k < 1$ → chhota
- $k < 0$ → **direction ulti** ho jaati hai

```python
2 * a      # [4, 6]
-1 * a     # [-2, -3]
```

**Zaroori baat:** Scalar multiplication direction nahi badalta (sirf sign flip ke alawa) — sirf magnitude badalta hai.

---

## 7. Dot Product ⭐⭐ (sabse important operation)

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \dots$$

Ya geometrically:
$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$$

**Result ek scalar hota hai, vector nahi.**

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.dot(a, b)     # 32
a @ b            # 32  (naya syntax)
(a * b).sum()    # 32  (manually)
```

### Dot product ka sign kya batata hai? ⭐

| Value | Angle | Matlab |
|---|---|---|
| $> 0$ | $\theta < 90°$ | Same direction ki taraf jhukav |
| $= 0$ | $\theta = 90°$ | **Perpendicular / Orthogonal** |
| $< 0$ | $\theta > 90°$ | Opposite direction |

**Ye ML mein kaise use hota hai?** Logistic Regression aur SVM mein $\mathbf{w} \cdot \mathbf{x} + b$ ka sign hi decide karta hai ki point kis class mein hai.

### Angle nikalna

$$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

```python
def angle_between(a, b, degrees=True):
    cos_t = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_t = np.clip(cos_t, -1.0, 1.0)          # floating point safety
    theta = np.arccos(cos_t)
    return np.degrees(theta) if degrees else theta

angle_between(np.array([1,0]), np.array([0,1]))    # 90.0
```

### Cosine Similarity ⭐ (NLP mein bahut use hota hai)

$$\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$$

Range: $[-1, 1]$. 1 = bilkul same direction, 0 = unrelated, -1 = ulta.

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Kyun Euclidean distance ki jagah cosine?** Text documents mein lambai matter nahi karti — 10 page aur 100 page ka document same topic pe ho sakta hai. Cosine sirf **direction** dekhta hai, magnitude nahi.

### Dot product ki properties
- Commutative: $\mathbf{a}\cdot\mathbf{b} = \mathbf{b}\cdot\mathbf{a}$
- Distributive: $\mathbf{a}\cdot(\mathbf{b}+\mathbf{c}) = \mathbf{a}\cdot\mathbf{b} + \mathbf{a}\cdot\mathbf{c}$
- $\mathbf{a}\cdot\mathbf{a} = \|\mathbf{a}\|^2$
- **NOT associative** (kyunki result scalar hota hai)

---

## 8. Vector Projection ⭐

`a` ka `b` pe projection = `a` ka wo hissa jo `b` ki direction mein hai.

**Scalar projection (kitna):**
$$\text{comp}_\mathbf{b}\mathbf{a} = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|}$$

**Vector projection (kaunsa vector):**
$$\text{proj}_\mathbf{b}\mathbf{a} = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|^2}\mathbf{b}$$

```python
def projection(a, b):
    return (np.dot(a, b) / np.dot(b, b)) * b

a = np.array([3, 4])
b = np.array([1, 0])
projection(a, b)        # [3, 0]  — sirf x-component bacha
```

**ML mein kahan?**
- **PCA** — data ko principal components pe project karna
- **Linear Regression** — y ko column space pe project karna (least squares ka geometric matlab)
- **Gram-Schmidt** — orthogonal basis banana

---

## 9. Cross Product (sirf 3D mein)

$$\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1 \end{bmatrix}$$

**Result ek VECTOR hota hai** (dot product ke ulat), jo dono vectors ke **perpendicular** hota hai.

$$\|\mathbf{a} \times \mathbf{b}\| = \|\mathbf{a}\|\|\mathbf{b}\|\sin\theta$$

```python
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
np.cross(a, b)     # [0, 0, 1]
```

**Note:** ML mein cross product kam use hota hai (computer graphics/physics mein zyada). Lekin concept clear hona chahiye.

| | Dot Product | Cross Product |
|---|---|---|
| Result | Scalar | Vector |
| Formula | $\|a\|\|b\|\cos\theta$ | $\|a\|\|b\|\sin\theta$ |
| Dimensions | Kisi bhi n-D | Sirf 3D (aur 7D) |
| Commutative | Haan | Nahi ($a\times b = -b\times a$) |
| Zero jab | Perpendicular | Parallel |

---

## 10. Linear Combination, Span, Basis ⭐

### Linear Combination
$$c_1\mathbf{v_1} + c_2\mathbf{v_2} + \dots + c_n\mathbf{v_n}$$

### Span
Kuch vectors ka **span** = un sab points ka set jo unke linear combinations se ban sakte hain.
- Ek non-zero vector ka span → ek **line**
- Do independent vectors ka span (2D mein) → **poora plane**
- Agar do vectors parallel hain → span sirf ek line rahega

### Linear Independence
Vectors independent hain agar koi bhi vector baaki vectors ke combination se nahi ban sakta.

$$c_1\mathbf{v_1} + c_2\mathbf{v_2} + \dots = \mathbf{0} \implies \text{sab } c_i = 0$$

```python
# Check karo determinant se
A = np.array([[1, 2], [2, 4]])
np.linalg.det(A)          # 0 → dependent
np.linalg.matrix_rank(A)  # 1 → sirf 1 independent vector
```

### Basis
Vectors ka wo **minimum set** jo poore space ko span kar de aur linearly independent ho.

2D ka standard basis: $\hat{i}=[1,0]$, $\hat{j}=[0,1]$

**ML connection:** Feature space ka dimension = basis vectors ki count. PCA ek **naya basis** dhoondhta hai jismein data ki variance maximum ho.

---

## 11. Orthogonality & Orthonormality

- **Orthogonal:** $\mathbf{a}\cdot\mathbf{b}=0$ (perpendicular)
- **Orthonormal:** orthogonal + dono unit vectors

```python
a = np.array([1, 0])
b = np.array([0, 1])
np.dot(a, b) == 0      # True → orthogonal
```

**Kyun important?** Orthonormal basis ke saath calculations bahut simple ho jaate hain — projection sirf ek dot product ban jaata hai. PCA ke components hamesha orthonormal hote hain.

---

## 12. Distance between vectors

| Distance | Formula | Kab use |
|---|---|---|
| **Euclidean (L2)** | $\sqrt{\sum(a_i-b_i)^2}$ | Default, KNN, K-means |
| **Manhattan (L1)** | $\sum\|a_i-b_i\|$ | High dimensions, grid-like data |
| **Minkowski** | $(\sum\|a_i-b_i\|^p)^{1/p}$ | General form |
| **Cosine distance** | $1 - \text{cos\_sim}$ | Text, embeddings |

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.linalg.norm(a - b)          # Euclidean
np.abs(a - b).sum()            # Manhattan
```

---

## 13. Python mein sab kuch ek jagah

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Basic
a + b                       # addition
a - b                       # subtraction
3 * a                       # scalar multiplication
a * b                       # element-wise (Hadamard) — dot product NAHI

# Dot product
np.dot(a, b)
a @ b

# Norms
np.linalg.norm(a)           # L2
np.linalg.norm(a, 1)        # L1
np.linalg.norm(a, np.inf)   # L∞

# Unit vector
a / np.linalg.norm(a)

# Angle
np.degrees(np.arccos(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))))

# Cross product (3D)
np.cross(a, b)

# Projection
(np.dot(a,b) / np.dot(b,b)) * b

# Distance
np.linalg.norm(a - b)
```

---

## 14. Visualization ka code

```python
import matplotlib.pyplot as plt

def plot_vectors(vectors, colors=None, labels=None):
    plt.figure(figsize=(7,7))
    colors = colors or ['r','b','g','m']
    for i, v in enumerate(vectors):
        plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy',
                   scale=1, color=colors[i % len(colors)],
                   label=labels[i] if labels else None)
    lim = max(np.abs(np.array(vectors)).max() + 1, 2)
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
    plt.grid(alpha=0.3); plt.gca().set_aspect('equal')
    if labels: plt.legend()
    plt.show()

plot_vectors([[2,3], [1,-1], [3,2]], labels=['a','b','a+b'])
```

---

## 15. Interview Questions

1. **Dot product aur cross product mein farq?** → scalar vs vector result; cos vs sin; kisi bhi dimension vs sirf 3D.
2. **L1 aur L2 norm mein farq, aur Lasso vs Ridge?** → L1 sparsity deta hai (features exactly 0 ho jaate hain), L2 shrink karta hai lekin 0 nahi karta.
3. **Cosine similarity Euclidean distance se kab better hai?** → jab magnitude matter na kare, sirf direction/pattern matter kare (text documents, user preferences).
4. **Do vectors orthogonal hain — kaise check karo?** → dot product = 0.
5. **Linear independence kya hai aur kyun matter karti hai?** → dependent features multicollinearity paida karte hain, regression coefficients unstable ho jaate hain.
6. **Vector projection ML mein kahan use hota hai?** → PCA, least squares regression.

---

## Aage padho
→ [[02 Lines Planes and Hyperplanes]]
← [[00 Maths for Machine Learning — MOC]]
