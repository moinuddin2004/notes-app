---
title: Maths for ML — Determinant, Inverse and Rank
tags:
  - maths
  - machine-learning
  - linear-algebra
  - determinant
  - inverse
date: 2026-08-30
---

# 04 — Determinant, Inverse and Rank

> **Ek line mein:** Determinant batata hai ki matrix space ko kitna "stretch" karta hai; agar 0 hai toh matrix ne space ko squash kar diya aur information kho gayi.

---

## 1. Determinant ⭐⭐

### 2×2 ke liye
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

### 3×3 ke liye (cofactor expansion)
$$\det(A) = a_{11}M_{11} - a_{12}M_{12} + a_{13}M_{13}$$
jahan $M_{ij}$ minor hai (row i aur column j hata ke bacha determinant).

```python
import numpy as np
A = np.array([[3, 8], [4, 6]])
np.linalg.det(A)          # -14.0
```

---

## 2. Determinant ka **geometric matlab** ⭐⭐⭐

Ye samajh lo toh sab clear ho jayega.

**Determinant = area/volume ka scaling factor.**

- 2D mein: unit square ($1\times1$) transformation ke baad kitne area ka parallelogram banega
- 3D mein: unit cube kitne volume ka parallelepiped banega

| $\det(A)$ | Kya hua |
|---|---|
| $= 2$ | Area 2 guna ho gaya |
| $= 0.5$ | Area aadha ho gaya |
| $= 0$ | **Space collapse ho gaya** (line ya point mein) ⚠️ |
| $< 0$ | Orientation **flip** ho gayi (mirror image) |
| $= 1$ | Area same (rotation jaisa) |

```python
# Scaling matrix
A = np.array([[2, 0], [0, 3]])
np.linalg.det(A)         # 6.0 → area 6 guna

# Rotation matrix
theta = np.pi/4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
np.linalg.det(R)         # 1.0 → area same

# Singular matrix
S = np.array([[1, 2], [2, 4]])
np.linalg.det(S)         # 0.0 → collapse!
```

---

## 3. Determinant ki Properties

| Property | Formula |
|---|---|
| Transpose | $\det(A^T) = \det(A)$ |
| Product | $\det(AB) = \det(A)\det(B)$ ⭐ |
| Inverse | $\det(A^{-1}) = 1/\det(A)$ |
| Scalar | $\det(kA) = k^n\det(A)$ (n = size) |
| Identity | $\det(I) = 1$ |
| Triangular | Diagonal elements ka product |
| Two rows same | $\det = 0$ |
| Row swap | Sign flip |
| Eigenvalues | $\det(A) = \prod \lambda_i$ ⭐ |

---

## 4. Singular vs Non-singular

| | Non-singular (Invertible) | Singular |
|---|---|---|
| $\det(A)$ | $\neq 0$ | $= 0$ |
| Inverse | Exist karta hai | Nahi hota |
| Rank | Full rank ($=n$) | $< n$ |
| Rows/Columns | Linearly independent | Dependent |
| $Ax=b$ | Ek unique solution | 0 ya infinite solutions |

**ML mein singular matrix ka matlab:** Tumhare features mein **multicollinearity** hai — koi feature doosre features ka linear combination hai. Linear Regression ka closed-form solution fail ho jayega.

```python
# Multicollinearity example
X = np.array([[1, 2, 3],
              [2, 4, 6],       # row 2 = 2 × row 1
              [3, 6, 9]])
np.linalg.det(X)               # ~0
np.linalg.matrix_rank(X)       # 1 (3 nahi!)
```

**Fix:** Ridge regression ($\lambda I$ add karna) matrix ko invertible bana deta hai. Yahi Ridge ka asli maths hai:
$$\mathbf{w} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$$

---

## 5. Inverse Matrix ⭐

$$AA^{-1} = A^{-1}A = I$$

### 2×2 ka formula
$$A^{-1} = \frac{1}{\det(A)}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

### General formula
$$A^{-1} = \frac{1}{\det(A)}\text{adj}(A)$$
jahan $\text{adj}(A)$ = adjugate = cofactor matrix ka transpose.

```python
A = np.array([[3, 8], [4, 6]])
A_inv = np.linalg.inv(A)
np.allclose(A @ A_inv, np.eye(2))     # True
```

### Properties
- $(A^{-1})^{-1} = A$
- $(AB)^{-1} = B^{-1}A^{-1}$ ⭐ (order ulti)
- $(A^T)^{-1} = (A^{-1})^T$
- $(kA)^{-1} = \frac{1}{k}A^{-1}$

### Pseudo-inverse (jab inverse na ho) ⭐
$$A^+ = (A^TA)^{-1}A^T$$

```python
np.linalg.pinv(A)     # Moore-Penrose pseudo-inverse
```

Non-square matrices ke liye bhi kaam karta hai. **Least squares** ka solution yahi hai.

---

## 6. Rank ⭐

**Rank = kitne linearly independent rows (ya columns) hain.**

$$\text{rank}(A) \leq \min(m, n)$$

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.linalg.matrix_rank(A)     # 2 (3 nahi — teesri row dependent hai)
```

### Full Rank
- Square matrix: $\text{rank} = n$ → invertible
- Rank deficient: $\text{rank} < n$ → singular

### ML mein rank kyun matter karta hai?
1. **Multicollinearity detect karna** — agar $\text{rank}(X) < $ features ki count, toh redundant features hain
2. **Dimensionality reduction** — data ka effective dimension rank hi hai
3. **Low-rank approximation** — recommendation systems (Matrix Factorization) mein user-item matrix ko low rank matrices mein todte hain
4. **LoRA (Low-Rank Adaptation)** — LLM fine-tuning ki modern technique

```python
# Effective dimensionality check
X = np.random.randn(100, 10)
X[:, 5] = X[:, 0] + X[:, 1]        # redundant feature banaya
np.linalg.matrix_rank(X)            # 9, not 10
```

---

## 7. System of Linear Equations ⭐⭐

$$A\mathbf{x} = \mathbf{b}$$

Example:
$$\begin{cases} 2x + 3y = 8 \\ x - y = -1 \end{cases} \implies \begin{bmatrix} 2 & 3 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 8 \\ -1 \end{bmatrix}$$

### Solutions kitne ho sakte hain?

| Case | Condition | Geometric |
|---|---|---|
| **Unique solution** | $\det(A) \neq 0$ | Lines ek point pe milti hain |
| **No solution** | Inconsistent | Parallel lines |
| **Infinite solutions** | $\det(A)=0$ aur consistent | Lines overlap karti hain |

### Solve karne ke tareeqe

**1. `np.linalg.solve()` — best ⭐**
```python
A = np.array([[2, 3], [1, -1]])
b = np.array([8, -1])
x = np.linalg.solve(A, b)      # [1., 2.]
```

**2. Inverse se (avoid karo)**
```python
x = np.linalg.inv(A) @ b       # kaam karega lekin slow + unstable
```

**3. Cramer's Rule (chhote systems ke liye)**
$$x_i = \frac{\det(A_i)}{\det(A)}$$
jahan $A_i$ = A jismein i-th column ko $b$ se replace kar diya.

**4. Gaussian Elimination**
Row operations se upper triangular banao, phir back-substitution.

Allowed row operations:
- Do rows swap karo
- Kisi row ko non-zero scalar se multiply karo
- Ek row mein doosri row ka multiple add karo

**5. Least Squares (jab exact solution na ho)** ⭐
```python
x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
```

---

## 8. Linear Regression ka Closed-Form Solution ⭐⭐

Yahi ye poora chapter ka climax hai.

**Problem:** $X\mathbf{w} \approx \mathbf{y}$ — exact solution nahi hai (over-determined system: equations zyada, unknowns kam).

**Solution (Normal Equation):**
$$\boxed{\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}}$$

### Derivation (short)
Loss: $L = \|X\mathbf{w} - \mathbf{y}\|^2$
$$\frac{\partial L}{\partial \mathbf{w}} = 2X^T(X\mathbf{w}-\mathbf{y}) = 0$$
$$X^TX\mathbf{w} = X^T\mathbf{y} \implies \mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$$

```python
def linear_regression_closed_form(X, y):
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])     # bias column
    return np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Better (numerically stable):
def linear_regression_stable(X, y):
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    return np.linalg.lstsq(X_b, y, rcond=None)[0]

# Test
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 3 * X.ravel() + 5 + np.random.randn(100)
print(linear_regression_stable(X, y))     # ≈ [5, 3]
```

### Ye kab fail karta hai? ⚠️
1. $X^TX$ **singular** ho (multicollinearity) → inverse nahi milega
2. **Bahut zyada features** — $O(n^3)$ complexity, slow
3. Bahut bade datasets — memory issue

**Solution:** Gradient Descent (dekho [[09 Optimization and Gradient Descent]]) ya Ridge regularization.

---

## 9. Condition Number ⭐

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{max}}{\sigma_{min}}$$

Batata hai matrix kitna "numerically stable" hai.

| $\kappa$ | Matlab |
|---|---|
| ≈ 1 | Well-conditioned ✅ |
| Bada (>1000) | Ill-conditioned ⚠️ — chhoti si input error, badi output error |
| ∞ | Singular ❌ |

```python
np.linalg.cond(A)
```

**ML connection:** Ill-conditioned $X^TX$ ka matlab regression coefficients unreliable hain. Feature scaling condition number improve karti hai — isiliye standardization matter karta hai.

---

## 10. Practical Code — sab kuch ek jagah

```python
import numpy as np

A = np.array([[4, 7], [2, 6]])

# Determinant
det = np.linalg.det(A)
print(f"det = {det}")

# Inverse
if not np.isclose(det, 0):
    A_inv = np.linalg.inv(A)
    print("inverse:\n", A_inv)
    print("check:\n", np.round(A @ A_inv, 10))
else:
    print("Singular — pseudo-inverse use karo")
    print(np.linalg.pinv(A))

# Rank
print("rank =", np.linalg.matrix_rank(A))

# Condition number
print("cond =", np.linalg.cond(A))

# Solve Ax = b
b = np.array([1, 2])
print("x =", np.linalg.solve(A, b))

# Trace & eigenvalue relations
print("trace =", np.trace(A), "= sum of eigenvalues =", np.linalg.eigvals(A).sum())
print("det =", det, "= product of eigenvalues =", np.prod(np.linalg.eigvals(A)))
```

---

## 11. Interview Questions

1. **Determinant ka geometric matlab kya hai?** → area/volume scaling factor.
2. **$\det(A)=0$ ka kya matlab hai?** → matrix singular hai, space collapse karta hai, inverse nahi hai, columns dependent hain.
3. **Rank kya batata hai aur ML mein kyun matter karta hai?** → independent dimensions ki count; multicollinearity detect karta hai.
4. **`np.linalg.solve()` `inv()` se better kyun hai?** → faster ($O(n^3)$ lekin chhota constant) aur numerically zyada stable.
5. **Linear Regression ka closed-form solution kab fail karta hai?** → $X^TX$ singular ho ya features bahut zyada hon.
6. **Ridge regression singular matrix problem kaise solve karta hai?** → $\lambda I$ add karke matrix ko full rank bana deta hai.
7. **Pseudo-inverse kya hai?** → non-square/singular matrices ke liye generalized inverse; least squares solution deta hai.
8. **Condition number kya batata hai?**

---

## Aage padho
→ [[05 Linear Transformations]]
← [[03 Matrices]] | [[00 Maths for Machine Learning — MOC]]
