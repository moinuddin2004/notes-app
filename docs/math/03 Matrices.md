---
title: Maths for ML — Matrices
tags:
  - maths
  - machine-learning
  - linear-algebra
  - matrices
date: 2026-08-30
---

# 03 — Matrices

> **Ek line mein:** Matrix numbers ka 2D grid hai. ML mein poora dataset ek matrix hai, aur neural network ka har layer ek matrix multiplication.

---

## 1. Matrix kya hai?

$$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix}_{2\times3}$$

$a_{ij}$ = i-th row, j-th column ka element.

**Shape/Order:** rows × columns. Upar wala matrix $2\times3$ hai.

```python
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
A.shape      # (2, 3)
A[0, 2]      # 3
```

### ML mein matrix ka matlab ⭐
$$X_{m \times n} = \begin{bmatrix} \text{--- } \mathbf{x}^{(1)} \text{ ---} \\ \text{--- } \mathbf{x}^{(2)} \text{ ---} \\ \vdots \\ \text{--- } \mathbf{x}^{(m)} \text{ ---} \end{bmatrix}$$

- $m$ = kitne samples (rows)
- $n$ = kitne features (columns)

Ye **design matrix** kehlata hai. Pandas ka DataFrame yahi hai.

---

## 2. Matrices ke types ⭐

| Type | Definition | Example |
|---|---|---|
| **Row matrix** | Sirf 1 row | $[1\ 2\ 3]$ |
| **Column matrix** | Sirf 1 column | $[1;2;3]$ |
| **Square** | rows = columns | $3\times3$ |
| **Rectangular** | rows ≠ columns | $2\times3$ |
| **Diagonal** | Sirf diagonal pe non-zero | $\text{diag}(1,2,3)$ |
| **Identity ($I$)** | Diagonal pe sab 1 | `np.eye(3)` |
| **Zero/Null** | Sab elements 0 | `np.zeros((3,3))` |
| **Upper Triangular** | Diagonal ke neeche sab 0 | — |
| **Lower Triangular** | Diagonal ke upar sab 0 | — |
| **Symmetric** | $A = A^T$ | Covariance matrix |
| **Skew-symmetric** | $A = -A^T$ | — |
| **Orthogonal** | $A^TA = I$ | Rotation matrix |
| **Singular** | $\det(A)=0$, inverse nahi hai | — |
| **Sparse** | Zyadatar elements 0 | Text data (TF-IDF) |

```python
np.eye(3)                        # Identity
np.zeros((3,3))                  # Zero matrix
np.diag([1,2,3])                 # Diagonal
np.triu(A)                       # Upper triangular
np.tril(A)                       # Lower triangular
np.allclose(A, A.T)              # Symmetric check
```

---

## 3. Matrix Operations

### 3.1 Addition / Subtraction
Element-wise. **Shapes same hone chahiye.**

$$(A+B)_{ij} = a_{ij} + b_{ij}$$

```python
A + B
A - B
```

### 3.2 Scalar Multiplication
```python
3 * A
```

### 3.3 Element-wise (Hadamard) Product $\odot$
```python
A * B            # ⚠️ ye MATRIX multiplication NAHI hai
np.multiply(A, B)
```

**Kahan use hota hai?** Neural nets mein masking, dropout, attention weights.

---

## 4. Matrix Multiplication ⭐⭐⭐

$$C = AB \quad\text{where}\quad c_{ij} = \sum_{k} a_{ik}b_{kj}$$

**Rule:** $(m \times n) \times (n \times p) = (m \times p)$

Pehle ka **columns** = doosre ka **rows** hona zaroori hai.

```
   A (2×3)  @  B (3×2)  =  C (2×2)
        ↑        ↑
        └── ye match hone chahiye
```

```python
A = np.array([[1, 2, 3], [4, 5, 6]])       # (2,3)
B = np.array([[7, 8], [9, 10], [11, 12]])  # (3,2)

A @ B                # recommended
np.matmul(A, B)
np.dot(A, B)
```

### Manual calculation samajhne ke liye
$c_{ij}$ = A ki i-th row ka **dot product** B ke j-th column ke saath.

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1(5)+2(7) & 1(6)+2(8) \\ 3(5)+4(7) & 3(6)+4(8) \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

### Properties ⭐
- **NOT commutative:** $AB \neq BA$ (aksar) ⚠️ sabse badi galti
- Associative: $(AB)C = A(BC)$ ✅
- Distributive: $A(B+C) = AB + AC$ ✅
- $AI = IA = A$ ✅
- $(AB)^T = B^TA^T$ ⭐ (order ulti ho jaati hai)

---

## 5. Transpose

Rows aur columns swap.
$$(A^T)_{ij} = a_{ji}$$

```python
A.T
np.transpose(A)
```

**Properties:**
- $(A^T)^T = A$
- $(A+B)^T = A^T + B^T$
- $(AB)^T = B^TA^T$ ⭐
- $(kA)^T = kA^T$

---

## 6. Trace

Diagonal elements ka sum (sirf square matrix ke liye).
$$\text{tr}(A) = \sum_i a_{ii}$$

```python
np.trace(A)
```

**Properties:**
- $\text{tr}(A+B) = \text{tr}(A)+\text{tr}(B)$
- $\text{tr}(AB) = \text{tr}(BA)$ ⭐ (cyclic property)
- $\text{tr}(A) = \sum \lambda_i$ (eigenvalues ka sum!)

---

## 7. Special Matrices detail mein

### Identity Matrix
$$I_3 = \begin{bmatrix} 1&0&0 \\ 0&1&0 \\ 0&0&1 \end{bmatrix}$$
Multiplication ka "1" — $AI = A$.

### Symmetric Matrix ⭐
$A = A^T$

**ML mein kahan?** **Covariance matrix** hamesha symmetric hoti hai. Symmetric matrices ke eigenvalues hamesha **real** hote hain aur eigenvectors **orthogonal** — isiliye PCA itna clean kaam karta hai.

### Orthogonal Matrix ⭐
$$Q^TQ = QQ^T = I \implies Q^{-1} = Q^T$$

Iska matlab inverse nikalna free ho gaya! Rotation matrices orthogonal hoti hain — length preserve karti hain.

```python
theta = np.pi/4
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
np.allclose(Q.T @ Q, np.eye(2))       # True
```

### Positive Definite Matrix
$\mathbf{x}^TA\mathbf{x} > 0$ for all $\mathbf{x} \neq 0$. Saare eigenvalues positive.

**Kyun important?** Agar Hessian positive definite hai → wo point ek **minimum** hai. Optimization mein critical.

```python
np.all(np.linalg.eigvals(A) > 0)     # positive definite check
```

---

## 8. Matrix Powers

```python
np.linalg.matrix_power(A, 3)     # A @ A @ A
A ** 3                            # ⚠️ ye element-wise power hai!
```

**Idempotent matrix:** $A^2 = A$ (projection matrices aisi hoti hain)

---

## 9. Broadcasting ke saath Matrix operations

```python
A = np.arange(6).reshape(2, 3)
b = np.array([1, 2, 3])

A + b           # b har row mein add ho jayega
A * b           # column-wise multiply

# Column vector ke saath
c = np.array([[10], [20]])
A + c           # c har column mein add
```

---

## 10. ML Applications ⭐⭐

### 10.1 Linear Regression — vectorized form
$$\hat{\mathbf{y}} = X\mathbf{w}$$

```python
X = np.array([[1, 2], [1, 3], [1, 4]])     # bias column ke saath
w = np.array([0.5, 2.0])
y_pred = X @ w
```

### 10.2 Neural Network layer ⭐
$$\mathbf{z} = W\mathbf{x} + \mathbf{b}, \quad \mathbf{a} = \sigma(\mathbf{z})$$

```python
def dense_layer(X, W, b, activation=lambda z: np.maximum(0, z)):
    return activation(X @ W + b)

X = np.random.randn(32, 784)     # 32 samples, 784 features
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros(128)
h1 = dense_layer(X, W1, b1)      # (32, 128)
```

**Poora deep learning bas matrix multiplications ki chain hai.**

### 10.3 Covariance Matrix ⭐
$$\Sigma = \frac{1}{n-1}X_c^TX_c \quad (X_c = \text{mean-centered } X)$$

```python
X = np.random.randn(100, 3)
Xc = X - X.mean(axis=0)
cov = (Xc.T @ Xc) / (len(X) - 1)
np.allclose(cov, np.cov(X, rowvar=False))    # True
```

Ye **symmetric** aur **positive semi-definite** hoti hai — PCA ka starting point.

### 10.4 Gram Matrix
$$G = X X^T$$
Har pair ki similarity. Kernel methods aur style transfer mein use hoti hai.

### 10.5 One-hot encoding
```python
labels = np.array([0, 2, 1, 2])
onehot = np.eye(3)[labels]
# [[1,0,0],
#  [0,0,1],
#  [0,1,0],
#  [0,0,1]]
```

---

## 11. Computational Complexity

| Operation | Complexity |
|---|---|
| Addition | $O(mn)$ |
| Matrix multiplication ($n\times n$) | $O(n^3)$ naive, $O(n^{2.37})$ best known |
| Transpose | $O(1)$ (NumPy sirf view banata hai!) |
| Inverse | $O(n^3)$ |
| Determinant | $O(n^3)$ |

**Isiliye bade matrices ke saath inverse avoid karte hain** — `np.linalg.solve()` faster aur numerically stable hai.

---

## 12. NumPy Cheatsheet

```python
import numpy as np

# Creation
np.array([[1,2],[3,4]])
np.zeros((3,3)); np.ones((3,3)); np.eye(3)
np.random.randn(3,3)
np.diag([1,2,3])
np.full((2,3), 7)

# Properties
A.shape; A.size; A.ndim; A.dtype
np.trace(A)
np.linalg.matrix_rank(A)
np.linalg.det(A)

# Operations
A + B; A - B; 3*A
A * B                    # element-wise
A @ B                    # matrix multiplication
A.T                      # transpose
np.linalg.inv(A)         # inverse
np.linalg.matrix_power(A, 3)

# Solving
np.linalg.solve(A, b)    # Ax = b  (inv se better!)
np.linalg.lstsq(A, b, rcond=None)   # least squares

# Decompositions
np.linalg.eig(A)         # eigenvalues, eigenvectors
np.linalg.svd(A)         # SVD
np.linalg.qr(A)          # QR
np.linalg.cholesky(A)    # Cholesky (positive definite ke liye)

# Norms
np.linalg.norm(A)              # Frobenius
np.linalg.norm(A, 'fro')
np.linalg.norm(A, 1)           # max column sum
np.linalg.norm(A, np.inf)      # max row sum

# Reshaping / stacking
A.reshape(3, 2); A.ravel()
np.hstack([A, B]); np.vstack([A, B])
np.concatenate([A, B], axis=0)
```

---

## 13. Common Mistakes ⚠️

1. **`A * B` ko matrix multiplication samajhna** — wo element-wise hai. `A @ B` use karo.
2. **`AB = BA` maan lena** — matrix multiplication commutative nahi hai.
3. **Shape mismatch** — hamesha `.shape` print karke check karo.
4. **`A ** 2` ko $A^2$ samajhna** — wo element-wise square hai; `np.linalg.matrix_power(A, 2)` use karo.
5. **Inverse use karna jab `solve()` chahiye** — slow aur numerically unstable.
6. **`(AB)^T = A^T B^T` likhna** — sahi hai $(AB)^T = B^TA^T$.

---

## 14. Interview Questions

1. **Matrix multiplication ki condition kya hai?**
2. **$AB = BA$ kab hota hai?** → jab dono diagonal hon, ya ek identity ho, ya wo commute karte hon (special cases).
3. **Symmetric matrix ki khaas baat?** → real eigenvalues, orthogonal eigenvectors.
4. **Orthogonal matrix kyun useful hai?** → $Q^{-1} = Q^T$, length aur angles preserve karta hai.
5. **Covariance matrix symmetric kyun hoti hai?** → $\text{cov}(X,Y) = \text{cov}(Y,X)$.
6. **Neural network mein matrix multiplication kahan hoti hai?** → har layer: $Wx + b$.
7. **Sparse matrix kya hai aur kab use karte hain?** → text data (TF-IDF), memory bachane ke liye `scipy.sparse`.

---

## Aage padho
→ [[04 Determinant Inverse Rank]]
← [[02 Lines Planes and Hyperplanes]] | [[00 Maths for Machine Learning — MOC]]
