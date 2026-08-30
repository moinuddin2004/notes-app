---
title: Maths for ML — Eigenvalues, Eigenvectors, PCA and SVD
tags:
  - maths
  - machine-learning
  - linear-algebra
  - eigenvalues
  - pca
  - svd
date: 2026-08-30
---

# 06 — Eigenvalues, Eigenvectors, PCA and SVD

> **Ek line mein:** Eigenvectors wo special directions hain jo transformation ke baad apni direction nahi badalte — sirf lambe/chhote hote hain. PCA aur SVD dono isi idea pe khade hain.

---

## 1. Eigen equation ⭐⭐⭐

$$\boxed{A\mathbf{v} = \lambda\mathbf{v}}$$

- $\mathbf{v}$ = **eigenvector** (non-zero)
- $\lambda$ = **eigenvalue** (scalar)

**Matlab kya hai?** Zyadatar vectors transformation ke baad apni direction badal dete hain. Lekin kuch special vectors **usi line pe** rehte hain — bas stretch/shrink ho jaate hain. Yahi eigenvectors hain.

| $\lambda$ | Kya hota hai |
|---|---|
| $\lambda > 1$ | Stretch |
| $0 < \lambda < 1$ | Shrink |
| $\lambda = 1$ | Bilkul same rehta hai |
| $\lambda < 0$ | Direction flip + scale |
| $\lambda = 0$ | Vector origin pe collapse (null space mein hai) |

---

## 2. Eigenvalues kaise nikaalein?

$$A\mathbf{v} = \lambda\mathbf{v} \implies (A - \lambda I)\mathbf{v} = \mathbf{0}$$

Non-zero $\mathbf{v}$ tabhi milega jab $(A-\lambda I)$ singular ho:

$$\boxed{\det(A - \lambda I) = 0}$$

Ye **characteristic equation** hai.

### Example (2×2)
$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

$$\det\begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix} = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = 0$$

$$\lambda = 5, \ 2$$

**Eigenvector for $\lambda=5$:**
$$(A - 5I)\mathbf{v} = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix}\mathbf{v} = 0 \implies \mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

```python
import numpy as np
A = np.array([[4, 1], [2, 3]])
eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)      # [5. 2.]
print(eigvecs)      # columns are eigenvectors (normalized)

# Verify
v = eigvecs[:, 0]
np.allclose(A @ v, eigvals[0] * v)     # True
```

**⚠️ Note:** NumPy eigenvectors ko **normalize** karke deta hai (unit length). Eigenvector kabhi unique nahi hota — koi bhi scalar multiple bhi eigenvector hai.

---

## 3. Important Properties ⭐

| Property | Formula |
|---|---|
| Sum | $\sum \lambda_i = \text{tr}(A)$ |
| Product | $\prod \lambda_i = \det(A)$ |
| Inverse | $A^{-1}$ ke eigenvalues = $1/\lambda_i$ |
| Power | $A^k$ ke eigenvalues = $\lambda_i^k$ |
| Transpose | $A^T$ ke eigenvalues = same as $A$ |
| Symmetric matrix | Eigenvalues **real**, eigenvectors **orthogonal** ⭐ |
| Positive definite | Saare $\lambda > 0$ |
| Singular ($\det=0$) | Kam se kam ek $\lambda = 0$ |

```python
A = np.array([[4, 1], [2, 3]])
eigvals = np.linalg.eigvals(A)
print(eigvals.sum(), np.trace(A))       # barabar
print(eigvals.prod(), np.linalg.det(A)) # barabar
```

---

## 4. Eigendecomposition ⭐

Agar $A$ ke $n$ linearly independent eigenvectors hain:

$$\boxed{A = V\Lambda V^{-1}}$$

- $V$ = eigenvectors columns mein
- $\Lambda$ = diagonal matrix eigenvalues ki

**Symmetric matrix ke liye (special case):**
$$A = Q\Lambda Q^T \quad (Q \text{ orthogonal hai})$$

```python
eigvals, V = np.linalg.eig(A)
L = np.diag(eigvals)
np.allclose(A, V @ L @ np.linalg.inv(V))     # True
```

### Fayda kya hai? — Matrix power
$$A^k = V\Lambda^kV^{-1}$$

$\Lambda^k$ nikalna trivial hai (bas diagonal elements ko power do). $A^{100}$ nikalna instant ho gaya.

```python
def matrix_power_eigen(A, k):
    vals, V = np.linalg.eig(A)
    return V @ np.diag(vals ** k) @ np.linalg.inv(V)
```

**Use case:** Markov chains, PageRank, recurrent networks ki stability analysis.

---

## 5. PCA — Principal Component Analysis ⭐⭐⭐

**Maksad:** Data ke dimensions kam karo lekin **maximum information (variance)** bachao.

### Intuition
Data ek cloud hai. PCA us cloud ki **sabse lambi direction** dhoondhta hai (jahan sabse zyada spread hai). Wo pehla principal component hai. Phir uske perpendicular sabse lambi direction — doosra PC. Aur aage.

### Steps ⭐

**Step 1 — Standardize (mean 0, std 1)**
$$X_{std} = \frac{X - \mu}{\sigma}$$

**Step 2 — Covariance matrix nikaalo**
$$\Sigma = \frac{1}{n-1}X_{std}^TX_{std}$$

**Step 3 — Eigenvalues aur eigenvectors nikaalo**
$$\Sigma\mathbf{v} = \lambda\mathbf{v}$$

**Step 4 — Eigenvalue ke hisaab se descending sort karo**
Bada $\lambda$ = us direction mein zyada variance.

**Step 5 — Top k eigenvectors chuno** → ye **principal components** hain

**Step 6 — Data ko project karo**
$$X_{new} = X_{std}W_k$$

### Explained Variance Ratio
$$\text{EVR}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

Batata hai har component kitna % information rakhta hai.

### Scratch se implementation ⭐

```python
import numpy as np

def pca(X, n_components=2):
    # 1. Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # 2. Covariance matrix
    cov = np.cov(X_std, rowvar=False)

    # 3. Eigendecomposition (symmetric → eigh use karo, faster & stable)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 4. Descending sort
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # 5. Top k
    W = eigvecs[:, :n_components]

    # 6. Project
    X_pca = X_std @ W

    evr = eigvals / eigvals.sum()
    return X_pca, W, eigvals, evr

# Test
from sklearn.datasets import load_iris
data = load_iris()
X_pca, W, eigvals, evr = pca(data.data, 2)

print("Explained variance ratio:", evr[:2])       # ~[0.73, 0.23]
print("Cumulative:", evr[:2].sum())               # ~0.96

import matplotlib.pyplot as plt
plt.scatter(X_pca[:,0], X_pca[:,1], c=data.target, cmap='viridis', edgecolor='k')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Iris — PCA')
plt.show()
```

### sklearn se comparison
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(data.data)
p = PCA(n_components=2)
X_p = p.fit_transform(X_std)
print(p.explained_variance_ratio_)
```

### Kitne components chunein?
```python
p = PCA().fit(X_std)
cum = np.cumsum(p.explained_variance_ratio_)

plt.plot(range(1, len(cum)+1), cum, 'o-')
plt.axhline(0.95, color='r', linestyle='--', label='95%')
plt.xlabel('Number of components'); plt.ylabel('Cumulative EVR')
plt.legend(); plt.grid(alpha=0.3); plt.show()

print("95% ke liye components:", np.argmax(cum >= 0.95) + 1)
```

**Rules of thumb:**
- 95% variance retain karo
- **Scree plot** ka "elbow" dekho
- Visualization ke liye 2 ya 3

### PCA ke Assumptions & Limitations ⚠️
- Linear relationships maanta hai
- Variance = information maanta hai (hamesha sach nahi)
- **Scaling ke bina kaam nahi karta** — bade scale wala feature dominate kar lega
- Components interpret karna mushkil hota hai
- Outliers se affected hota hai

---

## 6. SVD — Singular Value Decomposition ⭐⭐

Eigendecomposition sirf **square** matrices ke liye hai. SVD **kisi bhi** matrix ke liye kaam karta hai.

$$\boxed{A_{m\times n} = U_{m\times m}\Sigma_{m\times n}V^T_{n\times n}}$$

- $U$ = left singular vectors (orthogonal) — $AA^T$ ke eigenvectors
- $\Sigma$ = diagonal, **singular values** $\sigma_i \geq 0$ descending order mein
- $V$ = right singular vectors (orthogonal) — $A^TA$ ke eigenvectors

**Relation:** $\sigma_i = \sqrt{\lambda_i}$ jahan $\lambda_i$ = $A^TA$ ke eigenvalues.

```python
A = np.array([[3, 1, 1], [-1, 3, 1]])
U, s, Vt = np.linalg.svd(A)

print("U shape:", U.shape)       # (2,2)
print("s:", s)                    # singular values
print("Vt shape:", Vt.shape)      # (3,3)

# Reconstruct
S = np.zeros(A.shape)
np.fill_diagonal(S, s)
np.allclose(A, U @ S @ Vt)        # True
```

### Truncated SVD — Low-rank approximation ⭐
Sirf top $k$ singular values rakho:
$$A \approx U_k\Sigma_kV_k^T$$

Ye **best rank-k approximation** hai (Eckart–Young theorem).

```python
def low_rank_approx(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

### Image Compression example ⭐

```python
from skimage import data, color
import matplotlib.pyplot as plt

img = color.rgb2gray(data.astronaut())
U, s, Vt = np.linalg.svd(img, full_matrices=False)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, k in zip(axes, [5, 20, 50, 100, len(s)]):
    approx = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]
    ax.imshow(approx, cmap='gray')
    orig = img.size
    comp = k * (img.shape[0] + img.shape[1] + 1)
    ax.set_title(f'k={k}\n{comp/orig:.1%} storage')
    ax.axis('off')
plt.tight_layout(); plt.show()
```

**Sirf 50 singular values se image lagbhag original jaisi dikhti hai** — bade datasets mein information asal mein kam dimensions mein hoti hai.

---

## 7. PCA vs SVD ⭐

| | PCA | SVD |
|---|---|---|
| Input | Covariance matrix (square, symmetric) | Koi bhi matrix |
| Method | Eigendecomposition | Singular value decomposition |
| Centering | Zaroori | Optional |
| Stability | Kam (covariance banana loss deta hai) | **Zyada stable** ✅ |
| sklearn | Andar se SVD hi use karta hai! | — |

**Relation:** Agar $X$ mean-centered hai, toh $X$ ka SVD PCA de deta hai:
$$X = U\Sigma V^T \implies V \text{ ke columns = principal components}$$
$$\lambda_i = \frac{\sigma_i^2}{n-1}$$

```python
X_c = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
principal_components = Vt.T
explained_var = s**2 / (len(X) - 1)
```

---

## 8. ML Applications ⭐

| Application | Kya use hota hai |
|---|---|
| **Dimensionality reduction** | PCA / Truncated SVD |
| **Recommendation systems** | Matrix Factorization (SVD) — Netflix Prize |
| **LSA / Topic modelling** | Truncated SVD on TF-IDF matrix |
| **Image compression** | Low-rank SVD |
| **Noise reduction** | Chhote singular values drop karo |
| **PageRank** | Dominant eigenvector of link matrix |
| **Spectral clustering** | Graph Laplacian ke eigenvectors |
| **Face recognition (Eigenfaces)** | PCA on face images |
| **LoRA (LLM fine-tuning)** | Low-rank weight updates |
| **Whitening** | Eigendecomposition of covariance |

### Recommendation System example
```python
# User-item rating matrix (missing = 0)
R = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [1, 0, 0, 4],
              [0, 1, 5, 4]], dtype=float)

U, s, Vt = np.linalg.svd(R, full_matrices=False)
k = 2
R_pred = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]
print(np.round(R_pred, 2))     # missing ratings ka prediction
```

---

## 9. Eigenvectors ka visualization

```python
import matplotlib.pyplot as plt

def plot_eigen(A):
    vals, vecs = np.linalg.eig(A)

    # Random vectors
    theta = np.linspace(0, 2*np.pi, 20)
    circle = np.array([np.cos(theta), np.sin(theta)])
    transformed = A @ circle

    plt.figure(figsize=(8,8))
    plt.plot(circle[0], circle[1], 'lightgray', label='Unit circle')
    plt.plot(transformed[0], transformed[1], 'lightblue', label='Transformed')

    colors = ['red', 'green']
    for i in range(len(vals)):
        v = np.real(vecs[:, i])
        plt.quiver(0,0, v[0], v[1], angles='xy', scale_units='xy',
                   scale=1, color=colors[i], width=0.008,
                   label=f'Eigenvector λ={np.real(vals[i]):.2f}')
        Av = A @ v
        plt.quiver(0,0, Av[0], Av[1], angles='xy', scale_units='xy',
                   scale=1, color=colors[i], alpha=0.4, width=0.008)

    plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
    plt.gca().set_aspect('equal'); plt.grid(alpha=0.3); plt.legend()
    plt.title('Eigenvectors apni direction pe rehte hain')
    plt.show()

plot_eigen(np.array([[3, 1], [1, 2]]))
```

---

## 10. Interview Questions ⭐

1. **Eigenvector kya hai, simple words mein?** → wo direction jo transformation ke baad nahi badalti, sirf scale hoti hai.
2. **Eigenvalue kya batata hai?** → us direction mein kitna stretch/shrink hua.
3. **PCA ke steps batao.** → standardize → covariance → eigendecomposition → sort → top-k → project.
4. **PCA se pehle standardization kyun zaroori hai?** → warna bade scale wala feature (jaise salary vs age) variance dominate kar lega.
5. **Kitne principal components chunein?** → 95% cumulative explained variance ya scree plot ka elbow.
6. **PCA aur SVD mein farq?** → PCA covariance pe eigendecomposition; SVD direct data matrix pe, zyada stable, non-square ke liye bhi.
7. **Covariance matrix symmetric hone ka kya fayda?** → real eigenvalues + orthogonal eigenvectors → principal components hamesha perpendicular.
8. **PCA supervised hai ya unsupervised?** → **Unsupervised** — target label use nahi karta. (Supervised alternative: LDA)
9. **PCA kab use nahi karna chahiye?** → jab relationships non-linear hon (t-SNE/UMAP better), jab interpretability chahiye, jab features already kam hon.
10. **Recommendation systems mein SVD kaise kaam karta hai?** → user-item matrix ko latent factors mein todta hai; missing entries predict ho jaati hain.

---

## Aage padho
→ [[07 Functions and Derivatives]]
← [[05 Linear Transformations]] | [[00 Maths for Machine Learning — MOC]]
