---
title: Maths for ML — Linear Transformations
tags:
  - maths
  - machine-learning
  - linear-algebra
  - transformations
date: 2026-08-30
---

# 05 — Linear Transformations

> **Ek line mein:** Matrix sirf numbers ka grid nahi hai — wo ek **function** hai jo space ko rotate, stretch, squash, ya flip karta hai.

---

## 1. Transformation kya hai?

Ek function jo ek vector leta hai aur doosra vector deta hai:
$$T: \mathbb{R}^n \rightarrow \mathbb{R}^m, \quad T(\mathbf{x}) = A\mathbf{x}$$

### Linear kab kehlata hai? (do conditions)
1. **Additivity:** $T(\mathbf{u}+\mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity:** $T(c\mathbf{u}) = cT(\mathbf{u})$

**Geometric matlab:**
- Grid lines **straight** rehti hain
- Grid lines **parallel** rehti hain aur **equally spaced**
- **Origin fixed** rehta hai (0 → 0)

**Isiliye "affine" transformation ($A\mathbf{x}+\mathbf{b}$) technically linear nahi hai** — kyunki origin shift ho jaata hai.

---

## 2. Sabse important insight ⭐⭐⭐

**Matrix ke columns batate hain ki basis vectors kahan jaate hain.**

$$A = \begin{bmatrix} | & | \\ T(\hat{i}) & T(\hat{j}) \\ | & | \end{bmatrix}$$

Agar $\hat{i}=[1,0]$ transform hokar $[2,1]$ ban jaata hai aur $\hat{j}=[0,1]$ transform hokar $[-1,3]$:

$$A = \begin{bmatrix} 2 & -1 \\ 1 & 3 \end{bmatrix}$$

**Aur koi bhi vector $\mathbf{v}=[x,y]$ kahan jayega?**
$$T(\mathbf{v}) = x \cdot T(\hat{i}) + y \cdot T(\hat{j}) = A\mathbf{v}$$

Bas! Matrix multiplication ka poora matlab yahi hai.

```python
import numpy as np
A = np.array([[2, -1], [1, 3]])
v = np.array([1, 2])
A @ v           # [0, 7]
```

---

## 3. Common Transformations ⭐

### 3.1 Identity (kuch nahi hota)
$$I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

### 3.2 Scaling
$$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$
- $s_x = s_y$ → uniform scaling
- $s_x \neq s_y$ → stretch/squash
- $s < 1$ → shrink

```python
S = np.array([[2, 0], [0, 0.5]])     # x 2x, y aadha
```

### 3.3 Rotation ⭐
$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

```python
def rotation_matrix(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)],
                     [np.sin(t),  np.cos(t)]])

R = rotation_matrix(90)
R @ np.array([1, 0])       # [0, 1] — i-hat 90° ghum gaya
```

**Rotation matrix ki khoobiyan:**
- $\det(R) = 1$ (area preserve)
- Orthogonal: $R^TR = I$, toh $R^{-1} = R^T = R(-\theta)$
- Length aur angles preserve karta hai

### 3.4 Reflection (mirror)
$$\text{x-axis: } \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}, \quad \text{y-axis: } \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}, \quad \text{y=x: } \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

$\det = -1$ (orientation flip).

### 3.5 Shear
$$\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix} \quad \text{(horizontal shear)}$$
Square ko parallelogram bana deta hai. $\det = 1$ (area same!).

### 3.6 Projection
$$P_x = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \quad \text{(x-axis pe project)}$$
$\det = 0$ — **information kho gayi** (2D → 1D). Isiliye irreversible hai.

**Projection matrix idempotent hoti hai:** $P^2 = P$ (do baar project karne se kuch naya nahi hota).

---

## 4. Composition — transformations jodna ⭐

Pehle $A$ apply karo, phir $B$:
$$T(\mathbf{x}) = B(A\mathbf{x}) = (BA)\mathbf{x}$$

**⚠️ Order matter karta hai!** Matrix multiplication **right se left** padhi jaati hai.

```python
R = rotation_matrix(45)
S = np.array([[2, 0], [0, 1]])

RS = R @ S      # pehle scale, phir rotate
SR = S @ R      # pehle rotate, phir scale
np.allclose(RS, SR)     # False! — alag results
```

**Yahi wajah hai ki $AB \neq BA$.** Ab ye sirf ek rule nahi, ek geometric fact hai.

---

## 5. Determinant ka role (revisit)

Ab determinant ka matlab poori tarah clear ho jaata hai:

| Transformation | $\det$ | Area pe asar |
|---|---|---|
| Rotation | 1 | Same |
| Shear | 1 | Same |
| Scaling by 2 (dono axes) | 4 | 4× |
| Reflection | -1 | Same lekin flipped |
| Projection | 0 | **Collapse** |

```python
for name, M in [('Rotation', rotation_matrix(45)),
                ('Scale', np.array([[2,0],[0,3]])),
                ('Shear', np.array([[1,1],[0,1]])),
                ('Reflect', np.array([[1,0],[0,-1]])),
                ('Project', np.array([[1,0],[0,0]]))]:
    print(f"{name:10s} det = {np.linalg.det(M):.2f}")
```

---

## 6. Visualization ka code ⭐

```python
import matplotlib.pyplot as plt

def plot_transformation(A, title='Transformation'):
    # Grid banao
    lines = []
    for i in range(-5, 6):
        lines.append(np.array([[i, i], [-5, 5]]))       # vertical
        lines.append(np.array([[-5, 5], [i, i]]))       # horizontal

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))

    # Original
    for l in lines:
        ax[0].plot(l[0], l[1], 'lightgray', lw=0.8)
    ax[0].quiver(0,0,1,0, angles='xy', scale_units='xy', scale=1, color='r', label='î')
    ax[0].quiver(0,0,0,1, angles='xy', scale_units='xy', scale=1, color='g', label='ĵ')
    ax[0].set_title('Original'); ax[0].legend()

    # Transformed
    for l in lines:
        tl = A @ l
        ax[1].plot(tl[0], tl[1], 'lightblue', lw=0.8)
    i_new, j_new = A @ np.array([1,0]), A @ np.array([0,1])
    ax[1].quiver(0,0,i_new[0],i_new[1], angles='xy', scale_units='xy', scale=1, color='r', label='T(î)')
    ax[1].quiver(0,0,j_new[0],j_new[1], angles='xy', scale_units='xy', scale=1, color='g', label='T(ĵ)')
    ax[1].set_title(f'{title}  (det = {np.linalg.det(A):.2f})'); ax[1].legend()

    for a in ax:
        a.set_xlim(-6,6); a.set_ylim(-6,6)
        a.axhline(0, color='k', lw=0.8); a.axvline(0, color='k', lw=0.8)
        a.set_aspect('equal'); a.grid(alpha=0.2)
    plt.tight_layout(); plt.show()

plot_transformation(np.array([[2, 1], [1, 3]]), 'Custom')
plot_transformation(rotation_matrix(45), 'Rotation 45°')
plot_transformation(np.array([[1, 1], [0, 1]]), 'Shear')
```

---

## 7. Kernel (Null Space) aur Range (Column Space) ⭐

### Null Space (Kernel)
$$\text{null}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$$
Wo saare vectors jo transformation ke baad **origin pe** aa jaate hain.

- Agar $A$ invertible hai → null space sirf $\{\mathbf{0}\}$
- Agar $\det(A)=0$ → null space mein aur bhi vectors hain (information kho gayi)

```python
from scipy.linalg import null_space
A = np.array([[1, 2], [2, 4]])
null_space(A)          # non-trivial null space
```

### Column Space (Range)
$A$ ke columns ka **span** — wo saare vectors jo output ho sakte hain.

$$\dim(\text{col space}) = \text{rank}(A)$$

### Rank-Nullity Theorem ⭐
$$\text{rank}(A) + \dim(\text{null}(A)) = n$$

**Intuition:** Jitni dimensions "bachi" (rank) + jitni "kho gayi" (nullity) = total input dimensions.

---

## 8. Change of Basis ⭐⭐

Same vector, alag coordinate system mein alag numbers.

$$[\mathbf{v}]_B = B^{-1}\mathbf{v}$$

jahan $B$ ke columns naye basis vectors hain (standard basis mein likhe hue).

**Transformation ka change of basis:**
$$A' = B^{-1}AB$$

Ye kehlata hai **similarity transformation**. Matlab: "Naye basis mein wahi transformation kaisa dikhta hai."

```python
B = np.array([[2, -1], [1, 1]])       # naya basis
A = np.array([[3, 0], [0, 2]])        # standard basis mein transformation

A_new = np.linalg.inv(B) @ A @ B      # naye basis mein
```

**ML connection:** PCA asal mein ek **change of basis** hai — data ko aise naye basis mein le jaana jahan axes maximum variance ki direction mein hon.

---

## 9. ML Applications ⭐

### 9.1 Neural Network layer = transformation
$$\mathbf{h} = \sigma(W\mathbf{x}+\mathbf{b})$$
$W\mathbf{x}$ linear transformation hai, $\sigma$ non-linearity add karta hai.

**Agar activation na hoti toh?** Multiple layers bhi ek hi linear transformation ke barabar hote:
$$W_3(W_2(W_1\mathbf{x})) = (W_3W_2W_1)\mathbf{x} = W\mathbf{x}$$
**Isiliye activation function zaroori hai** — warna 100 layers ka network ek single layer jaisa hi hota.

### 9.2 PCA = rotation to new basis
Data ko rotate karo taaki naye axes maximum variance ki direction mein hon, phir kam variance wale axes drop kar do.

### 9.3 Data Augmentation (Computer Vision)
Rotation, scaling, shear, reflection — sab linear transformations hain.

```python
# Image rotation
theta = np.radians(15)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
# har pixel coordinate pe R apply karo
```

### 9.4 Word Embeddings
Word2Vec/GloVe embeddings mein famous example:
$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$
Semantic relations vector space mein linear translations ban jaate hain.

### 9.5 Whitening
Data ko transform karna taaki covariance identity ban jaaye — features uncorrelated aur unit variance ke.

---

## 10. Affine vs Linear ⚠️

| | Linear | Affine |
|---|---|---|
| Form | $A\mathbf{x}$ | $A\mathbf{x} + \mathbf{b}$ |
| Origin | Fixed | Shift ho sakta hai |
| ML mein | Weights | Weights + bias |

**Trick:** Affine ko linear banane ke liye homogeneous coordinates use karo (dekho [[02 Lines Planes and Hyperplanes]] section 7).

---

## 11. Interview Questions

1. **Matrix ko transformation ke roop mein samjhao.** → columns batate hain basis vectors kahan jaate hain.
2. **$\det(A)=0$ transformation ke terms mein kya matlab?** → space kisi lower dimension mein collapse ho gaya, transformation irreversible hai.
3. **$AB \neq BA$ kyun?** → transformations ka order matter karta hai (pehle rotate phir scale ≠ pehle scale phir rotate).
4. **Neural network mein activation function kyun zaroori hai?** → warna multiple linear layers ek hi linear layer ke barabar ho jaate.
5. **Rotation matrix orthogonal kyun hoti hai?** → length aur angles preserve karti hai, isliye $R^TR=I$.
6. **Null space kya batata hai?** → wo input directions jo output mein "kho" jaati hain.
7. **PCA linear transformation kaise hai?** → data ko naye orthogonal basis (eigenvectors) mein rotate karta hai.

---

## Aage padho
→ [[06 Eigenvalues Eigenvectors PCA SVD]]
← [[04 Determinant Inverse Rank]] | [[00 Maths for Machine Learning — MOC]]
