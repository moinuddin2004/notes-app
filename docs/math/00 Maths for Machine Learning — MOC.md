---
title: Maths for Machine Learning — MOC
tags:
  - maths
  - machine-learning
  - moc
  - campusx
date: 2026-08-30
---

# Maths for Machine Learning — Map of Content

**Source:** [Maths for Machine Learning — CampusX](https://www.youtube.com/playlist?list=PLKnIA16_RmvbYFaaeLY28cWeqV-3vADST) (23 videos)

ML ki har algorithm asal mein maths hai. Ye playlist wo saara maths cover karti hai jo ML aur Deep Learning ke liye chahiye — **Linear Algebra**, **Calculus**, aur **Probability & Statistics**.

---

## Notes Index

### 🟦 Linear Algebra
| Note | Topics |
|---|---|
| [[01 Vectors]] | Vector kya hai, row/column, magnitude, unit vector, addition, scalar multiplication, dot product, angle, projection, cross product, norms (L1/L2/L∞) |
| [[02 Lines Planes and Hyperplanes]] | 2D line, 3D plane, n-D hyperplane, general equation, distance from point, ML mein decision boundary |
| [[03 Matrices]] | Matrix types, operations, multiplication, transpose, trace, special matrices, NumPy implementation |
| [[04 Determinant Inverse Rank]] | Determinant ka matlab, inverse, rank, singular matrix, system of linear equations solve karna |
| [[05 Linear Transformations]] | Matrix as transformation, rotation/scaling/shear, composition, basis change, geometric intuition |
| [[06 Eigenvalues Eigenvectors PCA SVD]] | Eigen equation, eigendecomposition, PCA step-by-step, SVD, dimensionality reduction |

### 🟩 Calculus
| Note | Topics |
|---|---|
| [[07 Functions and Derivatives]] | Function, limit, continuity, derivative ka matlab, rules, common derivatives, higher order |
| [[08 Multivariable Calculus and Gradient]] | Partial derivative, gradient vector, chain rule, Jacobian, Hessian, directional derivative |
| [[09 Optimization and Gradient Descent]] | Maxima/minima, convexity, gradient descent, learning rate, variants, backpropagation ka maths |

### 🟨 Probability & Statistics
| Note | Topics |
|---|---|
| [[10 Descriptive Statistics]] | Mean/median/mode, variance, std, quartiles, skewness, kurtosis, covariance, correlation |
| [[11 Probability Basics]] | Sample space, conditional probability, independence, Bayes theorem, Naive Bayes |
| [[12 Random Variables and Distributions]] | Discrete/continuous RV, PMF/PDF/CDF, Bernoulli, Binomial, Poisson, Normal, expectation, variance |
| [[13 Inferential Statistics]] | Sampling, CLT, confidence interval, hypothesis testing, p-value, t-test, chi-square, ANOVA |

---

## ML mein Maths kahan use hota hai?

| Maths Topic | Kahan use hota hai |
|---|---|
| Vectors, Dot Product | Feature vectors, similarity (cosine), embeddings |
| Hyperplanes | SVM, Logistic Regression ki decision boundary |
| Matrix Multiplication | Neural network ka har layer |
| Determinant, Inverse | Linear Regression ka closed-form solution |
| Eigenvalues/Eigenvectors | PCA, spectral clustering, PageRank |
| SVD | Recommendation systems, LSA, image compression |
| Derivatives | Loss function ko minimize karna |
| Partial derivatives, Gradient | Backpropagation |
| Chain rule | Deep learning ki poori training |
| Convexity | Kya optimization guaranteed converge karega |
| Probability | Naive Bayes, generative models |
| Bayes theorem | Bayesian ML, spam filters |
| Distributions | Assumptions, noise modelling, GMM |
| Expectation/Variance | Bias-Variance tradeoff |
| Covariance matrix | PCA, Mahalanobis distance, LDA |
| Hypothesis testing | Feature selection, A/B testing |

---

## Study order (recommendation)

```
Vectors → Hyperplanes → Matrices → Determinant/Inverse
    ↓
Linear Transformations → Eigenvalues → PCA/SVD
    ↓
Derivatives → Partial Derivatives/Gradient → Gradient Descent
    ↓
Descriptive Stats → Probability → Distributions → Inference
```

**Time estimate:** har note ko theek se padhne aur code chalane mein 1–2 ghante. Poori series ~25–30 ghante.

---

## Setup (har note ka code chalane ke liye)

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sympy as sp
from scipy import linalg

np.set_printoptions(precision=3, suppress=True)
plt.rcParams['figure.figsize'] = (8, 6)
```

---

## Extra resources

- **3Blue1Brown — Essence of Linear Algebra** (visual intuition ke liye best)
- **3Blue1Brown — Essence of Calculus**
- **Mathematics for Machine Learning** book — mml-book.github.io (free PDF)
- **StatQuest** (statistics ke liye)
- **Khan Academy** — Linear Algebra, Multivariable Calculus, Statistics
