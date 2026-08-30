---
title: Maths for ML — Descriptive Statistics
tags:
  - maths
  - machine-learning
  - statistics
  - eda
date: 2026-08-30
---

# 10 — Descriptive Statistics

> **Ek line mein:** Descriptive statistics data ko summarize karti hai — center kahan hai, spread kitna hai, aur shape kaisa hai.

---

## 1. Population vs Sample ⭐

| | Population | Sample |
|---|---|---|
| Kya hai | Poora group | Uska ek hissa |
| Size | $N$ | $n$ |
| Mean | $\mu$ | $\bar{x}$ |
| Std dev | $\sigma$ | $s$ |
| Variance denominator | $N$ | $n-1$ ⭐ |

**$n-1$ kyun? (Bessel's correction)** Sample variance population variance ko underestimate karta hai kyunki hum sample mean use kar rahe hain (jo sample ke andar hi fit hai). $n-1$ se divide karke unbiased estimate milta hai.

```python
import numpy as np
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])

np.var(data)             # population variance (ddof=0)
np.var(data, ddof=1)     # sample variance (ddof=1) ⭐
```

**⚠️ Note:** NumPy default `ddof=0` (population), Pandas default `ddof=1` (sample). Ye common confusion hai.

---

## 2. Measures of Central Tendency ⭐

### Mean (average)
$$\bar{x} = \frac{1}{n}\sum x_i$$

**Outliers se bahut affected hota hai.**

### Median
Sorted data ka beech wala value.
- $n$ odd → beech ka element
- $n$ even → beech ke do ka average

**Outliers se safe hai** ⭐ — isiliye income/house prices mein median use karte hain.

### Mode
Sabse zyada baar aane wali value. Categorical data ke liye ek hi option.

```python
import pandas as pd
from scipy import stats

s = pd.Series([2, 4, 4, 4, 5, 5, 7, 9])
s.mean()      # 5.0
s.median()    # 4.5
s.mode()      # 4
stats.mode(s, keepdims=False)
```

### Kaunsa kab use karein? ⭐

| Situation | Use |
|---|---|
| Symmetric distribution, no outliers | **Mean** |
| Skewed data ya outliers hain | **Median** |
| Categorical data | **Mode** |
| Missing values fill karna (numeric, normal) | Mean |
| Missing values fill karna (numeric, skewed) | Median |
| Missing values fill karna (categorical) | Mode |

### Weighted Mean
$$\bar{x}_w = \frac{\sum w_ix_i}{\sum w_i}$$

```python
np.average(data, weights=w)
```

### Geometric Mean (growth rates ke liye)
$$GM = \left(\prod x_i\right)^{1/n}$$

```python
stats.gmean([1.05, 1.10, 0.95])
```

### Harmonic Mean (rates ke liye — F1-score!)
$$HM = \frac{n}{\sum \frac{1}{x_i}}$$

**F1-score precision aur recall ka harmonic mean hai:**
$$F1 = \frac{2 \cdot P \cdot R}{P + R}$$

```python
stats.hmean([0.8, 0.6])     # F1 = 0.686
```

---

## 3. Measures of Dispersion (Spread) ⭐

### Range
$$\text{Range} = \max - \min$$
Simple lekin outliers se poori tarah affected.

### Variance
$$\sigma^2 = \frac{1}{N}\sum(x_i-\mu)^2 \quad\text{(population)}$$
$$s^2 = \frac{1}{n-1}\sum(x_i-\bar{x})^2 \quad\text{(sample)}$$

**Squared units mein hoti hai** — interpret karna mushkil.

### Standard Deviation ⭐
$$\sigma = \sqrt{\sigma^2}$$

Original units mein wapas — interpret karna aasan.

### Interquartile Range (IQR) ⭐
$$IQR = Q_3 - Q_1$$

Middle 50% ka spread. **Outliers se safe.**

### Coefficient of Variation
$$CV = \frac{\sigma}{\mu} \times 100\%$$

Relative variability — alag units wale datasets compare karne ke liye.

```python
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])

data.max() - data.min()            # Range
np.var(data, ddof=1)               # Variance
np.std(data, ddof=1)               # Std dev
np.percentile(data, 75) - np.percentile(data, 25)   # IQR
stats.iqr(data)
np.std(data)/np.mean(data)*100     # CV
```

---

## 4. Quartiles & Percentiles ⭐

- $Q_1$ = 25th percentile
- $Q_2$ = 50th percentile = **median**
- $Q_3$ = 75th percentile

```python
np.percentile(data, [25, 50, 75])
np.quantile(data, [0.25, 0.5, 0.75])
pd.Series(data).describe()
```

### Five-number summary
`min, Q1, median, Q3, max` — box plot yahi dikhata hai.

---

## 5. Outlier Detection ⭐⭐

### Method 1: IQR method (most common)
$$\text{Outlier if: } x < Q_1 - 1.5\,IQR \quad\text{or}\quad x > Q_3 + 1.5\,IQR$$

```python
def detect_outliers_iqr(data, k=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k*iqr, q3 + k*iqr
    return data[(data < lower) | (data > upper)], (lower, upper)
```

**$k=1.5$ kyun?** Normal distribution mein ye ~$\pm 2.7\sigma$ ke barabar hai, yaani ~99.3% data andar aa jaata hai. $k=3$ "extreme outliers" ke liye.

### Method 2: Z-score
$$z = \frac{x-\mu}{\sigma}, \quad |z| > 3 \implies \text{outlier}$$

```python
def detect_outliers_zscore(data, threshold=3):
    z = np.abs(stats.zscore(data))
    return data[z > threshold]
```

**⚠️ Problem:** Mean aur std khud outliers se affected hote hain.

### Method 3: Modified Z-score (robust) ⭐
$$M_i = \frac{0.6745(x_i - \tilde{x})}{MAD}, \quad MAD = \text{median}(|x_i-\tilde{x}|)$$

```python
def modified_zscore(data, threshold=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    m = 0.6745 * (data - median) / mad
    return data[np.abs(m) > threshold]
```

Median-based hai, isliye outliers se affect nahi hota.

---

## 6. Skewness ⭐

Distribution ki **asymmetry**.

$$\text{Skew} = \frac{1}{n}\sum\left(\frac{x_i-\bar{x}}{s}\right)^3$$

| Skewness | Shape | Relation |
|---|---|---|
| $= 0$ | Symmetric | mean = median = mode |
| $> 0$ | **Right-skewed** (long right tail) | mean > median > mode |
| $< 0$ | **Left-skewed** (long left tail) | mean < median < mode |

**Rule of thumb:** $|skew| < 0.5$ approximately symmetric; $> 1$ highly skewed.

```python
stats.skew(data)
pd.Series(data).skew()
```

**ML mein kyun matter?** Bahut se algorithms normally-distributed features assume karte hain. Skewed features ko transform karo:

```python
np.log1p(data)                  # log transform (right skew ke liye)
np.sqrt(data)                   # square root
stats.boxcox(data)              # Box-Cox (data > 0 hona chahiye)
stats.yeojohnson(data)          # Yeo-Johnson (negative bhi handle karta hai)

from sklearn.preprocessing import PowerTransformer
PowerTransformer(method='yeo-johnson').fit_transform(X)
```

---

## 7. Kurtosis

Distribution kitni "peaked" hai aur tails kitni heavy hain.

| Type | Excess Kurtosis | Matlab |
|---|---|---|
| **Mesokurtic** | ≈ 0 | Normal jaisa |
| **Leptokurtic** | > 0 | Sharp peak, heavy tails (zyada outliers) |
| **Platykurtic** | < 0 | Flat, thin tails |

```python
stats.kurtosis(data)          # excess kurtosis (normal = 0)
stats.kurtosis(data, fisher=False)   # normal = 3
```

---

## 8. Covariance ⭐

Do variables saath mein kaise badalte hain.

$$\text{Cov}(X,Y) = \frac{1}{n-1}\sum(x_i-\bar{x})(y_i-\bar{y})$$

| Value | Matlab |
|---|---|
| $> 0$ | Saath badhte hain |
| $< 0$ | Ek badhta, doosra ghatta |
| $= 0$ | Koi linear relation nahi |

**⚠️ Problem:** Magnitude ka koi matlab nahi — units pe depend karta hai. Isliye **correlation** use karte hain.

```python
np.cov(x, y)                    # covariance matrix
np.cov(X, rowvar=False)         # multiple variables
df.cov()
```

### Covariance Matrix ⭐
$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

Diagonal pe variances, off-diagonal pe covariances. **Symmetric aur positive semi-definite.** PCA ka input yahi hai.

---

## 9. Correlation ⭐⭐

### Pearson Correlation
$$r = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y}$$

Range: $[-1, 1]$. **Sirf LINEAR relationship measure karta hai.**

| $|r|$ | Strength |
|---|---|
| 0.0 – 0.3 | Weak |
| 0.3 – 0.7 | Moderate |
| 0.7 – 1.0 | Strong |

### Spearman Correlation (rank-based)
Values ki jagah unke **ranks** ka Pearson correlation.
- **Monotonic** relationships capture karta hai (sirf linear nahi)
- Outliers se robust
- Ordinal data ke liye

### Kendall's Tau
Concordant vs discordant pairs pe based. Chhote samples ke liye better.

```python
stats.pearsonr(x, y)       # (correlation, p-value)
stats.spearmanr(x, y)
stats.kendalltau(x, y)

df.corr()                       # Pearson (default)
df.corr(method='spearman')
df.corr(method='kendall')
```

### ⚠️ Bahut important warnings

**1. Correlation ≠ Causation**
Ice cream sales aur drowning deaths correlated hain — dono ka asli cause "garmi" hai.

**2. $r = 0$ ka matlab "koi relation nahi" nahi hai**
```python
x = np.linspace(-5, 5, 100)
y = x**2                    # perfect relationship!
stats.pearsonr(x, y)        # r ≈ 0 !!
```
Pearson sirf **linear** relation dekhta hai.

**3. Anscombe's Quartet** — 4 datasets jinke mean, variance, aur correlation bilkul same hain lekin plots bilkul alag. **Hamesha plot karo.**

```python
import seaborn as sns
df = sns.load_dataset('anscombe')
sns.lmplot(data=df, x='x', y='y', col='dataset', col_wrap=2, height=3)
```

---

## 10. Standardization vs Normalization ⭐⭐

### Standardization (Z-score)
$$z = \frac{x-\mu}{\sigma}$$
Result: mean = 0, std = 1. Range unbounded.

**Kab use karein?** Zyadatar cases mein — PCA, SVM, Linear/Logistic Regression, Neural Networks.

### Min-Max Normalization
$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$
Result: [0, 1] range.

**Kab use karein?** Jab bounded range chahiye — image pixels, neural network inputs.

### Robust Scaling
$$x' = \frac{x - \text{median}}{IQR}$$

**Kab?** Jab outliers hain.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

StandardScaler().fit_transform(X)
MinMaxScaler().fit_transform(X)
RobustScaler().fit_transform(X)
```

**⚠️ Golden rule:** Scaler ko sirf **training data** pe `fit` karo, phir train aur test dono pe `transform`. Warna data leakage ho jayega.

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)     # fit + transform
X_test_s  = scaler.transform(X_test)          # sirf transform ⭐
```

### Kaunse algorithms ko scaling chahiye?

| Chahiye ✅ | Nahi chahiye ❌ |
|---|---|
| KNN, K-Means | Decision Trees |
| SVM | Random Forest |
| PCA | Gradient Boosting / XGBoost |
| Linear/Logistic Regression (with regularization) | Naive Bayes |
| Neural Networks | |

**Rule:** Jo algorithm **distance** ya **gradient** pe based hai, usko scaling chahiye. Tree-based algorithms splits pe kaam karte hain, unko fark nahi padta.

---

## 11. Complete EDA function ⭐

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def describe_column(series, name='Column'):
    s = series.dropna()
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1

    print(f"{'='*45}\n  {name}\n{'='*45}")
    print(f"Count        : {len(s)}")
    print(f"Missing      : {series.isnull().sum()}")
    print(f"\n--- Central Tendency ---")
    print(f"Mean         : {s.mean():.4f}")
    print(f"Median       : {s.median():.4f}")
    print(f"Mode         : {s.mode().values[:3]}")
    print(f"\n--- Dispersion ---")
    print(f"Std Dev      : {s.std():.4f}")
    print(f"Variance     : {s.var():.4f}")
    print(f"Range        : {s.max()-s.min():.4f}")
    print(f"IQR          : {iqr:.4f}")
    print(f"CV           : {s.std()/s.mean()*100:.2f}%")
    print(f"\n--- Shape ---")
    print(f"Skewness     : {stats.skew(s):.4f}")
    print(f"Kurtosis     : {stats.kurtosis(s):.4f}")
    print(f"\n--- Outliers (IQR) ---")
    lo, hi = q1-1.5*iqr, q3+1.5*iqr
    out = s[(s<lo)|(s>hi)]
    print(f"Bounds       : [{lo:.2f}, {hi:.2f}]")
    print(f"Count        : {len(out)} ({len(out)/len(s)*100:.1f}%)")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(s, kde=True, ax=ax[0]); ax[0].set_title('Distribution')
    sns.boxplot(y=s, ax=ax[1]); ax[1].set_title('Box Plot')
    stats.probplot(s, dist='norm', plot=ax[2]); ax[2].set_title('Q-Q Plot')
    plt.tight_layout(); plt.show()

# Usage
tips = sns.load_dataset('tips')
describe_column(tips['total_bill'], 'Total Bill')
```

---

## 12. Interview Questions

1. **Mean, median, mode mein kab kya use karein?**
2. **Sample variance mein $n-1$ kyun?** → Bessel's correction, unbiased estimate ke liye.
3. **Covariance aur correlation mein farq?** → correlation normalized covariance hai, [-1,1] range, unit-free.
4. **$r=0$ ka matlab kya hai?** → koi **linear** relation nahi; non-linear relation ho sakta hai.
5. **Outliers kaise detect karein?** → IQR method, Z-score, modified Z-score (MAD).
6. **Standardization aur normalization mein farq, aur kab kya?**
7. **Kaunse algorithms ko feature scaling chahiye?** → distance-based aur gradient-based; trees ko nahi.
8. **Skewness ML ko kaise affect karti hai?** → normality assumption todti hai; log/Box-Cox transform se fix.
9. **Correlation causation kyun nahi hai?** → confounding variables, reverse causation, coincidence.
10. **Test data pe `fit_transform` kyun nahi karte?** → data leakage — test ki information model tak pahunch jayegi.

---

## Aage padho
→ [[11 Probability Basics]]
← [[09 Optimization and Gradient Descent]] | [[00 Maths for Machine Learning — MOC]]
