---
title: Maths for ML — Random Variables and Distributions
tags:
  - maths
  - machine-learning
  - probability
  - distributions
date: 2026-08-30
---

# 12 — Random Variables and Distributions

> **Ek line mein:** Random variable ek function hai jo outcomes ko numbers mein badalta hai. Distribution batati hai wo numbers kaise spread hain.

---

## 1. Random Variable

$$X: S \rightarrow \mathbb{R}$$

**Example:** Do coins uchaalo, $X$ = kitne heads aaye. $X \in \{0, 1, 2\}$.

### Types

| Type | Values | Example |
|---|---|---|
| **Discrete** | Countable | Dice roll, kitne customers aaye |
| **Continuous** | Uncountable (range) | Height, temperature, time |

---

## 2. PMF, PDF, CDF ⭐

### PMF (Probability Mass Function) — discrete
$$p(x) = P(X = x)$$
Conditions: $p(x) \geq 0$, $\sum p(x) = 1$

### PDF (Probability Density Function) — continuous
$$P(a \leq X \leq b) = \int_a^b f(x)dx$$
Conditions: $f(x) \geq 0$, $\int_{-\infty}^{\infty}f(x)dx = 1$

**⚠️ Bahut important:** Continuous mein $P(X = x) = 0$ **hamesha**! Density probability nahi hai — sirf area probability hai. Isiliye PDF ki value 1 se zyada bhi ho sakti hai.

### CDF (Cumulative Distribution Function)
$$F(x) = P(X \leq x)$$
Hamesha non-decreasing, $F(-\infty)=0$, $F(\infty)=1$.

```python
from scipy import stats
import numpy as np

stats.norm.pdf(0, loc=0, scale=1)      # density at 0
stats.norm.cdf(0, loc=0, scale=1)      # 0.5
stats.norm.ppf(0.975)                  # inverse CDF → 1.96 ⭐
stats.norm.rvs(size=1000)              # random samples
```

**`ppf` (percent point function)** = inverse CDF. Confidence intervals mein use hota hai.

---

## 3. Expectation & Variance ⭐

### Expected Value (mean)
$$E[X] = \sum x\,p(x) \quad\text{(discrete)}, \qquad E[X] = \int xf(x)dx \quad\text{(continuous)}$$

**Properties:**
- $E[aX+b] = aE[X]+b$
- $E[X+Y] = E[X]+E[Y]$ (hamesha, independence ki zaroorat nahi!)
- $E[XY] = E[X]E[Y]$ **sirf agar independent hain**

### Variance
$$\text{Var}(X) = E[(X-\mu)^2] = E[X^2] - (E[X])^2$$

**Properties:**
- $\text{Var}(aX+b) = a^2\text{Var}(X)$ (constant $b$ ka koi asar nahi!)
- $\text{Var}(X+Y) = \text{Var}(X)+\text{Var}(Y)$ **sirf agar independent**
- General: $\text{Var}(X+Y) = \text{Var}(X)+\text{Var}(Y)+2\text{Cov}(X,Y)$

---

## 4. Discrete Distributions ⭐

### 4.1 Bernoulli
Ek trial, do outcomes (success/failure).

$$P(X=1)=p, \quad P(X=0)=1-p$$
$$E[X]=p, \quad \text{Var}(X)=p(1-p)$$

**ML mein:** Binary classification ka target. Logistic regression Bernoulli likelihood maximize karta hai.

```python
stats.bernoulli.rvs(p=0.3, size=10)
```

### 4.2 Binomial ⭐
$n$ independent Bernoulli trials mein kitne successes.

$$P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$$
$$E[X]=np, \quad \text{Var}(X)=np(1-p)$$

```python
stats.binom.pmf(k=3, n=10, p=0.5)      # exactly 3 heads in 10 flips
stats.binom.cdf(k=3, n=10, p=0.5)      # 3 ya usse kam
1 - stats.binom.cdf(k=3, n=10, p=0.5)  # 3 se zyada
```

### 4.3 Poisson ⭐
Fixed interval mein kitne events (rare events).

$$P(X=k) = \frac{\lambda^ke^{-\lambda}}{k!}$$
$$E[X] = \text{Var}(X) = \lambda$$

**Khaas baat:** mean = variance.

**Examples:** Ghante mein kitne customers, page pe kitni typos, server pe kitne requests.

```python
stats.poisson.pmf(k=3, mu=2)
stats.poisson.rvs(mu=2, size=1000)
```

**Poisson Regression:** count data predict karne ke liye (GLM).

### 4.4 Geometric
Pehle success tak kitne trials.
$$P(X=k) = (1-p)^{k-1}p, \quad E[X]=1/p$$

### 4.5 Multinomial
Binomial ka generalization — do se zyada categories.

**ML mein:** Multinomial Naive Bayes, softmax output.

---

## 5. Continuous Distributions ⭐

### 5.1 Uniform
$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$
$$E[X]=\frac{a+b}{2}, \quad \text{Var}(X)=\frac{(b-a)^2}{12}$$

**ML mein:** Random initialization, hyperparameter search.

```python
stats.uniform.rvs(loc=0, scale=1, size=100)
np.random.uniform(0, 1, 100)
```

### 5.2 Normal (Gaussian) ⭐⭐⭐

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

$$X \sim N(\mu, \sigma^2)$$

**Properties:**
- Symmetric bell curve
- mean = median = mode
- Sirf 2 parameters se poori tarah define ho jaati hai

### Empirical Rule (68-95-99.7) ⭐
| Range | Data |
|---|---|
| $\mu \pm 1\sigma$ | ~68% |
| $\mu \pm 2\sigma$ | ~95% |
| $\mu \pm 3\sigma$ | ~99.7% |

**Isiliye Z-score > 3 outlier maana jaata hai.**

### Standard Normal
$$Z = \frac{X-\mu}{\sigma} \sim N(0,1)$$

```python
stats.norm.pdf(0)                   # 0.3989
stats.norm.cdf(1.96)                # 0.975
stats.norm.ppf(0.975)               # 1.96 ⭐
stats.norm.interval(0.95)           # (-1.96, 1.96)

# Empirical rule verify
for k in [1,2,3]:
    p = stats.norm.cdf(k) - stats.norm.cdf(-k)
    print(f"±{k}σ: {p:.4f}")
```

### Normal itni common kyun hai? ⭐
**Central Limit Theorem** — bahut se independent random effects ka sum hamesha approximately normal hota hai. Height, measurement errors, noise — sab kai chhote factors ka sum hain.

### ML mein Normal distribution kahan?
- Linear regression ka noise assumption
- Weight initialization (Xavier/He)
- Gaussian Naive Bayes
- Batch Normalization
- Gaussian Mixture Models
- VAE ka latent space

### 5.3 Exponential
Events ke beech ka waqt.
$$f(x) = \lambda e^{-\lambda x}, \quad E[X]=1/\lambda$$

**Memoryless property:** $P(X>s+t|X>s) = P(X>t)$

### 5.4 Log-Normal
Agar $\log X$ normal hai. **Right-skewed.**

**Examples:** Income, house prices, stock prices, city populations.

```python
data = np.random.lognormal(0, 1, 1000)
np.log(data)      # ab normal hai ✅
```

**Isiliye skewed features pe log transform kaam karta hai.**

### 5.5 Student's t-Distribution
Normal jaisi lekin **heavier tails**. Chhote samples mein use hoti hai jab population $\sigma$ pata nahi.

$df \to \infty$ pe normal ban jaati hai.

```python
stats.t.ppf(0.975, df=10)     # 2.228 (normal ka 1.96 se bada)
```

### 5.6 Chi-Square
Standard normals ke squares ka sum. Categorical tests aur variance tests mein.

### 5.7 Beta
$[0,1]$ pe defined — **probabilities ki probability** model karne ke liye.

**ML mein:** Bayesian A/B testing, Thompson sampling, conjugate prior for Bernoulli.

---

## 6. Central Limit Theorem ⭐⭐⭐

**Statement:** Population ki distribution jo bhi ho, **sample means** ki distribution normal hogi (agar $n$ kaafi bada ho).

$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{as } n \to \infty$$

**Standard Error:** $SE = \frac{\sigma}{\sqrt{n}}$

**Rule of thumb:** $n \geq 30$ kaafi hota hai (skewed data mein zyada chahiye).

### Demonstration ⭐

```python
import numpy as np
import matplotlib.pyplot as plt

# Bilkul non-normal population (exponential)
population = np.random.exponential(scale=2, size=100000)

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
ax[0].hist(population, bins=50, color='salmon', edgecolor='k')
ax[0].set_title('Population (Exponential)\nBilkul normal nahi!')

for i, n in enumerate([2, 10, 50], start=1):
    means = [np.random.choice(population, n).mean() for _ in range(2000)]
    ax[i].hist(means, bins=40, color='skyblue', edgecolor='k')
    ax[i].set_title(f'Sample means (n={n})\nmean={np.mean(means):.2f}, '
                    f'se={np.std(means):.3f}')
plt.tight_layout(); plt.show()

print("Theoretical SE (n=50):", 2/np.sqrt(50))
```

**Kyun ye itna important hai?** CLT ki wajah se hum:
- Confidence intervals bana sakte hain
- Hypothesis tests kar sakte hain
- Sample se population ke baare mein infer kar sakte hain

**ML connection:** Mini-batch gradients asal mein full gradient ke sample means hain. CLT ki wajah se wo unbiased estimates hain, aur unka noise $1/\sqrt{\text{batch size}}$ ke hisaab se kam hota hai.

---

## 7. Q-Q Plot — normality check ⭐

```python
from scipy import stats
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

normal_data = np.random.normal(0, 1, 500)
skewed_data = np.random.exponential(2, 500)
heavy_tails = np.random.standard_t(3, 500)

for a, d, t in zip(ax, [normal_data, skewed_data, heavy_tails],
                   ['Normal ✅', 'Right-skewed', 'Heavy tails']):
    stats.probplot(d, dist='norm', plot=a)
    a.set_title(t)
plt.tight_layout(); plt.show()
```

**Kaise padhein?** Points diagonal line pe hon → normal. Curve upar/neeche → skewed. S-shape → heavy/light tails.

### Statistical normality tests
```python
stats.shapiro(data)              # Shapiro-Wilk (n < 5000 ke liye best)
stats.normaltest(data)           # D'Agostino-Pearson
stats.kstest(data, 'norm')       # Kolmogorov-Smirnov
stats.anderson(data, 'norm')     # Anderson-Darling
```

**⚠️ Warning:** Bade samples mein ye tests hamesha "not normal" bolenge (chhoti si deviation bhi significant ban jaati hai). **Q-Q plot pe zyada bharosa karo.**

---

## 8. Distribution kaise pehchanein? ⭐

```python
def identify_distribution(data):
    """Best-fit distribution dhoondo"""
    distributions = ['norm', 'expon', 'lognorm', 'gamma', 'beta', 'uniform']
    results = []
    for name in distributions:
        try:
            dist = getattr(stats, name)
            params = dist.fit(data)
            D, p = stats.kstest(data, name, args=params)
            results.append((name, D, p))
        except Exception:
            continue
    results.sort(key=lambda r: r[1])       # kam D = better fit
    print(f"{'Distribution':<12} {'KS stat':<10} {'p-value'}")
    for name, D, p in results:
        print(f"{name:<12} {D:<10.4f} {p:.4f}")
    return results[0][0]

data = np.random.lognormal(0, 0.5, 1000)
print("\nBest fit:", identify_distribution(data))
```

---

## 9. Joint, Marginal, Conditional Distributions

$$P(X,Y) \quad\text{joint}$$
$$P(X) = \sum_y P(X,y) \quad\text{marginal (Y ko "sum out" kar do)}$$
$$P(X|Y) = \frac{P(X,Y)}{P(Y)} \quad\text{conditional}$$

### Multivariate Normal ⭐
$$\mathbf{X} \sim N(\boldsymbol{\mu}, \Sigma)$$

$\Sigma$ = covariance matrix — features ke beech ka relation batata hai.

```python
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]        # correlated features
samples = np.random.multivariate_normal(mean, cov, 1000)

plt.scatter(samples[:,0], samples[:,1], alpha=0.4, s=10)
plt.gca().set_aspect('equal'); plt.title('Bivariate Normal (ρ = 0.8)')
plt.show()
```

**ML mein:** GMM, LDA/QDA, Gaussian Processes, Kalman filters.

---

## 10. Sampling & Random Number Generation

```python
rng = np.random.default_rng(42)     # modern API ⭐

rng.random(5)                        # uniform [0,1)
rng.integers(1, 7, 10)               # dice rolls
rng.normal(0, 1, 100)                # normal
rng.choice([1,2,3], size=10, p=[0.5, 0.3, 0.2])   # weighted
rng.permutation(10)                  # shuffle
rng.shuffle(arr)                     # in-place
```

### Bootstrap ⭐ (resampling technique)
```python
def bootstrap_ci(data, statistic=np.mean, n_boot=10000, ci=95):
    boots = [statistic(np.random.choice(data, len(data), replace=True))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [(100-ci)/2, 100-(100-ci)/2])
    return statistic(data), (lo, hi)

data = np.random.normal(50, 10, 100)
est, (lo, hi) = bootstrap_ci(data)
print(f"Mean = {est:.2f}, 95% CI = [{lo:.2f}, {hi:.2f}]")
```

**Fayda:** Koi distributional assumption nahi chahiye. Bagging (Random Forest) isi pe based hai.

---

## 11. ML mein distributions kahan use hote hain

| Distribution | ML Application |
|---|---|
| **Bernoulli** | Binary classification, dropout |
| **Binomial** | A/B testing, conversion rates |
| **Multinomial** | Multi-class output, text (bag of words) |
| **Poisson** | Count prediction, Poisson regression |
| **Normal** | Noise assumption, weight init, BatchNorm, GMM |
| **Log-normal** | Skewed features (income, prices) |
| **Uniform** | Random init, hyperparameter search |
| **Exponential** | Survival analysis, time-to-event |
| **Beta** | Bayesian A/B testing, Thompson sampling |
| **Dirichlet** | LDA topic modelling |
| **t-distribution** | Small sample inference, robust regression |
| **Chi-square** | Feature selection, goodness of fit |

---

## 12. Weight Initialization (distributions ka practical use) ⭐

```python
def xavier_init(n_in, n_out):
    """Sigmoid/tanh ke liye"""
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

def he_init(n_in, n_out):
    """ReLU ke liye ⭐"""
    return np.random.normal(0, np.sqrt(2/n_in), (n_in, n_out))
```

**Kyun?** Variance ko layers ke across constant rakhna hai — warna signal ya toh vanish ho jayega ya explode.

---

## 13. Interview Questions

1. **PDF aur PMF mein farq?** → PMF discrete (exact probability), PDF continuous (density, area = probability).
2. **Continuous distribution mein $P(X=x)=0$ kyun?** → single point ka area 0 hota hai.
3. **Central Limit Theorem samjhao.** → sample means normal hote hain, population ki distribution chahe jo ho.
4. **CLT ML mein kyun matter karta hai?** → confidence intervals, hypothesis testing, mini-batch gradients ka justification.
5. **Normal distribution itni common kyun hai?** → CLT — bahut se chhote independent effects ka sum.
6. **Empirical rule kya hai?** → 68-95-99.7.
7. **Poisson kab use karte hain?** → rare events ka count, fixed interval mein.
8. **Normality kaise check karein?** → Q-Q plot (best), Shapiro-Wilk, histogram.
9. **Skewed data ko normal kaise banayein?** → log, sqrt, Box-Cox, Yeo-Johnson.
10. **Bootstrap kya hai?** → replacement ke saath resampling; distribution assumptions ke bina CI banata hai.
11. **Standard error aur standard deviation mein farq?** → SD data ka spread; SE **estimate** (jaise mean) ka spread, $= \sigma/\sqrt{n}$.
12. **Weight initialization mein distribution kaise chunte hain?** → ReLU ke liye He (normal), tanh ke liye Xavier.

---

## Aage padho
→ [[13 Inferential Statistics]]
← [[11 Probability Basics]] | [[00 Maths for Machine Learning — MOC]]
