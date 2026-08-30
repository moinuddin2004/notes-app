---
title: Maths for ML — Inferential Statistics
tags:
  - maths
  - machine-learning
  - statistics
  - hypothesis-testing
date: 2026-08-30
---

# 13 — Inferential Statistics

> **Ek line mein:** Sample dekh kar population ke baare mein conclusions nikalna — aur ye batana ki hum kitne confident hain.

---

## 1. Sampling ⭐

### Sampling techniques

| Technique | Kaise |
|---|---|
| **Simple Random** | Har member ka barabar chance |
| **Stratified** ⭐ | Population ko groups mein baanto, har group se proportionally lo |
| **Systematic** | Har k-th member |
| **Cluster** | Poore groups randomly chuno |
| **Convenience** | Jo mil jaaye (biased ⚠️) |

**ML mein stratified sampling zaroori hai** jab classes imbalanced hon:

```python
from sklearn.model_selection import train_test_split, StratifiedKFold

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          stratify=y,      # ⭐ zaroori
                                          random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Sampling biases ⚠️
- **Selection bias** — sample representative nahi
- **Survivorship bias** — sirf "survivors" dekhna (WWII plane armor ka famous example)
- **Response bias** — sirf motivated log jawab dete hain

---

## 2. Point Estimate vs Interval Estimate

- **Point estimate:** ek single number ($\bar{x} = 52.3$)
- **Interval estimate:** ek range with confidence ($52.3 \pm 3.1$)

Point estimate akela adhoora hai — uncertainty nahi batata.

---

## 3. Confidence Interval ⭐⭐

$$CI = \bar{x} \pm z^*\frac{\sigma}{\sqrt{n}} \quad\text{($\sigma$ pata ho)}$$
$$CI = \bar{x} \pm t^*\frac{s}{\sqrt{n}} \quad\text{($\sigma$ pata na ho — common)}$$

### Common z-values
| Confidence | $z^*$ |
|---|---|
| 90% | 1.645 |
| **95%** | **1.96** ⭐ |
| 99% | 2.576 |

### Sahi interpretation ⭐⭐
**❌ Galat:** "95% chance hai ki true mean is interval mein hai."
**✅ Sahi:** "Agar hum ye experiment 100 baar repeat karein, toh unme se ~95 intervals mein true mean hoga."

True mean ek fixed number hai — wo random nahi hai. Interval random hai.

```python
from scipy import stats
import numpy as np

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)                      # s/√n
    margin = se * stats.t.ppf((1+confidence)/2, df=n-1)
    return mean, (mean-margin, mean+margin)

data = np.random.normal(50, 10, 100)
m, (lo, hi) = confidence_interval(data)
print(f"Mean = {m:.2f}, 95% CI = [{lo:.2f}, {hi:.2f}]")

# scipy shortcut
stats.t.interval(0.95, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data))
```

### CI ki width kis pe depend karti hai?
- $n$ badhao → CI **chhota** (accuracy $\sqrt{n}$ ke hisaab se badhti hai)
- Confidence badhao (95% → 99%) → CI **bada**
- Variance zyada → CI **bada**

---

## 4. Hypothesis Testing ⭐⭐

### Framework

**Step 1 — Hypotheses banao**
- $H_0$ (**Null**): "koi effect nahi", status quo
- $H_1$ (**Alternative**): "effect hai", jo prove karna hai

**Step 2 — Significance level chuno**
$\alpha = 0.05$ (typical). Ye Type I error ka acceptable rate hai.

**Step 3 — Test statistic calculate karo**

**Step 4 — p-value nikaalo**

**Step 5 — Decision**
- $p < \alpha$ → **Reject $H_0$** (result significant hai)
- $p \geq \alpha$ → **Fail to reject $H_0$**

**⚠️ Note:** Hum kabhi "$H_0$ accept" nahi karte — sirf "reject karne mein fail" hote hain. Evidence ki kami hona proof nahi hai.

---

## 5. p-value ⭐⭐

**Definition:** Agar $H_0$ **sach hai**, toh itna ya isse zyada extreme result milne ki probability.

### Sahi interpretation ⭐
- ❌ "$H_0$ sach hone ki probability"
- ❌ "Result important hai ya nahi"
- ✅ "$H_0$ sach hone par ye data dekhne ki probability"

**Chhota p-value** = data $H_0$ ke under bahut surprising hai → $H_0$ pe shak.

### p-hacking ⚠️
Agar tum 20 tests chalao, toh $\alpha=0.05$ pe average 1 test bina wajah "significant" nikal aayega.

**Fix — Multiple testing correction:**
```python
from statsmodels.stats.multitest import multipletests

p_values = [0.01, 0.04, 0.03, 0.20, 0.50]
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
print(p_corrected)

# FDR (kam conservative, zyada practical)
multipletests(p_values, alpha=0.05, method='fdr_bh')
```

---

## 6. Type I & Type II Errors ⭐⭐

|  | $H_0$ sach hai | $H_0$ galat hai |
|---|---|---|
| **Reject $H_0$** | **Type I Error ($\alpha$)** ❌ False Positive | ✅ Correct |
| **Fail to reject** | ✅ Correct | **Type II Error ($\beta$)** ❌ False Negative |

- **Type I** = jab kuch nahi tha tab bhi "effect hai" bol diya
- **Type II** = effect tha lekin miss kar diya

### Power
$$\text{Power} = 1 - \beta$$
Real effect detect karne ki probability. Typical target: **0.80**.

**Power kaise badhaye?** Sample size badhao (sabse effective), $\alpha$ badhao (trade-off), variance kam karo.

```python
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()
n = analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05)
print(f"Har group mein chahiye: {n:.0f} samples")
```

### ML connection ⭐
Confusion matrix mein:
- Type I error = **False Positive** → Precision affect karta hai
- Type II error = **False Negative** → Recall affect karta hai

**Kaunsa zyada bura hai?** Depends:
- Cancer screening → FN bahut bura (Recall optimize karo)
- Spam filter → FP bura (Precision optimize karo)

---

## 7. Common Statistical Tests ⭐⭐

### Test chunne ka decision tree

```
Data ka type?
│
├─ NUMERICAL
│   ├─ 1 sample vs known value        → One-sample t-test
│   ├─ 2 independent groups           → Independent t-test (ya Mann-Whitney)
│   ├─ 2 paired groups (before/after) → Paired t-test (ya Wilcoxon)
│   ├─ 3+ groups                      → ANOVA (ya Kruskal-Wallis)
│   └─ 2 numerical variables ka relation → Correlation test
│
└─ CATEGORICAL
    ├─ 1 variable ki distribution     → Chi-square goodness of fit
    ├─ 2 variables ka association     → Chi-square test of independence ⭐
    └─ 2 proportions                  → Z-test for proportions
```

### 7.1 One-sample t-test
"Kya sample mean ek particular value se alag hai?"

```python
stats.ttest_1samp(data, popmean=50)
```

### 7.2 Independent two-sample t-test ⭐
"Kya do groups ke means alag hain?"

```python
group_a = np.random.normal(50, 10, 100)
group_b = np.random.normal(55, 10, 100)

t, p = stats.ttest_ind(group_a, group_b)
print(f"t = {t:.4f}, p = {p:.4f}")

# Agar variances equal nahi (Welch's t-test — safer default)
stats.ttest_ind(group_a, group_b, equal_var=False)
```

**Assumptions:** normality, independence, (equal variance for standard version).

### 7.3 Paired t-test
Same subjects, do measurements (before/after).

```python
stats.ttest_rel(before, after)
```

### 7.4 ANOVA ⭐
3+ groups compare karne ke liye.

```python
stats.f_oneway(g1, g2, g3)
```

**Multiple t-tests ki jagah ANOVA kyun?** 3 groups mein 3 t-tests karne se Type I error rate 5% se badh kar ~14% ho jaata hai.

**Significant aane par:** post-hoc test (Tukey HSD) se pata karo kaunse pairs alag hain.

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
pairwise_tukeyhsd(values, groups)
```

### 7.5 Chi-Square Test ⭐⭐
Categorical variables ke liye.

```python
import pandas as pd

# Test of independence
contingency = pd.crosstab(df['gender'], df['purchased'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"chi2 = {chi2:.4f}, p = {p:.4f}")

# Goodness of fit
observed = [30, 25, 20, 25]
expected = [25, 25, 25, 25]
stats.chisquare(observed, expected)
```

**ML mein:** Categorical feature selection.
```python
from sklearn.feature_selection import SelectKBest, chi2
SelectKBest(chi2, k=10).fit_transform(X, y)
```

### 7.6 Non-parametric tests (jab normality assume na kar sakein)

| Parametric | Non-parametric |
|---|---|
| One-sample t-test | Wilcoxon signed-rank |
| Independent t-test | **Mann-Whitney U** |
| Paired t-test | Wilcoxon signed-rank |
| ANOVA | **Kruskal-Wallis** |
| Pearson correlation | **Spearman correlation** |

```python
stats.mannwhitneyu(g1, g2)
stats.kruskal(g1, g2, g3)
stats.wilcoxon(before, after)
```

---

## 8. Effect Size ⭐

**p-value batata hai ki effect hai — effect size batata hai wo kitna bada hai.**

Bade samples mein bilkul trivial differences bhi "significant" nikal aate hain. Isliye effect size hamesha report karo.

### Cohen's d
$$d = \frac{\bar{x_1}-\bar{x_2}}{s_{pooled}}$$

| $|d|$ | Effect |
|---|---|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

```python
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x,ddof=1) + (ny-1)*np.var(y,ddof=1))
                         / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled_std
```

---

## 9. A/B Testing ⭐ (industry mein sabse practical)

```python
def ab_test(control_conv, control_n, treat_conv, treat_n, alpha=0.05):
    p1 = control_conv / control_n
    p2 = treat_conv / treat_n
    p_pool = (control_conv + treat_conv) / (control_n + treat_n)
    se = np.sqrt(p_pool*(1-p_pool)*(1/control_n + 1/treat_n))
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    lift = (p2 - p1) / p1 * 100
    margin = stats.norm.ppf(1-alpha/2) * np.sqrt(p1*(1-p1)/control_n
                                                 + p2*(1-p2)/treat_n)

    print(f"Control  : {p1:.4f} ({control_conv}/{control_n})")
    print(f"Treatment: {p2:.4f} ({treat_conv}/{treat_n})")
    print(f"Lift     : {lift:+.2f}%")
    print(f"Z        : {z:.4f}")
    print(f"p-value  : {p_value:.4f}")
    print(f"95% CI on difference: [{(p2-p1)-margin:.4f}, {(p2-p1)+margin:.4f}]")
    print("→", "✅ Significant" if p_value < alpha else "❌ Not significant")

ab_test(control_conv=200, control_n=2000, treat_conv=250, treat_n=2000)
```

### A/B testing ki galtiyaan ⚠️
1. **Peeking** — result dekh kar jaldi rok dena (Type I error badha deta hai)
2. **Sample size pehle se decide na karna**
3. **Multiple metrics test karna** bina correction ke
4. **Novelty effect** ignore karna
5. **Practical significance** ignore karna (0.01% lift significant ho sakta hai lekin useless)

---

## 10. Statistics + ML: Model Evaluation ⭐

### Cross-validation ka statistical view
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=10)
mean, (lo, hi) = confidence_interval(scores)
print(f"Accuracy: {mean:.4f}, 95% CI: [{lo:.4f}, {hi:.4f}]")
```

### Do models compare karna ⭐
```python
from sklearn.model_selection import cross_val_score

s1 = cross_val_score(model1, X, y, cv=10)
s2 = cross_val_score(model2, X, y, cv=10)

# Paired t-test — same folds pe evaluate hue hain
t, p = stats.ttest_rel(s1, s2)
print(f"Model 1: {s1.mean():.4f} ± {s1.std():.4f}")
print(f"Model 2: {s2.mean():.4f} ± {s2.std():.4f}")
print(f"p-value: {p:.4f}")
print("→", "Farq significant hai" if p < 0.05 else "Farq significant nahi")
```

**Ye important kyun?** Model A ka 92% aur Model B ka 91.5% — kya B actually behsonestly better hai ya sirf random variation? Statistical test se pata chalta hai.

### Bias-Variance Tradeoff ⭐
$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

| | Bias | Variance |
|---|---|---|
| Matlab | Model kitna simplify kar raha hai | Training data badalne se prediction kitni badalti hai |
| Zyada hone par | **Underfitting** | **Overfitting** |
| Example | Linear model on non-linear data | Deep tree without pruning |
| Fix | Complex model, zyada features | Regularization, zyada data, ensemble |

---

## 11. Confusion Matrix aur Metrics ⭐

```
                Predicted
              Neg      Pos
Actual  Neg   TN       FP      ← Type I error
        Pos   FN       TP      ← Type II error
```

$$\text{Accuracy} = \frac{TP+TN}{\text{Total}}$$
$$\text{Precision} = \frac{TP}{TP+FP} \quad\text{(predicted positives mein kitne sahi)}$$
$$\text{Recall} = \frac{TP}{TP+FN} \quad\text{(actual positives mein kitne pakde)}$$
$$F1 = \frac{2 \cdot P \cdot R}{P+R} \quad\text{(harmonic mean)}$$

**Imbalanced data mein accuracy dhoka deti hai.** Agar 99% data ek class ka hai, toh "hamesha majority predict karo" model 99% accuracy dega — lekin bilkul bekaar hai.

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_proba))
```

---

## 12. Interview Questions

1. **p-value kya hai — exact definition?** → $H_0$ sach hone par itna ya isse zyada extreme data dekhne ki probability.
2. **Confidence interval ka sahi interpretation?** → repeated sampling mein 95% intervals true parameter ko contain karenge.
3. **Type I aur Type II error mein farq? ML metrics se kya relation?** → FP/Precision aur FN/Recall.
4. **Statistical power kya hai aur kaise badhaye?** → real effect detect karne ki probability; sample size badhao.
5. **3 groups ke liye multiple t-tests ki jagah ANOVA kyun?** → multiple comparisons se Type I error rate inflate ho jaata hai.
6. **p-hacking kya hai?**
7. **Kab parametric aur kab non-parametric test?** → normality assumption satisfy hoti hai ya nahi; sample size.
8. **Effect size kyun report karna chahiye?** → p-value significance batata hai, magnitude nahi.
9. **Do models statistically compare kaise karein?** → same folds pe cross-validation + paired t-test.
10. **Stratified sampling kab zaroori hai?** → imbalanced classes mein, taaki har fold mein class proportion maintain rahe.
11. **Bias-variance tradeoff samjhao.**
12. **Imbalanced data mein accuracy kyun mislead karti hai?**

---

## 🎓 Poori Series ka Summary

Ab tumne ye sab cover kar liya:

**Linear Algebra** → data ko represent aur transform karna
[[01 Vectors]] · [[02 Lines Planes and Hyperplanes]] · [[03 Matrices]] · [[04 Determinant Inverse Rank]] · [[05 Linear Transformations]] · [[06 Eigenvalues Eigenvectors PCA SVD]]

**Calculus** → models ko train karna
[[07 Functions and Derivatives]] · [[08 Multivariable Calculus and Gradient]] · [[09 Optimization and Gradient Descent]]

**Probability & Statistics** → uncertainty handle karna aur results validate karna
[[10 Descriptive Statistics]] · [[11 Probability Basics]] · [[12 Random Variables and Distributions]] · [[13 Inferential Statistics]]

**Agla kadam:** Ab ML algorithms padho — har algorithm mein ye maths dikhega. CampusX ka "100 Days of Machine Learning" playlist next logical step hai.

---

← [[12 Random Variables and Distributions]] | [[00 Maths for Machine Learning — MOC]]
