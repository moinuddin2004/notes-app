---
title: Maths for ML — Probability Basics
tags:
  - maths
  - machine-learning
  - probability
  - bayes
date: 2026-08-30
---

# 11 — Probability Basics

> **Ek line mein:** Probability uncertainty ki language hai. ML models predictions dete hain — probability batati hai ki un predictions pe kitna bharosa karein.

---

## 1. Basic Terminology

| Term | Matlab | Example |
|---|---|---|
| **Random Experiment** | Jiska outcome uncertain ho | Dice throw |
| **Sample Space ($S$)** | Saare possible outcomes | $\{1,2,3,4,5,6\}$ |
| **Event ($E$)** | Sample space ka subset | Even number = $\{2,4,6\}$ |
| **Outcome** | Ek single result | 3 |

### Probability ki definition
$$P(E) = \frac{\text{favourable outcomes}}{\text{total outcomes}}$$

### Axioms (Kolmogorov)
1. $0 \leq P(E) \leq 1$
2. $P(S) = 1$
3. Mutually exclusive events ke liye: $P(A \cup B) = P(A) + P(B)$

---

## 2. Types of Events

| Type | Definition |
|---|---|
| **Mutually Exclusive** | Ek saath nahi ho sakte: $P(A\cap B)=0$ |
| **Independent** | Ek doosre ko affect nahi karte: $P(A\cap B)=P(A)P(B)$ |
| **Exhaustive** | Milkar poora sample space |
| **Complementary** | $P(A') = 1 - P(A)$ |

**⚠️ Common confusion:** Mutually exclusive ≠ Independent. Agar do events mutually exclusive hain toh wo **dependent** hain (ek hua toh doosra bilkul nahi ho sakta).

---

## 3. Probability Rules ⭐

### Addition Rule
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

Mutually exclusive ho toh: $P(A\cup B) = P(A)+P(B)$

### Multiplication Rule
$$P(A \cap B) = P(A)P(B|A)$$

Independent ho toh: $P(A\cap B) = P(A)P(B)$

### Complement
$$P(A') = 1 - P(A)$$

**Trick:** "Kam se kam ek" wale problems mein hamesha complement use karo.

Example: 3 coins mein kam se kam ek head?
$$P(\text{at least 1 head}) = 1 - P(\text{no head}) = 1 - (1/2)^3 = 7/8$$

---

## 4. Conditional Probability ⭐⭐

$$P(A|B) = \frac{P(A\cap B)}{P(B)}, \quad P(B) > 0$$

"$B$ ho chuka hai — ab $A$ ki probability kya hai?"

### Intuition
Conditioning ka matlab hai **sample space ko chhota karna**. Ab hum sirf $B$ ke andar dekh rahe hain.

### Example
52 cards ka deck. Ek card nikala, wo red hai. King hone ki probability?
$$P(\text{King}|\text{Red}) = \frac{P(\text{King}\cap\text{Red})}{P(\text{Red})} = \frac{2/52}{26/52} = \frac{2}{26} = \frac{1}{13}$$

### Chain Rule
$$P(A\cap B\cap C) = P(A)P(B|A)P(C|A\cap B)$$

**Ye language models ka base hai:**
$$P(w_1,w_2,\dots,w_n) = \prod_i P(w_i | w_1,\dots,w_{i-1})$$

---

## 5. Law of Total Probability ⭐

Agar $B_1, B_2, \dots, B_n$ sample space ko partition karte hain:

$$P(A) = \sum_i P(A|B_i)P(B_i)$$

**Intuition:** Sab possible "raaste" jodo.

---

## 6. Bayes' Theorem ⭐⭐⭐

$$\boxed{P(A|B) = \frac{P(B|A)P(A)}{P(B)}}$$

Expanded form:
$$P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|A')P(A')}$$

### Terminology ⭐

| Term | Naam | Matlab |
|---|---|---|
| $P(A)$ | **Prior** | Evidence dekhne se pehle ka belief |
| $P(B\|A)$ | **Likelihood** | Agar $A$ sach hai toh $B$ dekhne ka chance |
| $P(B)$ | **Evidence/Marginal** | $B$ ki overall probability |
| $P(A\|B)$ | **Posterior** | Evidence ke baad updated belief |

**Ek line mein:** posterior ∝ likelihood × prior

---

## 7. Bayes ka classic example — Medical Test ⭐

**Given:**
- Disease 1% logon ko hai: $P(D) = 0.01$
- Test 99% accurate hai: $P(+|D) = 0.99$, $P(-|D') = 0.99$
- Test positive aaya. Actually disease hone ki probability?

**Zyadatar log kehte hain 99%. Sahi jawab: sirf 50%!**

$$P(+) = P(+|D)P(D) + P(+|D')P(D')$$
$$= (0.99)(0.01) + (0.01)(0.99) = 0.0099 + 0.0099 = 0.0198$$

$$P(D|+) = \frac{(0.99)(0.01)}{0.0198} = \frac{0.0099}{0.0198} = 0.5$$

```python
def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood*prior + false_positive_rate*(1-prior)
    return likelihood*prior/evidence

bayes(prior=0.01, likelihood=0.99, false_positive_rate=0.01)     # 0.5
```

**Sabak:** Agar **prior bahut chhota** hai (rare disease), toh accurate test bhi bahut se false positives dega. Isiliye rare events ke liye base rate matter karta hai.

**ML connection:** Imbalanced datasets mein 99% accuracy ka matlab kuch nahi — model sirf majority class predict kar raha ho sakta hai.

### Prior badalne ka asar
```python
for p in [0.001, 0.01, 0.1, 0.5]:
    print(f"Prior {p:5.3f} → Posterior {bayes(p, 0.99, 0.01):.3f}")
# 0.001 → 0.090
# 0.010 → 0.500
# 0.100 → 0.917
# 0.500 → 0.990
```

---

## 8. Naive Bayes Classifier ⭐⭐

$$P(y|x_1,\dots,x_n) = \frac{P(y)\prod_i P(x_i|y)}{P(x_1,\dots,x_n)}$$

**"Naive" kyun?** Kyunki maan leta hai ki saare features **independent** hain given the class. Ye assumption real world mein aksar galat hoti hai — lekin classifier phir bhi acha kaam karta hai.

### Prediction
$$\hat{y} = \arg\max_y P(y)\prod_i P(x_i|y)$$

**Practically log use karo** (underflow se bachne ke liye):
$$\hat{y} = \arg\max_y \left[\log P(y) + \sum_i \log P(x_i|y)\right]$$

### Scratch implementation ⭐

```python
import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean, self.var, self.prior = {}, {}, {}
        for c in self.classes:
            Xc = X[y == c]
            self.mean[c]  = Xc.mean(axis=0)
            self.var[c]   = Xc.var(axis=0) + 1e-9      # zero division se bachao
            self.prior[c] = len(Xc) / len(X)
        return self

    def _log_gaussian(self, x, mean, var):
        return -0.5*np.log(2*np.pi*var) - (x-mean)**2/(2*var)

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = [
                np.log(self.prior[c]) + self._log_gaussian(x, self.mean[c], self.var[c]).sum()
                for c in self.classes
            ]
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds)

# Test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

nb = GaussianNaiveBayes().fit(Xtr, ytr)
print("Accuracy:", (nb.predict(Xte) == yte).mean())
```

### Naive Bayes ke types

| Type | Kab use |
|---|---|
| **Gaussian NB** | Continuous features (normally distributed) |
| **Multinomial NB** | Word counts (text classification) ⭐ |
| **Bernoulli NB** | Binary features (word present/absent) |
| **Complement NB** | Imbalanced text data |

### Laplace Smoothing ⭐
Agar training mein koi word class ke saath kabhi nahi aaya, toh $P(x_i|y)=0$ → poora product 0 ho jayega.

**Fix:**
$$P(x_i|y) = \frac{\text{count}(x_i, y) + \alpha}{\text{count}(y) + \alpha n}$$

$\alpha = 1$ ko Laplace smoothing kehte hain.

```python
from sklearn.naive_bayes import MultinomialNB
MultinomialNB(alpha=1.0)     # default already 1.0
```

---

## 9. Independence ⭐

### Independent events
$$P(A\cap B) = P(A)P(B) \iff P(A|B) = P(A)$$

### Conditional Independence
$$P(A\cap B|C) = P(A|C)P(B|C)$$

**"$C$ pata hone ke baad, $A$ aur $B$ independent ho jaate hain."**

Ye Naive Bayes ka core assumption hai, aur Bayesian networks ka base bhi.

**Example:** Shoe size aur reading ability children mein correlated hain. Lekin **age** pata ho jaaye toh wo conditionally independent ho jaate hain. Age confounding variable tha.

---

## 10. Probability Distributions ka introduction

Detail ke liye dekho [[12 Random Variables and Distributions]].

- **Joint:** $P(X, Y)$
- **Marginal:** $P(X) = \sum_y P(X, y)$
- **Conditional:** $P(X|Y) = P(X,Y)/P(Y)$

---

## 11. Odds aur Log-Odds ⭐

### Odds
$$\text{odds} = \frac{p}{1-p}$$

### Log-odds (Logit) ⭐
$$\text{logit}(p) = \log\frac{p}{1-p}$$

**Logistic Regression ka asli formula:**
$$\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \mathbf{w}^T\mathbf{x} + b$$

Ye rearrange karo toh sigmoid mil jaata hai:
$$P(y=1|\mathbf{x}) = \frac{1}{1+e^{-(\mathbf{w}^T\mathbf{x}+b)}}$$

**Sigmoid asal mein logit ka inverse hai.** Isiliye logistic regression mein sigmoid aata hai — koi random choice nahi hai.

```python
def logit(p): return np.log(p/(1-p))
def sigmoid(z): return 1/(1+np.exp(-z))

p = 0.7
z = logit(p)
sigmoid(z)      # 0.7 — wapas mil gaya
```

---

## 12. Maximum Likelihood Estimation (MLE) ⭐⭐

**Idea:** Wo parameters chuno jinke under observed data ki probability maximum ho.

$$\hat{\theta}_{MLE} = \arg\max_\theta P(\text{data}|\theta) = \arg\max_\theta \prod_i P(x_i|\theta)$$

Log lena aasan hai:
$$\hat{\theta}_{MLE} = \arg\max_\theta \sum_i \log P(x_i|\theta)$$

### ML connection ⭐
- **Linear Regression** ka MSE = Gaussian noise assumption ke saath MLE
- **Logistic Regression** ka Cross-entropy = Bernoulli assumption ke saath MLE

**Yani loss functions randomly nahi chune gaye — wo negative log-likelihood hain.**

### MAP (Maximum A Posteriori)
$$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta|\text{data}) = \arg\max_\theta P(\text{data}|\theta)P(\theta)$$

Prior bhi shamil karta hai. **Regularization asal mein MAP estimation hai:**
- L2 regularization = Gaussian prior on weights
- L1 regularization = Laplace prior on weights

---

## 13. Entropy & Information Theory ⭐

### Entropy
$$H(X) = -\sum_i p_i\log p_i$$

Uncertainty ka measure. Maximum tab jab sab outcomes equally likely hon.

```python
def entropy(p):
    p = np.array(p)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

entropy([0.5, 0.5])       # 1.0 bit (maximum for 2 outcomes)
entropy([0.9, 0.1])       # 0.469 bits
entropy([1.0])            # 0.0 (koi uncertainty nahi)
```

### Cross Entropy ⭐
$$H(p,q) = -\sum_i p_i\log q_i$$

**Yahi classification ka loss function hai.** $p$ = true distribution, $q$ = predicted.

### KL Divergence
$$D_{KL}(p\|q) = \sum_i p_i\log\frac{p_i}{q_i} = H(p,q) - H(p)$$

Do distributions kitni alag hain. **Symmetric nahi hai:** $D_{KL}(p\|q) \neq D_{KL}(q\|p)$.

**Kahan use hota hai?** VAEs, t-SNE, model distillation, RLHF.

### Information Gain (Decision Trees) ⭐
$$IG = H(\text{parent}) - \sum_j \frac{n_j}{n}H(\text{child}_j)$$

Decision tree har split pe wo feature chunta hai jo maximum information gain de.

### Gini Impurity (alternative)
$$Gini = 1 - \sum_i p_i^2$$

Entropy jaisa hi, lekin log nahi hai toh compute karna faster. sklearn ka default.

---

## 14. Interview Questions

1. **Bayes' theorem samjhao aur ek real example do.**
2. **Prior, likelihood, posterior kya hain?**
3. **Naive Bayes "naive" kyun hai?** → features ko conditionally independent maanta hai.
4. **Assumption galat hone ke bawajood Naive Bayes kaam kyun karta hai?** → classification ke liye sirf **argmax** sahi hona chahiye, exact probabilities nahi.
5. **Laplace smoothing kya hai aur kyun chahiye?** → zero probability problem se bachne ke liye.
6. **Mutually exclusive aur independent mein farq?**
7. **Logistic Regression mein sigmoid kyun?** → log-odds linear hone se sigmoid nikalta hai.
8. **MLE kya hai? Loss functions se kya relation?** → zyadatar loss functions negative log-likelihood hain.
9. **Regularization ka Bayesian interpretation?** → MAP estimation with a prior (L2 = Gaussian, L1 = Laplace).
10. **Cross-entropy loss kya measure karta hai?** → predicted aur true distribution ke beech ka farq.
11. **KL divergence symmetric kyun nahi?**
12. **Medical test wala paradox samjhao.** → base rate fallacy — rare disease mein accurate test bhi 50% posterior de sakta hai.

---

## Aage padho
→ [[12 Random Variables and Distributions]]
← [[10 Descriptive Statistics]] | [[00 Maths for Machine Learning — MOC]]
