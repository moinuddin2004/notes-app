---
title: NumPy — Complete Notes
tags:
  - python
  - numpy
  - arrays
  - broadcasting
  - vectorization
  - data-science
  - campusx
  - notes
date: 2026-08-29
---

# NumPy — Complete Notes
### CampusX Session 13, 14, 15 (Numpy Fundamentals + Advanced Numpy + Numpy Tricks)

---

# SESSION 13 — NUMPY FUNDAMENTALS

## 1. NumPy hai kya?

NumPy = **Numerical Python**. Ye Python ki ek library hai jo **multi-dimensional arrays** aur unpe **fast mathematical operations** ke liye banayi gayi hai.

Core object ka naam hai **`ndarray`** (n-dimensional array).

```python
import numpy as np
```

### NumPy fast kyun hai? (ye interview mein poochte hain)

| Wajah | Explanation |
|---|---|
| **C mein likha hai** | NumPy ka core C language mein implement hua hai, Python loop ka overhead nahi lagta |
| **Contiguous memory** | Saara data memory mein ek continuous block mein rehta hai. Python list mein pointers hote hain jo alag alag jagah point karte hain |
| **Homogeneous data** | Ek array mein sab elements ka **same datatype** hota hai, isliye type-checking har element pe nahi karni padti |
| **Vectorization** | Loop C level pe chalta hai, Python level pe nahi |
| **SIMD** | Single Instruction Multiple Data — CPU ek hi instruction se kai values pe kaam karta hai |

**Yaad rakho:** Python list = flexible lekin slow. NumPy array = fixed type lekin bahut fast.

---

## 2. Array banane ke tareeqe

### 2.1 `np.array()` — list se array

```python
# 1D array (Vector)
a = np.array([1, 2, 3])
print(a)          # [1 2 3]

# 2D array (Matrix)
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]

# 3D array (Tensor)
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(c)
```

**Terminology (important):**
- 1D array → **Vector**
- 2D array → **Matrix**
- 3D ya usse zyada → **Tensor**

### 2.2 `dtype` — datatype specify karna

```python
np.array([1, 2, 3], dtype=float)      # [1. 2. 3.]
np.array([1, 2, 3], dtype=bool)       # [True True True]
np.array([1, 2, 3], dtype=complex)    # [1.+0.j 2.+0.j 3.+0.j]
np.array([1, 2, 3], dtype=np.int32)
```

Common dtypes: `int8, int16, int32, int64, float16, float32, float64, bool, complex, object, str_`

### 2.3 `np.arange()` — range jaisa

```python
np.arange(1, 11)          # [1 2 3 4 5 6 7 8 9 10]
np.arange(1, 11, 2)       # [1 3 5 7 9]   (step = 2)
np.arange(16)             # 0 se 15
```

### 2.4 `reshape()` — shape badalna

```python
np.arange(16).reshape(2, 2, 2, 2)
np.arange(12).reshape(3, 4)
np.arange(12).reshape(4, 3)
```

**Rule:** `rows × cols` = total elements hone chahiye. `np.arange(10).reshape(3,4)` → **Error**.

**`-1` ka jugaad:** ek dimension `-1` de do, NumPy khud calculate kar lega.
```python
np.arange(12).reshape(3, -1)   # (3,4) ban jayega
np.arange(12).reshape(-1, 2)   # (6,2)
```

### 2.5 `np.ones()` aur `np.zeros()`

```python
np.ones((3, 4))     # 3x4 ka array, sab 1
np.zeros((3, 4))    # 3x4 ka array, sab 0
np.full((3, 4), 7)  # sab 7
np.empty((3, 4))    # garbage values (fast, initialize nahi karta)
```

**Use case:** Machine Learning mein weights ko initialize karne ke liye.

### 2.6 `np.random` — random numbers

```python
np.random.random((3, 4))            # 0 se 1 ke beech uniform
np.random.randint(1, 100, (3, 4))   # 1 se 99 ke beech integers
np.random.rand(3, 4)                # same as random but shape args
np.random.randn(3, 4)               # standard normal distribution
np.random.uniform(1, 10, (3,4))     # uniform between 1 and 10
np.random.seed(42)                  # reproducibility ke liye
```

### 2.7 `np.linspace()` — linearly spaced

```python
np.linspace(-10, 10, 10)
# -10 se 10 tak, 10 equally-spaced numbers (dono ends INCLUDE)
np.linspace(-10, 10, 10, dtype=int)
```

**arange vs linspace ka farq:**
- `arange` → **step** batate ho, count NumPy nikaalta hai
- `linspace` → **count** batate ho, step NumPy nikaalta hai

### 2.8 `np.identity()` — identity matrix

```python
np.identity(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

np.eye(3, 4)     # non-square bhi bana sakte ho
np.eye(3, k=1)   # diagonal shift
```

---

## 3. Array Attributes (properties)

```python
a1 = np.arange(10)                 # 1D
a2 = np.arange(12, dtype=float).reshape(3, 4)   # 2D
a3 = np.arange(8).reshape(2, 2, 2)              # 3D
```

| Attribute | Kya batata hai | Example |
|---|---|---|
| `.ndim` | Kitne dimensions | `a3.ndim` → 3 |
| `.shape` | Har dimension ka size | `a2.shape` → (3, 4) |
| `.size` | Total elements | `a2.size` → 12 |
| `.itemsize` | Ek element kitne **bytes** ka | float64 → 8 |
| `.dtype` | Datatype | `a2.dtype` → float64 |
| `.nbytes` | Poore array ka total memory | size × itemsize |
| `.T` | Transpose | rows ↔ cols |

```python
print(a3.ndim)      # 3
print(a2.shape)     # (3, 4)
print(a2.size)      # 12
print(a2.itemsize)  # 8
print(a2.dtype)     # float64
```

---

## 4. Changing Datatype — `astype()`

```python
a3 = np.arange(8).reshape(2,2,2)
a3.astype(np.int32)
a3.astype(np.float64)
```

**Ye important kyun hai?** Memory bachane ke liye. Agar tumhare data mein sirf 0-255 ke numbers hain toh `int64` ki jagah `int8` use karo — **8x memory bachegi**. Bade datasets mein ye game-changer hai.

```python
# Example: 1 crore numbers
big = np.random.randint(0, 100, 10000000)
print(big.nbytes)                    # 80,000,000 bytes (~80 MB)
print(big.astype(np.int8).nbytes)    # 10,000,000 bytes (~10 MB)
```

---

## 5. Array Operations

Setup:
```python
a1 = np.arange(12).reshape(3, 4)
a2 = np.arange(12, 24).reshape(3, 4)
```

### 5.1 Scalar Operations (array + ek number)

**Arithmetic:**
```python
a1 * 2      # har element 2 se multiply
a1 + 2
a1 - 2
a1 / 2
a1 ** 2     # power
a1 % 2      # modulo
```

**Relational (comparison):**
```python
a2 == 15    # boolean array return karta hai
a2 > 15
a2 != 10
```

### 5.2 Vector Operations (array + array)

```python
a1 + a2     # element-wise addition
a1 - a2
a1 * a2     # element-wise multiplication (NOT matrix multiplication!)
a1 / a2
a1 ** a2
```

**⚠️ Bahut important:** `a1 * a2` **element-wise** hai. Matrix multiplication ke liye `a1 @ a2` ya `np.dot(a1, a2)` use karo.

**Condition:** dono arrays ka shape same hona chahiye (ya broadcasting-compatible — Session 14 mein).

---

## 6. Array Functions

Setup:
```python
a1 = np.random.random((3, 3))
a1 = np.round(a1 * 100)
```

### 6.1 max / min / sum / prod

```python
np.max(a1)              # poore array ka max
np.max(a1, axis=0)      # har COLUMN ka max
np.max(a1, axis=1)      # har ROW ka max

np.min(a1)
np.sum(a1)
np.prod(a1)
np.prod(a1, axis=0)
```

**`axis` samajhne ka trick:**
- `axis=0` → **neeche ki taraf** (rows collapse hoti hain) → **column-wise** result
- `axis=1` → **daayein ki taraf** (columns collapse hote hain) → **row-wise** result

### 6.2 Statistical functions

```python
np.mean(a1)                 # average
np.mean(a1, axis=0)
np.median(a1)               # beech ki value
np.std(a1)                  # standard deviation
np.var(a1)                  # variance
np.percentile(a1, 50)       # 50th percentile = median
```

### 6.3 Trigonometric functions

```python
np.sin(a1)
np.cos(a1)
np.tan(a1)
np.arcsin(a1)
```

### 6.4 Dot Product — `np.dot()`

Matrix multiplication.
```python
a2 = np.arange(12).reshape(3, 4)
a3 = np.arange(12, 24).reshape(4, 3)

np.dot(a2, a3)     # (3,4) @ (4,3) → (3,3)
a2 @ a3            # same cheez, naya syntax
```

**Rule:** Pehle matrix ke columns = doosre matrix ke rows.
`(m, n) @ (n, p) = (m, p)`

### 6.5 Log aur Exponent

```python
np.log(a1)      # natural log
np.log10(a1)
np.log2(a1)
np.exp(a1)      # e^x
```

**ML mein use:** log loss, exponential activation functions.

### 6.6 Rounding

```python
np.round(np.random.random((2,3)) * 100)
np.floor(np.random.random((2,3)) * 100)   # neeche wala integer
np.ceil(np.random.random((2,3)) * 100)    # upar wala integer
```

---

## 7. Indexing and Slicing

### 7.1 1D Array

```python
a1 = np.arange(10)     # [0 1 2 3 4 5 6 7 8 9]

a1[0]        # 0
a1[-1]       # 9  (last element)
a1[2:5]      # [2 3 4]
a1[2:5:2]    # [2 4]   (step)
a1[::2]      # [0 2 4 6 8]   (saare even index)
a1[::-1]     # reverse
```

### 7.2 2D Array

```python
a2 = np.arange(12).reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

a2[1, 2]        # 6   → row 1, col 2
a2[0, 0]        # 0
a2[2, 3]        # 11

a2[0, :]        # pehli poori row → [0 1 2 3]
a2[:, 2]        # teesra poora column → [2 6 10]
a2[1:3, 1:3]    # sub-matrix
a2[::2, ::3]    # [[0 3], [8 11]]
a2[::2, 1::2]   # [[1 3], [9 11]]
a2[1, ::3]      # [4 7]
a2[0:2, 1:]     # [[1 2 3], [5 6 7]]
```

**Format:** `a2[row_slice, col_slice]`

### 7.3 3D Array (Tensor)

```python
a3 = np.arange(27).reshape(3, 3, 3)

a3[1]           # doosra 2D matrix
a3[1, 0, 1]     # ek single number
a3[::2, 0, ::2] # complex slicing
a3[2, 1:, 1:]
a3[0, 1, :]
```

**Format:** `a3[matrix_no, row, col]`

---

## 8. Iterating

### 1D
```python
for i in a1:
    print(i)     # ek ek number
```

### 2D
```python
for i in a2:
    print(i)     # ek ek ROW milegi (poora array nahi, row)
```

### 3D
```python
for i in a3:
    print(i)     # ek ek 2D MATRIX milega
```

### Har element chahiye? → `np.nditer()`
```python
for i in np.nditer(a3):
    print(i)     # ek ek scalar element
```

---

## 9. Reshaping Operations

### 9.1 `reshape()`
```python
a2.reshape(4, 3)
a2.reshape(2, 6)
```

### 9.2 `Transpose` — rows aur cols swap
```python
np.transpose(a2)
a2.T              # shortcut
```

### 9.3 `ravel()` — flatten (kisi bhi array ko 1D banana)
```python
a3.ravel()       # 3D → 1D
a2.ravel()       # 2D → 1D
```

**`ravel()` vs `flatten()`:**
- `ravel()` → **view** deta hai (original array se juda hua, memory share karta hai)
- `flatten()` → **copy** deta hai (independent)

---

## 10. Stacking (arrays ko jodna)

```python
a4 = np.arange(12).reshape(3, 4)
a5 = np.arange(12, 24).reshape(3, 4)
```

### Horizontal Stacking
```python
np.hstack((a4, a5))   # side by side → shape (3, 8)
```

### Vertical Stacking
```python
np.vstack((a4, a5))   # ek ke neeche ek → shape (6, 4)
```

**Condition:** hstack ke liye rows same, vstack ke liye columns same.

---

## 11. Splitting (array ko todna)

### Horizontal Splitting
```python
np.hsplit(a4, 2)      # columns ko 2 parts mein
np.hsplit(a4, 4)      # 4 parts
```

### Vertical Splitting
```python
np.vsplit(a5, 3)      # rows ko 3 parts mein
```

**Condition:** Barabar barabar divide hona chahiye, warna error.

---
---

# SESSION 14 — ADVANCED NUMPY

## 1. NumPy Array vs Python List

### 1.1 Speed ka comparison

```python
# Python list
a = [i for i in range(10000000)]
b = [i for i in range(10000000, 20000000)]

import time
start = time.time()
c = [a[i] + b[i] for i in range(len(a))]
print(time.time() - start)      # ~4-6 seconds

# NumPy array
a = np.arange(10000000)
b = np.arange(10000000, 20000000)

start = time.time()
c = a + b
print(time.time() - start)      # ~0.05 seconds
```

**Result: NumPy roughly 50–100x faster hai.**

### 1.2 Memory ka comparison

```python
import sys

a = [i for i in range(10000000)]
print(sys.getsizeof(a))         # ~89 MB

a = np.arange(10000000)
print(sys.getsizeof(a))         # ~40 MB

print(sys.getsizeof(np.arange(10000000, dtype=np.int8)))   # ~10 MB
```

### 1.3 Summary table

| Feature | Python List | NumPy Array |
|---|---|---|
| Speed | Slow | Fast |
| Memory | Zyada | Kam |
| Datatype | Mixed allowed | Homogeneous only |
| Convenience | Zyada flexible | Math ke liye better |
| Vectorization | Nahi | Haan |

---

## 2. Advanced Indexing

### 2.1 Fancy Indexing

Ek saath multiple non-continuous rows/columns nikalna — index ki **list** pass karte hain.

```python
a = np.arange(24).reshape(6, 4)

a[[0, 2, 3]]         # row 0, 2, 3 nikaalo
a[:, [0, 2, 3]]      # column 0, 2, 3 nikaalo
a[[0, 2, 3], :]
```

Normal slicing se ye nahi ho sakta kyunki rows continuous nahi hain.

### 2.2 Boolean Indexing ⭐ (bahut important)

Condition ke basis pe data filter karna.

```python
a = np.random.randint(1, 100, 24).reshape(6, 4)

# 50 se badi saari values
a[a > 50]

# saari even numbers
a[a % 2 == 0]

# 50 se badi AUR even
a[(a > 50) & (a % 2 == 0)]

# 50 se badi YA even
a[(a > 50) | (a % 2 == 0)]

# jo 7 se divisible NAHI hain
a[~(a % 7 == 0)]
```

**⚠️ Rule:** NumPy mein `and`, `or`, `not` ki jagah `&`, `|`, `~` use karo. Aur har condition ko **round brackets** mein rakho.

---

## 3. Broadcasting ⭐⭐

Broadcasting NumPy ka wo mechanism hai jo **alag-alag shapes** ke arrays pe arithmetic operations allow karta hai.

### Simple example
```python
a = np.arange(6).reshape(2, 3)   # shape (2,3)
b = np.arange(3).reshape(1, 3)   # shape (1,3)
print(a + b)                     # kaam kar jayega!
```
Chhoti array ko NumPy internally **stretch** kar deta hai badi wali ke shape tak.

### Broadcasting ke Rules

1. **Rule 1:** Agar dono arrays ke dimensions alag hain, toh chhoti wali ke shape ke **left side** pe `1` add kar do jab tak dimensions barabar na ho jayein.
2. **Rule 2:** Agar kisi dimension mein shapes match nahi karte lekin ek ki value `1` hai, toh usko doosre ke barabar **stretch** kar do.
3. **Rule 3:** Agar kisi dimension mein shapes match nahi karte aur koi bhi `1` nahi hai → **Error**.

### Examples jo kaam karte hain ✅

```python
# (3,3) aur (3,)
a = np.arange(12).reshape(4, 3)
b = np.arange(3)
a + b       # ✅ b (3,) → (1,3) → (4,3)

# (3,4) aur (3,1)
a = np.arange(12).reshape(3, 4)
b = np.arange(3).reshape(3, 1)
a + b       # ✅

# (1,3) aur (3,1)  → result (3,3)
a = np.arange(3).reshape(1, 3)
b = np.arange(3).reshape(3, 1)
a + b       # ✅ dono stretch honge

# (3,4) aur (1,1)
a = np.arange(12).reshape(3, 4)
b = np.array([1])
a + b       # ✅

# (1,1) aur (2,2)
a = np.array([[1]])
b = np.arange(4).reshape(2,2)
a + b       # ✅
```

### Examples jo FAIL karte hain ❌

```python
# (3,4) aur (4,3)
a = np.arange(12).reshape(3, 4)
b = np.arange(12).reshape(4, 3)
a + b       # ❌ Error

# (3,4) aur (4,)
a = np.arange(12).reshape(3, 4)
b = np.arange(4)
a + b       # ✅ ye kaam karta hai! (4,) → (1,4) → (3,4)

# (16,3) aur (3,1)
a = np.arange(48).reshape(16, 3)
b = np.arange(3).reshape(3, 1)
a + b       # ❌ Error (3 ≠ 16 aur 16 mein 1 nahi)
```

**Yaad rakhne ka tareeqa:** Shapes ko **right se left** align karo. Har position pe ya toh numbers barabar hone chahiye, ya kisi ek ka `1` hona chahiye.

```
   (4, 3)
      (3,)  →  (1, 3)  →  ✅
   
   (3, 4)
   (4, 3)   →  ❌
```

---

## 4. Working with Mathematical Formulas ⭐

Ye section ML ke liye bahut zaroori hai. NumPy se formulas ko **vectorized** form mein likhna.

### 4.1 Sigmoid Function

Neural networks ka classic activation function.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```python
def sigmoid(array):
    return 1 / (1 + np.exp(-array))

a = np.arange(100)
sigmoid(a)
```

Ek hi line mein poore array pe apply ho gaya — koi loop nahi.

### 4.2 Mean Squared Error (MSE)

Regression ka loss function.

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2$$

```python
actual = np.random.randint(1, 50, 25)
predicted = np.random.randint(1, 50, 25)

def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

mse(actual, predicted)
```

### 4.3 Binary Cross Entropy (Log Loss)

$$BCE = -\frac{1}{n}\sum \left[ y\log(\hat{y}) + (1-y)\log(1-\hat{y}) \right]$$

```python
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

---

## 5. Working with Missing Values

NumPy mein missing value ko `np.nan` se represent karte hain. `nan` ka datatype **float** hota hai.

```python
a = np.array([1, 2, 3, 4, np.nan, 6])
print(a)         # [ 1.  2.  3.  4. nan  6.]

# ⚠️ nan == nan → False! Isliye == se check nahi kar sakte
print(np.nan == np.nan)   # False

# Sahi tareeqa:
np.isnan(a)          # [False False False False True False]
a[~np.isnan(a)]      # nan hata do → [1. 2. 3. 4. 6.]
```

**Note:** `np.nan` kisi bhi operation ko `nan` bana deta hai:
```python
np.sum(a)          # nan
np.nansum(a)       # 16.0  (nan ko ignore karta hai)
np.nanmean(a)
np.nanmax(a)
```

---

## 6. Plotting Graphs (NumPy + Matplotlib)

```python
import matplotlib.pyplot as plt
```

### y = x
```python
x = np.linspace(-10, 10, 100)
y = x
plt.plot(x, y)
plt.show()
```

### y = x²
```python
x = np.linspace(-10, 10, 100)
y = x ** 2
plt.plot(x, y)
```

### y = sin(x)
```python
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y)
```

### y = x · log(x)
```python
x = np.linspace(-10, 10, 100)
y = x * np.log(x)
plt.plot(x, y)
```

### Sigmoid curve
```python
x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y)
```

---
---

# SESSION 15 — NUMPY TRICKS

Ye session bilkul function-by-function hai. Har ek zaroori NumPy function.

Setup:
```python
a = np.random.randint(1, 100, 15)
b = np.random.randint(1, 100, 24).reshape(6, 4)
```

---

## 1. `np.sort()` — sorting

```python
np.sort(a)                # ascending
np.sort(a)[::-1]          # descending
np.sort(b, axis=0)        # column-wise
np.sort(b, axis=1)        # row-wise (default)

a.sort()                  # in-place sorting (original badal jayega)
```

---

## 2. `np.append()` — element/array add karna

```python
np.append(a, 200)                        # end mein 200 add
np.append(b, np.ones((b.shape[0], 1)), axis=1)   # ek naya column of 1s
np.append(b, np.random.random((b.shape[0],1)), axis=1)
```

---

## 3. `np.concatenate()` — arrays jodna

```python
c = np.arange(6).reshape(2, 3)
d = np.arange(6, 12).reshape(2, 3)

np.concatenate((c, d), axis=0)   # vertically → (4,3)
np.concatenate((c, d), axis=1)   # horizontally → (2,6)
```

`hstack`/`vstack` isi ka shortcut hain.

---

## 4. `np.unique()` — unique values

```python
e = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
np.unique(e)                          # [1 2 3 4 5 6]
np.unique(e, return_counts=True)      # values + kitni baar aaye
np.unique(e, return_index=True)       # pehla index
```

---

## 5. `np.expand_dims()` — dimension add karna

```python
a.shape                          # (15,)
np.expand_dims(a, axis=0).shape  # (1, 15)  → row vector
np.expand_dims(a, axis=1).shape  # (15, 1)  → column vector
```

**Use case:** ML models ko aksar 2D input chahiye hota hai, single sample ko 2D banane ke liye.

---

## 6. `np.where()` ⭐ — vectorized if-else

Syntax: `np.where(condition, value_if_true, value_if_false)`

```python
# 50 se badi values ke indices
np.where(a > 50)

# 50 se badi values ko 0 kar do
np.where(a > 50, 0, a)

# saare even numbers ko 0 kar do
np.where(a % 2 == 0, 0, a)

# Nested
np.where(a > 50, 'High', 'Low')
```

---

## 7. `np.argmax()` aur `np.argmin()` ⭐

Max/min ki **value** nahi, uska **index** batata hai.

```python
np.argmax(a)              # sabse badi value ka index
np.argmin(a)              # sabse chhoti value ka index

np.argmax(b, axis=0)      # har column mein max ka index
np.argmax(b, axis=1)      # har row mein max ka index
```

**ML mein use:** Classification model probabilities deta hai, `argmax` se pata chalta hai kaunsi class predict hui.

Related: `np.argsort()` — sorted order ke indices.

---

## 8. `np.cumsum()` aur `np.cumprod()` — cumulative

```python
np.cumsum(a)              # [1, 1+2, 1+2+3, ...]
np.cumsum(b, axis=0)      # column-wise cumulative sum
np.cumsum(b, axis=1)

np.cumprod(a)             # cumulative product
```

**Use case:** Running totals, cumulative revenue charts.

---

## 9. `np.percentile()`

```python
np.percentile(a, 50)      # median
np.percentile(a, 100)     # max
np.percentile(a, 0)       # min
np.percentile(a, 25)      # Q1
np.percentile(a, 75)      # Q3

np.median(a)              # == percentile(a, 50)
```

**Use case:** Outlier detection (IQR method).

---

## 10. `np.histogram()`

Frequency distribution nikalta hai.
```python
np.histogram(a, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# (counts_array, bin_edges_array)
```

---

## 11. `np.corrcoef()` — correlation

```python
salary = np.array([20000, 40000, 25000, 35000, 60000])
experience = np.array([1, 3, 2, 4, 2])

np.corrcoef(salary, experience)
# Correlation matrix return karta hai (-1 se 1 ke beech)
```

---

## 12. `np.isin()` — membership check

```python
items = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
np.isin(a, items)         # boolean array
a[np.isin(a, items)]      # sirf wo values jo list mein hain
```

---

## 13. `np.flip()` — reverse

```python
np.flip(a)              # 1D reverse
np.flip(b)              # dono axes reverse
np.flip(b, axis=0)      # rows reverse
np.flip(b, axis=1)      # columns reverse
```

---

## 14. `np.put()` — in-place replacement

```python
np.put(a, [0, 1], [110, 530])
# index 0 pe 110, index 1 pe 530 rakh do (ORIGINAL badal jayega)
```

---

## 15. `np.delete()`

```python
np.delete(a, 0)              # index 0 hata do
np.delete(a, [0, 2, 4])      # multiple indices
np.delete(b, 0, axis=0)      # pehli row hata do
np.delete(b, 1, axis=1)      # doosra column hata do
```

---

## 16. Set Functions ⭐

```python
m = np.array([1, 2, 3, 4, 5])
n = np.array([3, 4, 5, 6, 7])

np.union1d(m, n)        # [1 2 3 4 5 6 7]  — dono ka combined unique
np.intersect1d(m, n)    # [3 4 5]          — common
np.setdiff1d(n, m)      # [6 7]            — n mein hai, m mein nahi
np.setxor1d(m, n)       # [1 2 6 7]        — common ke alawa sab
np.in1d(m, 1)           # membership check
```

---

## 17. `np.clip()` — values ko range mein baandhna

```python
np.clip(a, a_min=25, a_max=75)
# 25 se chhoti values → 25
# 75 se badi values → 75
# beech wali → waisi ki waisi
```

**Use case:** Outliers ko cap karna, gradient clipping.

---

## 18. Bonus functions (extra tricks)

```python
np.swapaxes(b, 0, 1)                 # axes swap
np.count_nonzero(a)                  # non-zero elements ginta hai
np.tile(a, 3)                        # array ko repeat karta hai
np.repeat(a, 3)                      # har element ko repeat karta hai
np.allclose(a, b)                    # do arrays lagbhag barabar hain?
np.equal(a, b)                       # element-wise equality
np.array_equal(a, b)                 # poori array barabar hai?
np.random.permutation(a)             # shuffle
np.random.shuffle(a)                 # in-place shuffle
np.random.choice(a, 5)               # 5 random elements
np.flatnonzero(a > 50)               # non-zero ke flat indices
np.bincount(a)                       # occurrence count
np.trapz(y, x)                       # area under curve
```

---

## NumPy Quick Revision Cheat Sheet

| Kaam | Function |
|---|---|
| Array banao | `np.array`, `np.arange`, `np.linspace`, `np.zeros`, `np.ones`, `np.identity`, `np.random.*` |
| Shape jaano | `.shape`, `.ndim`, `.size`, `.dtype`, `.itemsize` |
| Shape badlo | `reshape`, `ravel`, `flatten`, `.T`, `expand_dims` |
| Jodo | `hstack`, `vstack`, `concatenate`, `append` |
| Todo | `hsplit`, `vsplit`, `split` |
| Filter | Boolean indexing, `np.where`, `np.isin`, `np.clip` |
| Stats | `mean`, `median`, `std`, `var`, `percentile`, `corrcoef` |
| Aggregate | `sum`, `prod`, `min`, `max`, `cumsum`, `cumprod` |
| Index dhoondo | `argmax`, `argmin`, `argsort` |
| Sets | `union1d`, `intersect1d`, `setdiff1d`, `setxor1d` |
| Missing | `np.nan`, `np.isnan`, `nansum`, `nanmean` |
| Matrix math | `np.dot`, `@`, `np.linalg.inv`, `np.linalg.det` |
