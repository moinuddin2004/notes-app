# 📚 Playlist Notes — Index & Roadmap
## "Numpy + Pandas + Matplotlib + Seaborn — CampusX" (15 videos)

---

## Ye notes kaise use karein

Ye notes CampusX ke DSMP (Data Science Mentorship Program) sessions ke hisaab se organize kiye gaye hain — wahi order jo playlist follow karti hai. Har file self-contained hai, code ke saath.

**Revision strategy:**
1. Pehle **Quick Revision table** dekho (har file ke end mein hai)
2. Jo topic bhool gaye ho, uska section khol ke code padho
3. Colab kholo aur code khud chala ke dekho — sirf padhne se yaad nahi rehta

---

## 📁 Files

| File | Kya hai andar | Sessions |
|---|---|---|
| **01_NumPy_Complete_Notes.md** | Array creation, attributes, indexing/slicing, operations, functions, reshaping, stacking, splitting, fancy & boolean indexing, broadcasting, ML formulas, missing values, 30+ NumPy tricks | Session 13, 14, 15 |
| **02_Pandas_Series_DataFrame.md** | Series banana/methods/math/boolean indexing, DataFrame creation, iloc/loc, filtering, naye columns, 19 important DataFrame methods | Session 16, 17, 18 |
| **03_Advanced_Pandas.md** | GroupBy (split-apply-combine), merge/concat/join, MultiIndex, stack/unstack, melt, pivot_table, `.str` vectorized string ops, DateTime & `.dt` | Session 19, 20, 21, 22 |
| **04_Matplotlib_Complete.md** | Line, scatter, bar, histogram, pie, subplots, 3D plots, contour, heatmap, annotations, styles, Pandas plotting | Matplotlib sessions |
| **05_Seaborn_Complete.md** | Figure vs axes level, relational/distribution/categorical/regression/matrix/multiplot, pairplot, jointplot, heatmap, themes, full EDA workflow | Seaborn sessions |

---

## 🗺️ Playlist ka Structure (CampusX DSMP order)

### Week 5 — NumPy
| # | Session | File |
|---|---|---|
| 1 | Session 13 — Numpy Fundamentals | `01_NumPy` → Session 13 |
| 2 | Session 14 — Advanced Numpy | `01_NumPy` → Session 14 |
| 3 | Session 15 — Numpy Tricks | `01_NumPy` → Session 15 |

### Week 6 — Pandas
| # | Session | File |
|---|---|---|
| 4 | Session 16 — Pandas Series | `02_Pandas` → Session 16 |
| 5 | Important Series Methods (Supplementary) | `02_Pandas` → Section 10 |
| 6 | Session 17 — Pandas DataFrame | `02_Pandas` → Session 17 |
| 7 | Session 18 — Important DataFrame Methods | `02_Pandas` → Session 18 |

### Week 7 — Advanced Pandas
| # | Session | File |
|---|---|---|
| 8 | Session 19 — GroupBy Object | `03_Advanced` → Session 19 |
| 9 | Session 20 — Merging, Joining, Concatenating | `03_Advanced` → Session 20 |
| 10 | Session 21 — MultiIndex Series & DataFrames | `03_Advanced` → Session 21 |
| 11 | Session 22 — Vectorized String Ops + DateTime | `03_Advanced` → Session 22 |

### Visualization
| # | Session | File |
|---|---|---|
| 12 | Plotting using Matplotlib (Part 1) | `04_Matplotlib` → Part 1-5 |
| 13 | Advanced Matplotlib (Part 2) | `04_Matplotlib` → Part 6-11 |
| 14 | Plotting using Seaborn (Part 1) | `05_Seaborn` → Part 1-3 |
| 15 | Advanced Seaborn (Part 2) | `05_Seaborn` → Part 4-8 |

---

## 📦 Datasets jo CampusX ke Colab notebooks mein use hote hain

Agar tumhe khud practice karni hai toh ye datasets chahiye honge:

| Dataset | Kahan use hota hai |
|---|---|
| `kohli_ipl.csv` | Series (Session 16) |
| `subs.csv` | Series — YouTube subscribers |
| `bollywood.csv` | Series — movie → actor |
| `movies.csv` / `imdb-top-1000.csv` | DataFrame, GroupBy |
| `ipl-matches.csv` | DataFrame filtering, value_counts |
| `titanic.csv` | String ops, EDA |
| `batsman_runs_ipl.csv` | rank, sort |
| `courses.csv`, `students.csv`, `nov.csv`, `dec.csv`, `regs.csv` | Merging (Session 20) |
| `matches.csv`, `deliveries.csv` | Merging practice questions |
| `time_series_covid19_*.csv` | melt (Session 21) |
| `expense_data.csv` | pivot_table |
| `sharma-kohli.csv`, `batter.csv`, `vk.csv`, `gayle-175.csv` | Matplotlib |
| Seaborn built-in: `tips`, `iris`, `titanic`, `flights` | Seaborn |

**Zyadatar CampusX datasets yahan milte hain:** GitHub par `campusx-official` organization mein, ya har session ke YouTube description mein diye gaye Colab notebook mein.

---

## ⚡ Setup — har notebook ke shuru mein

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)     # saare columns dikhao
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

%matplotlib inline
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

---

## 🎯 Ek Nazar Mein — Poore Playlist ka Nichod

### NumPy (kya seekha)
- Array banane ke 8 tareeqe, attributes, dtype se memory optimize karna
- Vectorized operations — loops ki zaroorat nahi
- Boolean indexing + fancy indexing se filtering
- Broadcasting ke 3 rules
- ML formulas (sigmoid, MSE, cross-entropy) NumPy mein likhna
- 30+ utility functions (`where`, `argmax`, `cumsum`, `percentile`, set functions...)

### Pandas (kya seekha)
- Series & DataFrame — do core objects
- `loc` vs `iloc` ka farq
- Boolean masking se filtering
- Missing values handle karna (`dropna`, `fillna`)
- GroupBy — split/apply/combine
- Merge & concat — tables jodna
- MultiIndex, melt, pivot_table — data reshape karna
- `.str` aur `.dt` accessors

### Matplotlib (kya seekha)
- 6 basic plots: line, scatter, bar, hist, pie, box
- Subplots aur object-oriented API
- 3D plotting aur contour
- Customization — colors, styles, annotations

### Seaborn (kya seekha)
- Figure-level vs Axes-level ka farq
- 5 plot categories: relational, distribution, categorical, regression, matrix
- `hue`, `col`, `row` se multi-dimensional analysis
- `pairplot` + `heatmap(corr)` — EDA ke do sabse powerful tools

---

## 🔥 Interview ke liye Top Questions

**NumPy:**
1. NumPy list se fast kyun hai? (C implementation, contiguous memory, vectorization, SIMD)
2. Broadcasting ke rules kya hain?
3. `ravel()` vs `flatten()` — view vs copy
4. `np.where()` kaise kaam karta hai?
5. `axis=0` aur `axis=1` mein kya farq hai?
6. View aur copy mein kya antar hai?

**Pandas:**
1. `loc` vs `iloc` — label vs position, aur slicing mein end include/exclude
2. `merge` vs `join` vs `concat` — kab kya use karein
3. `apply` vs `map` vs `applymap`
4. `pivot` vs `pivot_table` ka farq
5. Missing values handle karne ke tareeqe
6. `groupby().agg()` mein multiple aggregations kaise
7. `SettingWithCopyWarning` kyun aata hai?
8. Memory optimize kaise karein? (`astype`, `category` dtype)

**Visualization:**
1. Histogram vs Bar chart mein kya farq hai?
2. Box plot kaise padhte hain? Outlier kaise identify karein?
3. Seaborn mein figure-level aur axes-level functions ka farq
4. Correlation heatmap se kya insight milta hai?
5. Kaunsa plot kab use karein? (num vs num, num vs cat, cat vs cat)
