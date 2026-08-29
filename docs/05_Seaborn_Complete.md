# Seaborn — Complete Notes
### CampusX: Plotting using Seaborn (Part 1 + Part 2)

---

## 0. Setup

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

**Seaborn kya hai?** Matplotlib ke upar bani ek high-level statistical plotting library. Kam code, zyada khoobsurat plots, aur Pandas DataFrames ke saath directly kaam karta hai.

### Built-in datasets (practice ke liye)
```python
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')
flights = sns.load_dataset('flights')
```

---

## 1. Figure-level vs Axes-level Functions ⭐⭐

Ye Seaborn ka **sabse important concept** hai.

| | **Axes-level** | **Figure-level** |
|---|---|---|
| Kya banata hai | Ek single Matplotlib `Axes` | Poora `Figure` (FacetGrid) |
| Matplotlib subplot mein daal sakte ho? | ✅ Haan (`ax=` parameter) | ❌ Nahi |
| Multiple subplots automatic? | ❌ Nahi | ✅ `col=` / `row=` se |
| Return karta hai | `Axes` object | `FacetGrid` object |
| Example | `sns.scatterplot()` | `sns.relplot()` |

### Seaborn ka structure (ye tree yaad kar lo)

```
FIGURE-LEVEL          AXES-LEVEL (uske andar)
──────────────────────────────────────────────
relplot()      →      scatterplot(), lineplot()
displot()      →      histplot(), kdeplot(), ecdfplot(), rugplot()
catplot()      →      stripplot(), swarmplot(), boxplot(),
                      violinplot(), boxenplot(), pointplot(),
                      barplot(), countplot()
lmplot()       →      regplot()
```

Alag standalone: `heatmap()`, `clustermap()`, `pairplot()`, `jointplot()`

**Kab kaunsa use karein?**
- Sirf ek plot chahiye → **axes-level** (`sns.scatterplot()`)
- Category ke hisaab se kai subplots chahiye → **figure-level** (`sns.relplot(col='sex')`)

---
---

# PART 1 — RELATIONAL PLOTS

**Maksad:** Do **numerical** columns ke beech ka relationship dekhna.

## 1.1 `scatterplot()` ⭐

```python
sns.scatterplot(data=tips, x='total_bill', y='tip')
```

### `hue` — teesra (categorical) dimension ⭐
```python
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex')
```

### `style` — chautha dimension (marker shape)
```python
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', style='time')
```

### `size` — panchwa dimension (point size)
```python
sns.scatterplot(data=tips, x='total_bill', y='tip',
                hue='sex', style='time', size='size')
```

**Ek scatterplot mein 5 dimensions dikha sakte ho: x, y, hue, style, size.**

---

## 1.2 `lineplot()` ⭐

**Kab?** Time series / continuous trend.

```python
gap = px.data.gapminder()      # ya koi bhi time-based dataset
temp_df = gap[gap['country'] == 'India']

sns.lineplot(data=temp_df, x='year', y='lifeExp')
```

### Multiple lines
```python
temp_df = gap[gap['country'].isin(['India', 'Pakistan', 'China'])]
sns.lineplot(data=temp_df, x='year', y='lifeExp', hue='country')
sns.lineplot(data=temp_df, x='year', y='lifeExp', hue='country', style='continent')
```

**Note:** `lineplot` automatically **confidence interval** (shaded band) bhi bana deta hai agar ek x ke multiple y values hon.
```python
sns.lineplot(data=tips, x='size', y='total_bill', errorbar=None)  # band hatane ke liye
```

---

## 1.3 `relplot()` — Figure-level ⭐⭐

```python
sns.relplot(data=tips, x='total_bill', y='tip', kind='scatter')
sns.relplot(data=tips, x='total_bill', y='tip', kind='line')
```

### Facets — `col` aur `row` ⭐
```python
# Har 'sex' ka alag plot (side by side)
sns.relplot(data=tips, x='total_bill', y='tip', kind='scatter', col='sex')

# Grid banao
sns.relplot(data=tips, x='total_bill', y='tip', kind='scatter',
            col='day', row='time')

# col_wrap se wrap karo
sns.relplot(data=gap, x='lifeExp', y='gdpPercap', kind='scatter',
            col='year', col_wrap=3)
```

**Ye Seaborn ki superpower hai** — ek line mein poora dashboard.

---
---

# PART 2 — DISTRIBUTION PLOTS

**Maksad:** Ek ya do numerical columns ka **distribution** samajhna — data kaisa spread hai, normal hai ya skewed, outliers hain kya.

## 2.1 `histplot()` ⭐

```python
sns.histplot(data=tips, x='total_bill')
sns.histplot(data=tips, x='total_bill', bins=20)
sns.histplot(data=tips, x='total_bill', hue='sex')
sns.histplot(data=tips, x='total_bill', hue='sex', multiple='stack')
sns.histplot(data=tips, x='total_bill', kde=True)     # KDE curve bhi
sns.histplot(data=tips, x='total_bill', element='step')
```

**Categorical column pe bhi kaam karta hai:**
```python
sns.histplot(data=titanic, x='survived')
```

---

## 2.2 `kdeplot()` ⭐

Kernel Density Estimation — histogram ka **smooth** version.

```python
sns.kdeplot(data=tips, x='total_bill')
sns.kdeplot(data=tips, x='total_bill', hue='sex')
sns.kdeplot(data=tips, x='total_bill', hue='sex', fill=True)
sns.kdeplot(data=tips, x='total_bill', hue='sex', multiple='stack')
```

**Bivariate KDE (2D contour):**
```python
sns.kdeplot(data=tips, x='total_bill', y='tip')
sns.kdeplot(data=tips, x='total_bill', y='tip', fill=True, cmap='Blues')
```

---

## 2.3 `rugplot()`

Har observation ke liye chhoti si line — usually kisi aur plot ke upar.

```python
sns.kdeplot(data=tips, x='total_bill')
sns.rugplot(data=tips, x='total_bill')
```

---

## 2.4 `ecdfplot()`

Empirical Cumulative Distribution Function.
```python
sns.ecdfplot(data=tips, x='total_bill')
```
Batata hai: "kitna % data is value se kam hai".

---

## 2.5 `displot()` — Figure-level ⭐

```python
sns.displot(data=tips, x='total_bill', kind='hist')
sns.displot(data=tips, x='total_bill', kind='kde')
sns.displot(data=tips, x='total_bill', kind='ecdf')

sns.displot(data=tips, x='total_bill', kind='hist', col='sex')
sns.displot(data=tips, x='total_bill', kind='kde', col='day', row='time')
```

### Bivariate histogram (heatmap jaisa)
```python
sns.displot(data=tips, x='total_bill', y='tip', kind='hist')
```

---
---

# PART 3 — CATEGORICAL PLOTS ⭐⭐

**Maksad:** Ek **categorical** aur ek **numerical** column ka relation dekhna.

Categorical plots ke **3 sub-groups**:

| Group | Plots | Kya dikhate hain |
|---|---|---|
| **Scatter type** | `stripplot`, `swarmplot` | Har individual point |
| **Distribution type** | `boxplot`, `violinplot`, `boxenplot` | Distribution ka shape |
| **Estimate type** | `barplot`, `pointplot`, `countplot` | Central tendency (mean/count) |

---

## 3.1 `stripplot()`

```python
sns.stripplot(data=tips, x='day', y='total_bill')
sns.stripplot(data=tips, x='day', y='total_bill', jitter=0.2)
sns.stripplot(data=tips, x='day', y='total_bill', hue='sex')
```
Points overlap ho jaate hain — isliye `jitter` use karte hain.

## 3.2 `swarmplot()`

```python
sns.swarmplot(data=tips, x='day', y='total_bill')
sns.swarmplot(data=tips, x='day', y='total_bill', hue='sex')
```
Points ko aise arrange karta hai ki koi overlap na ho. Bade datasets pe slow ho jaata hai.

---

## 3.3 `boxplot()` ⭐⭐

Sabse zyada use hone wala EDA plot.

```python
sns.boxplot(data=tips, x='day', y='total_bill')
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex')
sns.boxplot(data=tips, y='total_bill')          # sirf ek variable
```

### Box plot kaise padhein? ⭐
```
     ┌─── Upper whisker (Q3 + 1.5*IQR)
     │
  ┌──┴──┐  ← Q3 (75th percentile)
  │─────│  ← Median (Q2, 50th percentile)
  └──┬──┘  ← Q1 (25th percentile)
     │
     └─── Lower whisker (Q1 - 1.5*IQR)

  ● ● ●    ← Outliers (whiskers ke bahar ke points)
```
- **Box** = middle 50% data (IQR = Q3 − Q1)
- **Line inside box** = median
- **Dots** = outliers

---

## 3.4 `violinplot()` ⭐

Box plot + KDE ka combination. Distribution ka shape bhi dikhta hai.

```python
sns.violinplot(data=tips, x='day', y='total_bill')
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex')
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True)  # ⭐
```
`split=True` bahut useful hai — ek hi violin ke do halves.

## 3.5 `boxenplot()`

Bade datasets ke liye box plot ka enhanced version. Zyada quantiles dikhata hai.
```python
sns.boxenplot(data=tips, x='day', y='total_bill')
```

---

## 3.6 `barplot()` ⭐

Har category ka **mean** (default) dikhata hai + confidence interval.

```python
sns.barplot(data=tips, x='sex', y='total_bill')
sns.barplot(data=tips, x='sex', y='total_bill', hue='smoker')

# Estimator badlo
sns.barplot(data=tips, x='sex', y='total_bill', estimator=np.std)
sns.barplot(data=tips, x='sex', y='total_bill', estimator=np.median)

# Confidence interval hatao
sns.barplot(data=tips, x='sex', y='total_bill', errorbar=None)
```

**⚠️ Yaad rakho:** Matplotlib ka `plt.bar()` raw values plot karta hai; Seaborn ka `barplot()` **aggregate (mean)** plot karta hai.

## 3.7 `pointplot()`

Bar plot jaisa hi, lekin points aur lines se. Trend dekhne mein aasan.
```python
sns.pointplot(data=tips, x='sex', y='total_bill')
sns.pointplot(data=tips, x='day', y='total_bill', hue='sex')
```

## 3.8 `countplot()` ⭐

Har category kitni baar aayi — `value_counts()` ka visual version.
```python
sns.countplot(data=tips, x='sex')
sns.countplot(data=tips, x='day', hue='sex')
sns.countplot(data=titanic, x='pclass', hue='survived')
```

---

## 3.9 `catplot()` — Figure-level ⭐

Saare categorical plots ka figure-level version.

```python
sns.catplot(data=tips, x='day', y='total_bill', kind='strip')
sns.catplot(data=tips, x='day', y='total_bill', kind='swarm')
sns.catplot(data=tips, x='day', y='total_bill', kind='box')
sns.catplot(data=tips, x='day', y='total_bill', kind='violin')
sns.catplot(data=tips, x='day', y='total_bill', kind='boxen')
sns.catplot(data=tips, x='day', y='total_bill', kind='bar')
sns.catplot(data=tips, x='day', y='total_bill', kind='point')
sns.catplot(data=tips, x='day', kind='count')

# Facets
sns.catplot(data=tips, x='sex', y='total_bill', kind='box', col='day')
sns.catplot(data=tips, x='sex', y='total_bill', kind='violin',
            col='day', row='time')
```

---
---

# PART 4 — REGRESSION PLOTS

**Maksad:** Do numerical columns ke beech ka **linear relationship** dikhana.

## 4.1 `regplot()` — Axes-level

```python
sns.regplot(data=tips, x='total_bill', y='tip')
```
Scatter + regression line + confidence band.

## 4.2 `lmplot()` — Figure-level ⭐

```python
sns.lmplot(data=tips, x='total_bill', y='tip')
sns.lmplot(data=tips, x='total_bill', y='tip', hue='sex')
sns.lmplot(data=tips, x='total_bill', y='tip', col='sex')
sns.lmplot(data=tips, x='total_bill', y='tip', order=2)   # polynomial fit
```

## 4.3 `residplot()`

Regression ke residuals dikhata hai. Agar random scatter hai → linear model theek hai.
```python
sns.residplot(data=tips, x='total_bill', y='tip')
```

---
---

# PART 5 — MATRIX PLOTS

## 5.1 `heatmap()` ⭐⭐

```python
gap = px.data.gapminder()
temp_df = gap.pivot_table(index='country', columns='year', values='lifeExp')

plt.figure(figsize=(15, 15))
sns.heatmap(temp_df)
```

### Useful parameters
```python
sns.heatmap(data, annot=True)              # numbers dikhao ⭐
sns.heatmap(data, annot=True, fmt='.1f')   # format
sns.heatmap(data, cmap='coolwarm')
sns.heatmap(data, linewidth=0.5)
sns.heatmap(data, cbar=False)
sns.heatmap(data, vmin=0, vmax=100)
```

### Sabse common use — Correlation Matrix ⭐⭐
```python
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
```

---

## 5.2 `clustermap()`

Heatmap + hierarchical clustering (similar rows/cols ko saath rakhta hai).
```python
sns.clustermap(iris.iloc[:, [0,1,2,3]])
```

---
---

# PART 6 — MULTIPLOT GRIDS ⭐⭐

## 6.1 `pairplot()` ⭐⭐⭐

**Sabse zyada use hone wala EDA tool.** Saare numerical columns ke pairs ka scatter plot ek saath.

```python
sns.pairplot(iris)
sns.pairplot(iris, hue='species')          # ⭐ super useful
sns.pairplot(iris, hue='species', diag_kind='kde')
sns.pairplot(iris, vars=['sepal_length', 'petal_length'])
sns.pairplot(iris, kind='reg')             # regression lines ke saath
sns.pairplot(iris, corner=True)            # sirf lower triangle
```

**Diagonal pe** har column ka distribution (histogram/KDE), **baaki cells mein** scatter plots.

## 6.2 `PairGrid()` — customized pairplot

```python
g = sns.PairGrid(iris, hue='species')
g.map(sns.scatterplot)
g.add_legend()

# Alag alag diagonal/upper/lower
g = sns.PairGrid(iris, hue='species')
g.map_diag(sns.histplot)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.add_legend()

# Specific columns
g = sns.PairGrid(iris, hue='species', vars=['sepal_length', 'petal_length'])
g.map(sns.scatterplot)
```

---

## 6.3 `jointplot()` ⭐

Do variables ka relationship + dono ke individual distributions (margins pe).

```python
sns.jointplot(data=tips, x='total_bill', y='tip')
sns.jointplot(data=tips, x='total_bill', y='tip', kind='scatter')
sns.jointplot(data=tips, x='total_bill', y='tip', kind='kde')
sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')
sns.jointplot(data=tips, x='total_bill', y='tip', kind='hist')
sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')
sns.jointplot(data=tips, x='total_bill', y='tip', hue='sex')
```

## 6.4 `JointGrid()` — customized jointplot

```python
g = sns.JointGrid(data=tips, x='total_bill', y='tip')
g.plot(sns.scatterplot, sns.violinplot)

g = sns.JointGrid(data=tips, x='total_bill', y='tip')
g.plot_joint(sns.kdeplot, fill=True)
g.plot_marginals(sns.histplot, kde=True)
```

---

## 6.5 `FacetGrid()`

Manual facets banane ke liye.
```python
g = sns.FacetGrid(data=tips, col='day', row='time')
g.map(sns.scatterplot, 'total_bill', 'tip')

g = sns.FacetGrid(data=tips, col='day', hue='sex')
g.map(sns.histplot, 'total_bill')
g.add_legend()
```

---
---

# PART 7 — STYLING & THEMES

## 7.1 Themes

```python
sns.set_theme()                        # default seaborn theme
sns.set_style('whitegrid')
sns.set_style('darkgrid')
sns.set_style('white')
sns.set_style('dark')
sns.set_style('ticks')
```

## 7.2 Context (element sizes)

```python
sns.set_context('paper')       # sabse chhota
sns.set_context('notebook')    # default
sns.set_context('talk')
sns.set_context('poster')      # sabse bada
```

## 7.3 Color Palettes ⭐

```python
sns.set_palette('Set2')
sns.color_palette('deep')
sns.color_palette('pastel')
sns.color_palette('bright')
sns.color_palette('dark')
sns.color_palette('colorblind')

# Har plot mein directly
sns.boxplot(data=tips, x='day', y='total_bill', palette='Set2')
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', palette='Dark2')
```

**Palette types:**
- **Qualitative** (categories ke liye): `Set1`, `Set2`, `Set3`, `Paired`, `Accent`, `tab10`
- **Sequential** (kam se zyada): `Blues`, `Greens`, `viridis`, `rocket`, `mako`
- **Diverging** (do extremes): `coolwarm`, `RdBu`, `vlag`, `icefire`

## 7.4 Figure size

```python
# Axes-level ke liye
plt.figure(figsize=(12, 6))
sns.boxplot(...)

# Figure-level ke liye
sns.catplot(..., height=5, aspect=1.5)
```

## 7.5 Matplotlib ke saath mix karna

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.title('Tips vs Total Bill')
plt.xlabel('Bill Amount')
plt.ylabel('Tip Amount')
plt.show()
```

Kyunki Seaborn Matplotlib pe bana hai, saari `plt.*` commands kaam karti hain.

---
---

# PART 8 — EDA WORKFLOW (Seaborn se real analysis)

Ye wo sequence hai jo tum kisi bhi naye dataset pe follow kar sakte ho:

## Step 1 — Univariate Analysis (ek column)

```python
# Numerical column
sns.histplot(data=df, x='age', kde=True)
sns.boxplot(data=df, y='age')

# Categorical column
sns.countplot(data=df, x='sex')
```

## Step 2 — Bivariate Analysis (do columns)

| Column 1 | Column 2 | Plot |
|---|---|---|
| Numerical | Numerical | `scatterplot`, `regplot`, `jointplot`, `lineplot` |
| Numerical | Categorical | `boxplot`, `violinplot`, `barplot`, `stripplot`, `swarmplot` |
| Categorical | Categorical | `countplot(hue=)`, `heatmap(crosstab)` |

```python
# Num vs Num
sns.scatterplot(data=titanic, x='age', y='fare', hue='survived')

# Num vs Cat
sns.boxplot(data=titanic, x='pclass', y='age', hue='survived')

# Cat vs Cat
sns.heatmap(pd.crosstab(titanic['pclass'], titanic['survived']), annot=True, fmt='d')
```

## Step 3 — Multivariate Analysis (3+ columns)

```python
sns.pairplot(df, hue='target')
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
sns.relplot(data=df, x='a', y='b', hue='c', size='d', col='e')
```

---

## Titanic Dataset — Complete EDA Example ⭐

```python
titanic = sns.load_dataset('titanic')

# 1. Survival breakdown
sns.countplot(data=titanic, x='survived')

# 2. Class ke hisaab se survival
sns.countplot(data=titanic, x='pclass', hue='survived')

# 3. Gender ke hisaab se survival
sns.countplot(data=titanic, x='sex', hue='survived')

# 4. Age distribution
sns.histplot(data=titanic, x='age', kde=True)

# 5. Age vs Survival
sns.boxplot(data=titanic, x='survived', y='age')
sns.violinplot(data=titanic, x='survived', y='age', hue='sex', split=True)

# 6. Fare vs Age vs Survival
sns.scatterplot(data=titanic, x='age', y='fare', hue='survived', size='pclass')

# 7. Class + Gender + Survival (3-way)
sns.catplot(data=titanic, x='sex', y='survived', col='pclass', kind='bar')

# 8. Correlation
sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap='coolwarm')

# 9. Pairplot
sns.pairplot(titanic[['survived','age','fare','pclass']], hue='survived')
```

---

## Seaborn — Master Cheatsheet

| Kya dekhna hai | Plot |
|---|---|
| Ek numerical ka distribution | `histplot`, `kdeplot`, `displot`, `boxplot` |
| Ek categorical ki frequency | `countplot` |
| Num vs Num | `scatterplot`, `regplot`, `jointplot`, `lineplot` |
| Num vs Cat | `boxplot`, `violinplot`, `barplot`, `stripplot`, `swarmplot` |
| Cat vs Cat | `countplot(hue=)`, `heatmap(crosstab)` |
| Time series | `lineplot`, `relplot(kind='line')` |
| Correlation | `heatmap(df.corr())` |
| Saare pairs ek saath | `pairplot` |
| Multiple subplots by category | `relplot`, `displot`, `catplot`, `lmplot` (col/row) |
| Outliers dhoondhna | `boxplot` |

### Universal parameters (lagbhag har plot mein kaam karte hain)
```python
data=      # DataFrame
x=, y=     # columns
hue=       # color grouping
palette=   # colors
ax=        # matplotlib axes (axes-level ke liye)
col=, row= # facets (figure-level ke liye)
height=, aspect=   # size (figure-level)
```
