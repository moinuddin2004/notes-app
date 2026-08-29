# Matplotlib — Complete Notes
### CampusX: Plotting using Matplotlib (Basic + Advanced)

---

## 0. Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline        # Jupyter/Colab mein plot inline dikhane ke liye
```

**Matplotlib kya hai?** Python ki sabse purani aur sabse widely-used plotting library. Seaborn, Pandas plotting — sab isi ke upar bane hain.

**Do interfaces:**
1. **`plt.` (pyplot / state-machine)** — quick plots ke liye, MATLAB jaisa
2. **Object-Oriented (Figure + Axes)** — complex/multiple plots ke liye, zyada control

---

# PART 1 — 2D LINE PLOT

## 1.1 Basic line plot

```python
price = [48000, 54000, 57000, 49000, 47000, 45000]
year = [2015, 2016, 2017, 2018, 2019, 2020]

plt.plot(year, price)
plt.show()
```

## 1.2 Mathematical function plot karna

```python
x = np.linspace(-10, 10, 100)
y = x ** 2
plt.plot(x, y)
plt.show()
```

## 1.3 Multiple lines ek hi plot pe

```python
batsman = pd.read_csv('sharma-kohli.csv')

plt.plot(batsman['index'], batsman['V Kohli'])
plt.plot(batsman['index'], batsman['RG Sharma'])
plt.show()
```

---

## 1.4 Labels, Title, Legend ⭐

```python
plt.plot(batsman['index'], batsman['V Kohli'], label='Virat Kohli')
plt.plot(batsman['index'], batsman['RG Sharma'], label='Rohit Sharma')

plt.title('Rohit Sharma vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')
plt.legend()             # label wale names dikhaega
plt.show()
```

**Legend ki position:**
```python
plt.legend(loc='upper right')
# options: 'best', 'upper left', 'lower right', 'center', etc.
```

---

## 1.5 Colors ⭐

```python
plt.plot(x, y, color='red')
plt.plot(x, y, color='#D9F10F')          # hex code
plt.plot(x, y, color=(0.1, 0.2, 0.5))    # RGB tuple
```

**Shortcut color codes:** `b` blue, `g` green, `r` red, `c` cyan, `m` magenta, `y` yellow, `k` black, `w` white.

---

## 1.6 Line style aur width ⭐

```python
plt.plot(x, y, linestyle='solid')     # or '-'
plt.plot(x, y, linestyle='dashed')    # or '--'
plt.plot(x, y, linestyle='dashdot')   # or '-.'
plt.plot(x, y, linestyle='dotted')    # or ':'

plt.plot(x, y, linewidth=3)           # motai
```

---

## 1.7 Markers ⭐

```python
plt.plot(x, y, marker='o')
plt.plot(x, y, marker='D', markersize=10)
```

**Marker types:** `.` point, `o` circle, `v` `^` `<` `>` triangles, `s` square, `p` pentagon, `*` star, `+` plus, `x` cross, `D` diamond, `h` hexagon.

**Sab kuch ek saath:**
```python
plt.plot(batsman['index'], batsman['V Kohli'],
         color='#D9F10F', linestyle='solid', linewidth=3,
         marker='D', markersize=10, label='Virat Kohli')
```

**Format string shortcut:**
```python
plt.plot(x, y, 'ro--')   # red, circle marker, dashed line
plt.plot(x, y, 'g^:')    # green, triangle-up, dotted
```

---

## 1.8 Limiting axes — `xlim` / `ylim` ⭐

```python
price = [48000, 54000, 57000, 49000, 47000, 45000, 4500000]
year = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

plt.plot(year, price)
plt.ylim(0, 75000)       # outlier ki wajah se plot kharab ho raha tha
plt.xlim(2017, 2019)
plt.show()
```

---

## 1.9 Grid

```python
plt.grid()
plt.grid(True, linestyle='--', alpha=0.5)
```

---

## 1.10 Figure size ⭐

```python
plt.figure(figsize=(15, 6))
plt.plot(x, y)
plt.show()
```

**Note:** `figure()` hamesha plotting se **pehle** call karo.

---

## 1.11 Saving the plot

```python
plt.savefig('sample.png')
plt.savefig('sample.pdf', dpi=300, bbox_inches='tight')
```

---

# PART 2 — SCATTER PLOT

**Kab use karein?** Do numerical columns ka **relationship** dekhne ke liye.

## 2.1 `plt.scatter()`

```python
x = np.linspace(-10, 10, 50)
y = 10 * x + 3 + np.random.randint(0, 300, 50)

plt.scatter(x, y)
plt.show()
```

## 2.2 Real dataset pe

```python
df = pd.read_csv('batter.csv')
df = df.head(50)

plt.scatter(df['avg'], df['strike_rate'],
            color='red', marker='+')
plt.title('Avg vs Strike Rate of Top 50 Batsman')
plt.xlabel('Average')
plt.ylabel('Strike Rate')
plt.show()
```

## 2.3 `plt.plot()` se scatter (faster!) ⭐

```python
plt.plot(df['avg'], df['strike_rate'], 'o')
```
Bade datasets pe `plt.plot()` `plt.scatter()` se **tez** hota hai, lekin usmein har point ka size/color alag nahi kar sakte.

## 2.4 Size (`s`), Color (`c`), Transparency (`alpha`) ⭐

```python
tips = sns.load_dataset('tips')

plt.scatter(tips['total_bill'], tips['tip'],
            s=tips['size'] * 20,       # bubble size
            c=tips['size'],            # color mapping
            alpha=0.5)                 # transparency
plt.colorbar()                          # color scale dikhao
plt.show()
```

## 2.5 Colormaps

```python
plt.scatter(x, y, c=z, cmap='viridis')
plt.colorbar()
```
Popular cmaps: `viridis`, `plasma`, `inferno`, `magma`, `coolwarm`, `jet`, `Blues`, `Reds`, `RdYlGn`.

---

# PART 3 — BAR CHART

**Kab use karein?** Numerical vs **Categorical** data compare karne ke liye.

## 3.1 Simple bar chart

```python
children = [10, 20, 40, 10, 30]
colors = ['red', 'blue', 'green', 'yellow', 'pink']

plt.bar(colors, children, color='black')
plt.show()
```

## 3.2 Horizontal bar chart

```python
plt.barh(colors, children, color='black')
```

**Kab horizontal?** Jab category names lambe hon.

## 3.3 Rotating x-labels

```python
plt.bar(df['IPL Team'], df['Titles'])
plt.xticks(rotation='vertical')      # ya rotation=45
plt.show()
```

## 3.4 Grouped (multiple) bar chart ⭐

```python
df = pd.read_csv('batsman_season_record.csv')

plt.bar(np.arange(df.shape[0]) - 0.2, df['2015'], width=0.2, label='2015')
plt.bar(np.arange(df.shape[0]),       df['2016'], width=0.2, label='2016')
plt.bar(np.arange(df.shape[0]) + 0.2, df['2017'], width=0.2, label='2017')

plt.xticks(np.arange(df.shape[0]), df['batsman'])
plt.legend()
plt.show()
```

## 3.5 Stacked bar chart ⭐

```python
plt.bar(df['batsman'], df['2017'], label='2017')
plt.bar(df['batsman'], df['2016'], bottom=df['2017'], label='2016')
plt.bar(df['batsman'], df['2015'], bottom=df['2017'] + df['2016'], label='2015')
plt.legend()
plt.show()
```

`bottom` parameter batata hai ki naya bar kahan se shuru ho.

---

# PART 4 — HISTOGRAM

**Kab use karein?** Ek **numerical** column ka **distribution** dekhne ke liye.

## 4.1 Basic

```python
data = [32, 45, 56, 10, 15, 27, 61]
plt.hist(data)
plt.show()
```

## 4.2 `bins` ⭐

```python
df = pd.read_csv('vk.csv')

plt.hist(df['batsman_runs'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.show()

plt.hist(df['batsman_runs'], bins=20)     # sirf count bhi de sakte ho
```

## 3 Useful parameters

```python
plt.hist(data, bins=20, color='skyblue', edgecolor='black',
         alpha=0.7, logy=False)

plt.hist(data, bins=20, log=True)      # y-axis log scale (outliers ke liye)
plt.hist(data, bins=20, density=True)  # probability density
plt.hist(data, bins=20, cumulative=True)
plt.hist(data, bins=20, histtype='step')
```

## 4.4 Multiple histograms

```python
plt.hist(df1['col'], bins=20, alpha=0.5, label='Group 1')
plt.hist(df2['col'], bins=20, alpha=0.5, label='Group 2')
plt.legend()
```

---

# PART 5 — PIE CHART

**Kab use karein?** Whole ka part dikhane ke liye. **Warning:** 5-6 se zyada categories ho toh pie chart bura lagta hai — bar chart better hai.

## 5.1 Basic

```python
data = [23, 45, 100, 20, 49]
subjects = ['eng', 'science', 'maths', 'sst', 'hindi']

plt.pie(data, labels=subjects)
plt.show()
```

## 5.2 Percentage dikhana — `autopct` ⭐

```python
plt.pie(df['Percentage'], labels=df['Party'], autopct='%0.1f%%')
plt.show()
```

## 5.3 `explode` — slice ko bahar nikalna

```python
plt.pie(df['Percentage'], labels=df['Party'], autopct='%0.1f%%',
        explode=[0.3, 0, 0, 0, 0, 0.1])
```

## 5.4 Shadow, colors, startangle

```python
plt.pie(data, labels=subjects, autopct='%0.1f%%',
        explode=[0.1, 0, 0, 0, 0],
        shadow=True,
        colors=['blue', 'green', 'yellow', 'pink', 'cyan'],
        startangle=90)
```

## 5.5 Multiple pie charts

```python
df = pd.read_csv('gayle-175.csv')
plt.pie(df['batsman_runs'], labels=df['batsman'], autopct='%0.1f%%')
```

---

# PART 6 — SUBPLOTS ⭐⭐

Ek figure mein multiple plots.

## 6.1 `plt.subplot()` (purana tareeqa)

```python
plt.subplot(2, 2, 1)          # 2 rows, 2 cols, plot #1
plt.plot(x, y)

plt.subplot(2, 2, 2)
plt.scatter(x, y)
```

## 6.2 `plt.subplots()` — Object Oriented ⭐ (recommended)

```python
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

ax[0, 0].scatter(df['avg'], df['strike_rate'], color='red')
ax[0, 0].set_title('Avg vs SR')

ax[0, 1].hist(df['runs'])
ax[0, 1].set_title('Runs Distribution')

ax[1, 0].bar(df['batsman'], df['runs'])
ax[1, 0].set_title('Runs by Batsman')

ax[1, 1].pie(df['runs'].head(5), labels=df['batsman'].head(5))
ax[1, 1].set_title('Top 5 Share')

fig.suptitle('IPL Analysis Dashboard')
plt.tight_layout()
plt.show()
```

### Axes objects pe methods (thoda alag naam!)

| pyplot | Axes object |
|---|---|
| `plt.title()` | `ax.set_title()` |
| `plt.xlabel()` | `ax.set_xlabel()` |
| `plt.ylabel()` | `ax.set_ylabel()` |
| `plt.xlim()` | `ax.set_xlim()` |
| `plt.xticks()` | `ax.set_xticks()` |

### 1D subplots
```python
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))
ax[0].plot(x, y1)
ax[1].plot(x, y2)
```

### Ek plot ke andar dusra (inset)
```python
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 1])          # [left, bottom, width, height]
ax2 = fig.add_axes([0.6, 0.6, 0.3, 0.3])  # chhota inset
```

---

# PART 7 — 3D PLOTS ⭐

```python
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(projection='3d')
```

## 7.1 3D Scatter Plot

```python
x = np.random.randint(0, 100, 50)
y = np.random.randint(0, 100, 50)
z = np.random.randint(0, 100, 50)

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(projection='3d')
ax.scatter3D(x, y, z, s=[100]*50, marker='+')
ax.set_title('3D Scatter Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

## 7.2 3D Line Plot

```python
x = np.linspace(-10, 10, 100)
y = np.sin(x)
z = np.cos(x)

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(projection='3d')
ax.plot3D(x, y, z, color='red')
plt.show()
```

## 7.3 3D Surface Plot ⭐

```python
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

xx, yy = np.meshgrid(x, y)      # grid banata hai
z = xx**2 + yy**2
# ya: z = np.sin(xx) + np.cos(yy)

fig = plt.figure(figsize=(12, 8))
ax = plt.subplot(projection='3d')
p = ax.plot_surface(xx, yy, z, cmap='viridis')
fig.colorbar(p)
plt.show()
```

**`np.meshgrid()` samajh lo:** Ye 1D x aur y se 2D coordinate grid banata hai. 3D surface plotting ke liye zaroori hai.

## 7.4 Contour Plot (2D mein 3D dikhana)

```python
fig = plt.figure(figsize=(12, 8))
ax = plt.subplot()
p = ax.contourf(xx, yy, z, cmap='viridis')
fig.colorbar(p)
plt.show()
```

`contour()` sirf lines banata hai, `contourf()` filled version hai.

---

# PART 8 — HEATMAP

```python
df = pd.read_csv('delivery.csv')
grid = df.pivot_table(index='over', columns='ball', values='runs', aggfunc='sum')

plt.figure(figsize=(20, 10))
plt.imshow(grid)
plt.xticks(np.arange(grid.shape[1]), list(grid.columns))
plt.yticks(np.arange(grid.shape[0]), list(grid.index))
plt.colorbar()
plt.show()
```

> Seaborn ka `sns.heatmap()` isse kaafi behtar aur aasan hai.

---

# PART 9 — ANNOTATIONS

## 9.1 `plt.text()`

```python
plt.scatter(df['avg'], df['strike_rate'], s=df['runs'])

for i in range(df.shape[0]):
    plt.text(df['avg'][i], df['strike_rate'][i], df['batter'][i])
```

## 9.2 `plt.annotate()` — arrow ke saath

```python
plt.annotate('Best Player',
             xy=(50, 140),                  # kahan point karna hai
             xytext=(40, 160),              # text kahan
             arrowprops=dict(facecolor='black', shrink=0.05))
```

## 9.3 Horizontal / Vertical lines

```python
plt.axhline(y=50, color='red', linestyle='--')      # horizontal line
plt.axvline(x=30, color='blue', linestyle=':')      # vertical line
plt.axhspan(40, 60, color='yellow', alpha=0.3)      # horizontal band
```

---

# PART 10 — STYLES

```python
plt.style.available          # saare available styles

plt.style.use('ggplot')
plt.style.use('seaborn-v0_8')
plt.style.use('dark_background')
plt.style.use('fivethirtyeight')
plt.style.use('bmh')
plt.style.use('default')     # wapas normal
```

---

# PART 11 — PANDAS SE DIRECT PLOTTING (shortcut)

Pandas ke andar hi matplotlib built-in hai:

```python
df['col'].plot(kind='line')
df['col'].plot(kind='bar')
df['col'].plot(kind='barh')
df['col'].plot(kind='hist', bins=20)
df['col'].plot(kind='pie')
df['col'].plot(kind='box')
df['col'].plot(kind='kde')
df['col'].plot(kind='area')
df.plot(kind='scatter', x='col1', y='col2')

# Options
df.plot(kind='bar', figsize=(12,6), title='My Chart', color='green', legend=True)
```

---

## Matplotlib — Kaunsa plot kab use karein?

| Plot | Kab use karein | Function |
|---|---|---|
| **Line plot** | Time series / trend | `plt.plot()` |
| **Scatter plot** | Do numerical columns ka relation | `plt.scatter()` |
| **Bar chart** | Categorical comparison | `plt.bar()`, `plt.barh()` |
| **Histogram** | Ek numerical column ka distribution | `plt.hist()` |
| **Pie chart** | Part-to-whole (kam categories) | `plt.pie()` |
| **Box plot** | Distribution + outliers | `plt.boxplot()` |
| **Heatmap** | 2D matrix ki intensity | `plt.imshow()` |
| **3D Surface** | 3-variable relationship | `ax.plot_surface()` |

---

## Common Matplotlib Cheatsheet

```python
plt.figure(figsize=(w, h))     # size
plt.plot(x, y)                 # plot
plt.title('Title')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.xlim(a, b); plt.ylim(a, b)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('name.png', dpi=300, bbox_inches='tight')
plt.show()
```
