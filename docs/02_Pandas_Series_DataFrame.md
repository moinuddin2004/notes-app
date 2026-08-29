# Pandas — Series & DataFrame (Complete Notes)
### CampusX Session 16, 17, 18 (Pandas Series + Pandas DataFrame + Important DataFrame Methods)

---

# SESSION 16 — PANDAS SERIES

## 1. Pandas hai kya?

Pandas ek **data analysis aur manipulation** library hai. Ye NumPy ke upar bana hua hai.

**NumPy vs Pandas:**

| NumPy | Pandas |
|---|---|
| Sirf ek datatype | Har column ka alag datatype ho sakta hai |
| Numeric index (0,1,2...) | Custom labels index bana sakte ho |
| Numerical computing | Real-world tabular data (Excel/CSV jaisa) |
| Missing values handle karna mushkil | Built-in missing value support |

```python
import pandas as pd
import numpy as np
```

**Pandas ke do main objects:**
- **Series** → 1D labeled array (Excel ka ek column)
- **DataFrame** → 2D labeled table (poori Excel sheet)

---

## 2. Series banana

### 2.1 List se
```python
country = ['India', 'Pakistan', 'USA', 'Nepal', 'Sri Lanka']
pd.Series(country)
# 0        India
# 1     Pakistan
# 2          USA
# 3        Nepal
# 4    Sri Lanka
# dtype: object
```

### 2.2 Custom index ke saath
```python
marks = [67, 57, 89, 100]
subjects = ['maths', 'english', 'science', 'hindi']

pd.Series(marks, index=subjects)
# maths       67
# english     57
# science     89
# hindi      100
```

### 2.3 `name` parameter
```python
marks = pd.Series(marks, index=subjects, name='Nitish ke marks')
```

### 2.4 Dictionary se (sabse aasan)
```python
marks = {'maths': 67, 'english': 57, 'science': 89, 'hindi': 100}
marks_series = pd.Series(marks, name='Nitish ke marks')
```
Dictionary ki **keys → index** ban jaati hain, **values → data**.

---

## 3. Series Attributes

```python
marks_series.size        # kitne elements
marks_series.dtype       # datatype
marks_series.name        # series ka naam
marks_series.is_unique   # saari values unique hain?
marks_series.index       # index object
marks_series.values      # underlying NumPy array
marks_series.shape       # (4,)
marks_series.ndim        # 1
```

`.index.is_unique` se index unique hai ya nahi ye check karte hain.

---

## 4. Series `read_csv()` se banana

```python
# Tareeqa 1: usecols + squeeze
subs = pd.read_csv('subs.csv').squeeze()

# Tareeqa 2: index_col ke saath
vk = pd.read_csv('kohli_ipl.csv', index_col='match_no').squeeze()

movies = pd.read_csv('bollywood.csv', index_col='movie').squeeze()
```

**`.squeeze()`** single-column DataFrame ko Series mein badal deta hai.
> Purane code mein `squeeze=True` parameter hota tha, ab wo deprecated hai — `.squeeze()` method use karo.

---

## 5. Series Methods

### 5.1 `head()` aur `tail()`
```python
subs.head()        # pehle 5
subs.head(3)       # pehle 3
subs.tail()        # aakhri 5
subs.tail(3)
```

### 5.2 `sample()`
```python
movies.sample()     # random 1
movies.sample(5)    # random 5
```

### 5.3 `value_counts()` ⭐
Har unique value kitni baar aayi hai — descending order mein.
```python
movies.value_counts()
# Akshay Kumar     48
# Amitabh Bachchan 45
# ...

movies.value_counts(normalize=True)   # percentage mein
```

### 5.4 `sort_values()` ⭐
```python
vk.sort_values()                      # ascending
vk.sort_values(ascending=False)       # descending
vk.sort_values(ascending=False).head(1)          # highest score
vk.sort_values(ascending=False).head(1).values[0] # sirf value

vk.sort_values(ascending=False, inplace=True)    # original badal do
```

### 5.5 `sort_index()`
```python
movies.sort_index()
movies.sort_index(ascending=False)
```

### 5.6 `inplace` parameter
`inplace=True` → original series modify ho jayegi, kuch return nahi hoga.
`inplace=False` (default) → nayi series return hogi, original safe.

---

## 6. Series Math Methods

```python
subs.count()        # non-null values ki count (NaN nahi ginta)
subs.sum()
subs.product()
subs.mean()
subs.median()
subs.mode()
subs.std()
subs.var()
subs.min()
subs.max()
subs.describe()     # sab ek saath: count, mean, std, min, 25%, 50%, 75%, max
```

**⚠️ Note:** `len()` saare elements ginta hai, `count()` sirf non-null.

---

## 7. Series with Python Functionality

### 7.1 Built-in functions
```python
len(marks_series)          # length
type(marks_series)         # pandas.core.series.Series
dir(marks_series)          # saare available methods
sorted(marks_series)       # sorted list return karega
min(marks_series)
max(marks_series)
```

### 7.2 Type conversion
```python
list(marks_series)         # list ban jayegi (sirf values)
dict(marks_series)         # dictionary (index → value)
```

### 7.3 Membership operator (`in`) ⚠️
```python
'maths' in marks_series      # True  → INDEX check karta hai
67 in marks_series           # False → values check NAHI karta!

67 in marks_series.values    # True  → values check karne ka sahi tareeqa
```

**Ye trap hai — `in` hamesha index pe kaam karta hai.**

### 7.4 Looping
```python
for i in marks_series:
    print(i)                 # VALUES print hongi

for i in marks_series.index:
    print(i)                 # INDEX print hoga

for index, value in marks_series.items():
    print(index, value)      # dono
```

### 7.5 Arithmetic operators (Broadcasting)
```python
100 - marks_series           # har value ko 100 se minus
marks_series + 10
marks_series * 2
```

### 7.6 Relational operators
```python
vk >= 50                     # boolean series
```

---

## 8. Boolean Indexing on Series ⭐⭐

Ye Pandas ka sabse powerful feature hai.

```python
# Kohli ne kitni baar 50+ score kiya?
(vk >= 50).sum()

# Kitni baar duck out hua (0 runs)?
(vk == 0).sum()

# Kitne din mein 200 se zyada subscribers mile?
(subs > 200).sum()

# Actual values dekho
vk[vk >= 50]
```

**Pattern:** `series[condition]` → sirf wo values jahan condition True hai.

**Multiple conditions:**
```python
vk[(vk >= 50) & (vk < 100)]     # fifties (not centuries)
vk[(vk == 0) | (vk == 1)]
```
`and/or/not` ki jagah `&`, `|`, `~` — aur har condition brackets mein.

---

## 9. Plotting on Series

```python
subs.plot()                          # line plot (time series ke liye)
movies.value_counts().head(20).plot(kind='bar')
movies.value_counts().head(10).plot(kind='pie')
vk.plot(kind='hist')
```

---

## 10. Important Series Methods (Supplementary Session)

### 10.1 `astype()` — memory optimization ⭐
```python
import sys
sys.getsizeof(vk)                       # pehle
vk = vk.astype('int16')
sys.getsizeof(vk)                       # ab bahut kam
```

### 10.2 `between()`
```python
vk[vk.between(51, 99)]                  # 51 se 99 ke beech (dono include)
vk[vk.between(51, 99)].size             # kitni fifties
```

### 10.3 `clip()`
```python
subs.clip(100, 200)     # 100 se kam → 100, 200 se zyada → 200
```

### 10.4 `drop_duplicates()`
```python
temp = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
temp.drop_duplicates()                  # pehla wala rakhta hai
temp.drop_duplicates(keep='last')       # aakhri wala rakhta hai
temp.duplicated()                       # boolean: duplicate hai ya nahi
temp.duplicated().sum()                 # kitne duplicates
```

### 10.5 `isnull()` / `dropna()` / `fillna()`
```python
temp = pd.Series([1, 2, 3, np.nan, 5, 6, np.nan, 8, np.nan, 10])

temp.isnull()          # boolean
temp.isnull().sum()    # kitne missing → 3
temp.count()           # non-null count → 7
temp.size              # total → 10

temp.dropna()          # missing rows hata do
temp.fillna(0)         # 0 se bharo
temp.fillna(temp.mean())   # mean se bharo ⭐ (common technique)
temp.fillna(method='ffill')  # pichli value se bharo
temp.fillna(method='bfill')  # agli value se bharo
```

### 10.6 `isin()`
```python
vk[vk.isin([49, 99])]     # sirf 49 aur 99 wale scores
```

### 10.7 `apply()` ⭐
Custom function har element pe apply karta hai.
```python
# Har movie ke actor ka sirf pehla naam
movies.apply(lambda x: x.split()[0].upper())

# Subscribers ko 'good day' / 'bad day' mein classify karo
subs.apply(lambda x: 'good day' if x > subs.mean() else 'bad day')
```

### 10.8 `copy()` — View vs Copy ⚠️
```python
new = vk.head()          # ye ek VIEW hai
new[1] = 100             # original vk BHI badal jayega!

new = vk.head().copy()   # ye ek COPY hai
new[1] = 100             # original safe hai
```

---
---

# SESSION 17 — PANDAS DATAFRAME

## 1. DataFrame kya hai?

2D labeled data structure — rows aur columns dono ke labels hote hain. Excel sheet ki tarah socho.

**DataFrame = Series ka collection** (har column ek Series hai).

---

## 2. DataFrame banana

### 2.1 Lists se
```python
student_data = [
    [100, 80, 10],
    [90, 70, 7],
    [120, 100, 14],
    [80, 50, 2]
]
pd.DataFrame(student_data, columns=['iq', 'marks', 'package'])
```

### 2.2 Dictionary se
```python
student_dict = {
    'name': ['nitish', 'ankit', 'rupesh', 'rishabh', 'amit', 'ankita'],
    'iq': [100, 90, 120, 80, 0, 0],
    'marks': [80, 70, 100, 50, 0, 0],
    'package': [10, 7, 14, 2, 0, 0]
}
students = pd.DataFrame(student_dict)
students.set_index('name', inplace=True)
```

### 2.3 CSV se ⭐ (real world mein yahi use hota hai)
```python
movies = pd.read_csv('movies.csv')
ipl = pd.read_csv('ipl-matches.csv')

# Useful parameters:
pd.read_csv('file.csv', index_col='name')
pd.read_csv('file.csv', usecols=['col1', 'col2'])
pd.read_csv('file.csv', nrows=100)
pd.read_csv('file.csv', sep='\t')          # tab separated
pd.read_csv('file.csv', encoding='latin-1')
pd.read_csv('file.csv', header=None)
pd.read_csv('file.csv', na_values=['?', '-'])
pd.read_csv('file.csv', parse_dates=['date_col'])
```

---

## 3. DataFrame Attributes

```python
movies.shape        # (rows, columns)
movies.dtypes       # har column ka datatype
movies.index        # row labels
movies.columns      # column names
movies.values       # NumPy 2D array
movies.size         # total cells
movies.ndim         # 2
```

---

## 4. DataFrame Methods

### 4.1 `head()` / `tail()` / `sample()`
```python
movies.head(2)
movies.tail(3)
ipl.sample(5)
```

### 4.2 `info()` ⭐
```python
movies.info()
```
Ye batata hai: column names, non-null count, dtype, memory usage. **Sabse pehle yahi chalao kisi naye dataset pe.**

### 4.3 `describe()` ⭐
```python
movies.describe()      # sirf numerical columns ka stats
```
count, mean, std, min, 25%, 50%, 75%, max.

### 4.4 `isnull()`
```python
movies.isnull()             # boolean DataFrame
movies.isnull().sum()       # har column mein kitne missing ⭐
```

### 4.5 `duplicated()`
```python
movies.duplicated().sum()   # kitni duplicate rows
students.drop_duplicates()
```

### 4.6 `rename()`
```python
students.rename(columns={'marks': 'percent', 'package': 'lpa'}, inplace=True)
students.rename(index={'nitish': 'Nitish', 'ankit': 'Ankit'}, inplace=True)
```

---

## 5. DataFrame Math Methods

```python
students.sum()             # column-wise (axis=0 default)
students.sum(axis=1)       # row-wise
students.mean()
students.median()
students.mode()
students.std()
students.var()
students.min()
students.max()
```

---

## 6. Selecting Columns aur Rows ⭐⭐

### 6.1 Single column (Series milegi)
```python
movies['title_x']
movies.title_x         # dot notation (space wale names mein kaam nahi karega)
```

### 6.2 Multiple columns (DataFrame milega)
```python
movies[['year_of_release', 'actors', 'title_x']]
```
**Note:** Double brackets `[[ ]]`.

### 6.3 Rows select karna — `iloc` aur `loc` ⭐⭐⭐

| | `iloc` | `loc` |
|---|---|---|
| Kaam | **Position** (integer) se | **Label** se |
| Slicing | End **exclude** hota hai | End **include** hota hai |
| Example | `df.iloc[0:5]` → 0,1,2,3,4 | `df.loc['a':'e']` → a,b,c,d,e |

```python
# iloc — integer position
movies.iloc[0]              # pehli row
movies.iloc[0:5]            # pehli 5 rows
movies.iloc[[0, 4, 5]]      # fancy indexing
movies.iloc[::2]            # har doosri row

# loc — label based
students.loc['nitish']
students.loc['nitish':'rishabh']       # rishabh BHI include hoga
students.loc[['nitish', 'ankita', 'rupesh']]
students.loc['nitish':'rishabh':2]
```

### 6.4 Rows aur Columns dono ek saath
```python
movies.iloc[0:3, 0:3]                          # position se
movies.loc[0:2, 'title_x':'poster_path']       # label se

students.iloc[0:3, 1:]
students.loc['nitish':'rishabh', 'iq':'marks']

# Mix karna ho toh:
movies.iloc[0:3][['title_x', 'actors']]
```

---

## 7. Filtering a DataFrame ⭐⭐⭐

Ye asli data analysis hai. Pattern: `df[condition]`

```python
ipl = pd.read_csv('ipl-matches.csv')
```

### Example 1: Final matches
```python
mask = ipl['MatchNumber'] == 'Final'
ipl[mask]
ipl[mask][['Season', 'WinningTeam']]      # sirf 2 columns
```

### Example 2: Ek team ne kitne match jeete
```python
ipl[ipl['WinningTeam'] == 'Chennai Super Kings'].shape[0]
```

### Example 3: Toss jeeta AUR match bhi jeeta (AND)
```python
ipl[(ipl['TossWinner'] == ipl['WinningTeam'])].shape[0]
```

### Example 4: Multiple conditions
```python
# 200+ runs se jeetne wale matches
ipl[(ipl['WonBy'] == 'Runs') & (ipl['Margin'] > 100)]

# Super 8 ya Final
ipl[ipl['MatchNumber'].isin(['Final', 'Qualifier 1'])]
```

### Example 5: Movies dataset
```python
# 8+ rating AND 10000+ votes
movies[(movies['imdb_rating'] > 8) & (movies['no_of_votes'] > 10000)]

# Action movies with 7.5+ rating
mask1 = movies['genres'].str.contains('Action')
mask2 = movies['imdb_rating'] > 7.5
movies[mask1 & mask2]
```

**Zaroori operators:**
```python
&    # AND
|    # OR
~    # NOT
.isin([...])            # multiple values mein se koi ek
.str.contains('text')   # text search
.between(a, b)          # range
```

---

## 8. Naye Columns add karna

### 8.1 Bilkul naya column
```python
movies['Country'] = 'India'
```

### 8.2 Existing columns se derive karke ⭐
```python
students['percentage'] = (students['marks'] / 100) * 100

movies['lead_actor'] = movies['actors'].str.split('|').apply(lambda x: x[0])

ipl['margin_type'] = np.where(ipl['WonBy'] == 'Runs', 'Batting First', 'Chasing')
```

### 8.3 `insert()` — specific position pe column
```python
students.insert(loc=1, column='gender', value=['M','M','M','M','F','F'])
```

---

## 9. `astype()` — Memory Optimization ⭐

```python
ipl.info()                         # pehle memory dekho

ipl['Season'] = ipl['Season'].astype('category')
ipl['Team1'] = ipl['Team1'].astype('category')
ipl['Team2'] = ipl['Team2'].astype('category')
ipl['ID'] = ipl['ID'].astype('int32')

ipl.info()                         # ab memory bahut kam
```

**Rule of thumb:** Jin columns mein kam unique values hain (jaise city names, categories), unko `category` bana do. Numbers ko sabse chhote possible int/float type mein daalo.

---
---

# SESSION 18 — IMPORTANT DATAFRAME METHODS

Setup:
```python
movies = pd.read_csv('imdb-top-1000.csv')
ipl = pd.read_csv('ipl-matches.csv')
batsman = pd.read_csv('batsman_runs_ipl.csv')
```

---

## 1. `value_counts()` ⭐

### Series pe
```python
marks.value_counts()
```

### DataFrame pe (multiple columns ka combination)
```python
ipl[['Team1', 'Team2', 'WinningTeam']].value_counts()
```

**Real use cases:**
```python
# Sabse zyada player of the match jeetne wale
ipl['Player_of_Match'].value_counts().head(10)

# Toss decision ka breakdown
ipl['TossDecision'].value_counts()

# Har season mein kitne match
ipl['Season'].value_counts()
```

---

## 2. `sort_values()` ⭐

```python
movies.sort_values('title_x')                       # ascending
movies.sort_values('title_x', ascending=False)      # descending

# Multiple columns
movies.sort_values(['year_of_release', 'title_x'], ascending=[True, False])

# NaN ko upar/neeche rakhna
movies.sort_values('imdb_rating', na_position='first')

movies.sort_values('imdb_rating', inplace=True)
```

---

## 3. `rank()`

```python
batsman['batting_rank'] = batsman['batsman_run'].rank(ascending=False)
batsman.sort_values('batting_rank')
```

Ranking assign karta hai. Ties ke liye average rank deta hai by default.

---

## 4. `sort_index()`

```python
marks_series.sort_index(ascending=False)
movies.sort_index()
```

---

## 5. `set_index()` aur `reset_index()` ⭐

```python
batsman.set_index('batter', inplace=True)      # column ko index bana do
batsman.reset_index(inplace=True)              # index ko wapas column bana do

# Index ko bilkul hata dena
batsman.reset_index(drop=True, inplace=True)
```

**Use case:** `value_counts()` ka result Series hota hai; `reset_index()` se DataFrame bana lete hain.
```python
ipl['Player_of_Match'].value_counts().reset_index()
```

---

## 6. `rename()`

```python
movies.rename(columns={'imdb_id': 'imdb', 'poster_path': 'link'}, inplace=True)
movies.rename(index={0: 'zero', 1: 'one'}, inplace=True)
```

---

## 7. `unique()` aur `nunique()` ⭐

```python
temp = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, np.nan, np.nan])

temp.unique()         # array with NaN included
temp.nunique()        # 4 (NaN ko count NAHI karta)
temp.nunique(dropna=False)   # 5

ipl['Season'].unique()
ipl['Season'].nunique()      # kitne seasons hue
```

**Farq:** `unique()` values deta hai, `nunique()` count deta hai. `unique()` NaN ginta hai, `nunique()` nahi.

---

## 8. `isnull()` / `notnull()` / `hasnans`

```python
students['name'].isnull()      # missing hai
students['name'].notnull()     # missing nahi hai
students['name'].hasnans       # koi missing hai kya? (True/False)

movies.isnull().sum()          # har column ka missing count ⭐
movies.isnull().sum().sum()    # total missing
```

---

## 9. `dropna()` ⭐

```python
students.dropna()                          # jis row mein bhi NaN ho, hata do
students.dropna(how='all')                 # sirf jab POORI row NaN ho
students.dropna(how='any')                 # default
students.dropna(subset=['name'])           # sirf 'name' column dekho
students.dropna(subset=['name', 'iq'])     # in dono mein se koi bhi NaN ho
students.dropna(axis=1)                    # NaN wale COLUMNS hata do
students.dropna(thresh=2)                  # kam se kam 2 non-null hone chahiye
```

---

## 10. `fillna()` ⭐

```python
students['name'].fillna('unknown')
students['package'].fillna(students['package'].mean())      # mean imputation
students['package'].fillna(students['package'].median())    # median (outliers ho toh better)
students['name'].fillna(students['name'].mode()[0])         # mode (categorical)

students.fillna(method='ffill')     # forward fill (pichli value)
students.fillna(method='bfill')     # backward fill (agli value)
students.fillna(0)
```

---

## 11. `drop_duplicates()` ⭐

```python
marks.drop_duplicates()
marks.drop_duplicates(keep='last')
marks.drop_duplicates(subset=['name'])       # sirf name column dekho

# Classic question: har IPL final ka winner
ipl.drop_duplicates(subset=['Season'], keep='last')[['Season', 'WinningTeam']]
```

`duplicated()` boolean deta hai:
```python
students.duplicated().sum()
```

---

## 12. `drop()` ⭐

```python
# Columns hatao
temp.drop(columns=['branch', 'cgpa'])
temp.drop(columns=['branch'], inplace=True)

# Rows hatao
temp.drop(index=[0, 1, 2])
students.drop(index=['nitish', 'aditya'])

# axis se
temp.drop('branch', axis=1)     # column
temp.drop(0, axis=0)            # row
```

---

## 13. `apply()` ⭐⭐

### Series pe
```python
points_df['distance'] = points_df['2d_point'].apply(euclidean)

# Simple lambda
movies['lead_actor'] = movies['actors'].apply(lambda x: x.split('|')[0])
```

### DataFrame pe (axis matter karta hai)
```python
def euclidean(row):
    pt_A = row['1st point']
    pt_B = row['2nd point']
    return ((pt_A[0] - pt_B[0])**2 + (pt_A[1] - pt_B[1])**2) ** 0.5

points_df['distance'] = points_df.apply(euclidean, axis=1)
```
`axis=1` → function ko har **row** milegi.
`axis=0` → function ko har **column** milega.

---

## 14. `isin()`

```python
ipl[ipl['Season'].isin(['2007/08', '2009', '2020/21'])]
students[students['branch'].isin(['EEE', 'ECE'])]
```

---

## 15. `corr()` — correlation

```python
movies.corr(numeric_only=True)
movies.corr(numeric_only=True)['imdb_rating'].sort_values(ascending=False)
```
-1 se 1 tak. 1 = perfect positive, -1 = perfect negative, 0 = koi relation nahi.

---

## 16. `nlargest()` aur `nsmallest()`

```python
movies.nlargest(5, 'imdb_rating')
movies.nsmallest(5, 'imdb_rating')
students.nlargest(3, 'package')
```

`sort_values().head(n)` se better hai — zyada efficient.

---

## 17. `insert()`

```python
movies.insert(loc=0, column='rank', value=range(1, len(movies)+1))
```

---

## 18. `copy()` — View vs Copy ⚠️

```python
new = movies.head()          # VIEW — original se juda
new = movies.head().copy()   # COPY — independent
```
Agar copy nahi banaya aur modify kiya toh `SettingWithCopyWarning` aata hai.

---

## 19. Extra useful methods

```python
df.replace({'old': 'new'})
df.astype({'col': 'int32'})
df.round(2)
df.abs()
df.cumsum()
df.diff()                    # consecutive difference
df.pct_change()              # percentage change
df.shift(1)                  # values ko shift karo
df.clip(lower, upper)
df.memory_usage(deep=True)
df.select_dtypes(include='number')
df.columns.tolist()
df.T                         # transpose
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx')
```

---

## Session 16-18 Quick Revision

| Kaam | Method |
|---|---|
| Data load karo | `read_csv`, `read_excel` |
| Pehli nazar | `head`, `tail`, `sample`, `info`, `describe`, `shape` |
| Column select | `df['col']`, `df[['a','b']]` |
| Row select | `iloc` (position), `loc` (label) |
| Filter | `df[mask]`, `&`, `\|`, `~`, `isin`, `str.contains`, `between` |
| Sort | `sort_values`, `sort_index`, `nlargest`, `nsmallest`, `rank` |
| Missing values | `isnull`, `dropna`, `fillna` |
| Duplicates | `duplicated`, `drop_duplicates` |
| Naya column | `df['new'] = ...`, `insert`, `apply` |
| Frequency | `value_counts`, `unique`, `nunique` |
| Index | `set_index`, `reset_index`, `rename` |
| Memory | `astype`, `memory_usage` |
