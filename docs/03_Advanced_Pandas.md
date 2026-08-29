# Advanced Pandas — Complete Notes
### CampusX Session 19–22 (GroupBy + Merging/Joining/Concatenating + MultiIndex + Vectorized String Ops & DateTime + Pivot Table)

---

# SESSION 19 — GROUPBY OBJECT IN PANDAS

## 1. GroupBy hai kya?

GroupBy ka concept **Split → Apply → Combine** hai:

1. **Split** — data ko groups mein baant do (kisi column ke basis pe)
2. **Apply** — har group pe koi function lagao (sum, mean, count...)
3. **Combine** — results ko wapas ek result mein jod do

**SQL ka `GROUP BY` yaad hai? Bilkul wahi cheez hai.**

```python
import pandas as pd
import numpy as np

movies = pd.read_csv('imdb-top-1000.csv')
```

---

## 2. GroupBy object banana

```python
genres = movies.groupby('Genre')
print(genres)
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x...>
```

**Note:** Ye khud kuch print nahi karta — ye ek "lazy" object hai. Aggregation lagane pe result deta hai.

---

## 3. Built-in Aggregation Functions

```python
genres.sum(numeric_only=True)
genres.mean(numeric_only=True)
genres.min()
genres.max()
genres.count()
genres.std()
genres.var()
genres.median()
genres.first()      # har group ki pehli row
genres.last()       # har group ki aakhri row
genres.nth(5)       # har group ki 6th row
genres.size()       # har group mein kitni rows ⭐
genres.describe()
genres.sample()
genres.nunique()
```

**Practical examples:**
```python
# Har genre ki total gross earning
movies.groupby('Genre')['Gross'].sum().sort_values(ascending=False)

# Har genre mein kitni movies
movies.groupby('Genre').size().sort_values(ascending=False)

# Har genre ki average IMDB rating
movies.groupby('Genre')['IMDB_Rating'].mean().sort_values(ascending=False)

# Top 3 genres by number of movies
movies['Genre'].value_counts().head(3)
```

---

## 4. GroupBy Attributes & Methods

### 4.1 `len()` — kitne groups bane
```python
len(movies.groupby('Genre'))       # kitne unique genres
movies['Genre'].nunique()          # same answer
```

### 4.2 `size()` — har group ka size
```python
movies.groupby('Genre').size()
```

### 4.3 `first()`, `last()`, `nth()`
```python
genres.first()
genres.last()
genres.nth(6)
```

### 4.4 `get_group()` ⭐
Ek particular group ka poora data nikaalo.
```python
genres.get_group('Fantasy')
# Barabar hai:
movies[movies['Genre'] == 'Fantasy']
```

### 4.5 `groups` attribute
```python
genres.groups
# Dictionary: {group_name: [index positions]}
```

### 4.6 `describe()`
```python
genres.describe()
```

### 4.7 `sample()`
```python
genres.sample(2, replace=True)      # har group se 2 random rows
```

### 4.8 `nunique()`
```python
genres.nunique()      # har group mein har column ki unique values
```

---

## 5. `agg()` — Multiple aggregations ek saath ⭐⭐

### 5.1 List pass karo
```python
genres.agg(['min', 'max', 'mean', 'sum'])
```

### 5.2 Dictionary pass karo (alag column, alag function)
```python
genres.agg({
    'Runtime': 'mean',
    'IMDB_Rating': 'mean',
    'No_of_Votes': 'sum',
    'Gross': 'sum',
    'Metascore': 'min'
})
```

### 5.3 Named aggregation (naye syntax)
```python
movies.groupby('Genre').agg(
    avg_rating=('IMDB_Rating', 'mean'),
    total_gross=('Gross', 'sum'),
    movie_count=('Series_Title', 'count')
)
```

---

## 6. GroupBy pe loop chalana

```python
for group_name, group_df in movies.groupby('Genre'):
    print(group_name)
    print(group_df.shape)
    print('---')
```

**Practical example — har genre ki top-rated movie:**
```python
df = pd.DataFrame(columns=movies.columns)

for group, data in movies.groupby('Genre'):
    df = pd.concat([df, data.sort_values('IMDB_Rating', ascending=False).head(1)])
```

---

## 7. `apply()` on GroupBy ⭐

Custom function har group pe apply karta hai.

```python
# Har genre ke andar har column ki unique value count
def foo(group):
    return group['IMDB_Rating'].max() - group['IMDB_Rating'].min()

movies.groupby('Genre').apply(foo)
```

**Popular example — har group ke andar percentage nikalna:**
```python
def rating_percent(group):
    group['rating_share'] = group['IMDB_Rating'] / group['IMDB_Rating'].sum()
    return group

movies.groupby('Genre').apply(rating_percent)
```

---

## 8. Multiple columns pe GroupBy ⭐

```python
duo = movies.groupby(['Director', 'Star1'])
duo.size()
duo['Gross'].sum().sort_values(ascending=False).head(5)

# Har director-actor jodi ki sabse zyada earning wali movie
duo['Gross'].max()
```

Result ek **MultiIndex Series** hota hai (Session 21 mein detail).

```python
# Access karna
duo.get_group(('Aamir Khan', 'Amole Gupte'))
```

---

## 9. GroupBy — Practical Questions (Colab exercises)

```python
# 1. Sabse zyada rated genre
movies.groupby('Genre')['IMDB_Rating'].mean().sort_values(ascending=False).head(1)

# 2. Har genre ka top director (by number of movies)
movies.groupby(['Genre', 'Director']).size().sort_values(ascending=False)

# 3. Har actor ki highest rated movie
movies.groupby('Star1')['IMDB_Rating'].max().sort_values(ascending=False).head(5)

# 4. Har genre mein kitne unique directors
movies.groupby('Genre')['Director'].nunique().sort_values(ascending=False)

# 5. Har genre ka runtime average
movies.groupby('Genre')['Runtime'].mean().sort_values(ascending=False)
```

---
---

# SESSION 20 — MERGING, JOINING & CONCATENATING

## 1. `pd.concat()` ⭐

Do ya zyada DataFrames ko **jodna** (stack karna).

### 1.1 Vertically (axis=0, default) — rows add
```python
courses = pd.read_csv('courses.csv')
students = pd.read_csv('students.csv')

pd.concat([df1, df2])                     # ek ke neeche ek
pd.concat([df1, df2], ignore_index=True)  # index reset kar do ⭐
```

### 1.2 Horizontally (axis=1) — columns add
```python
pd.concat([df1, df2], axis=1)
```

### 1.3 `keys` — MultiIndex banana
```python
multi = pd.concat([df1, df2], keys=['dec', 'nov'])
multi.loc['dec']            # sirf december ka data
multi.loc[('nov', 3)]       # november ka index 3
```

### 1.4 Sirf common columns rakhna
```python
pd.concat([df1, df2], join='inner')     # sirf common columns
pd.concat([df1, df2], join='outer')     # sab columns (default), missing → NaN
```

---

## 2. `pd.merge()` ⭐⭐⭐

SQL ke JOIN jaisa. Common column ke basis pe do tables jodna.

### Syntax
```python
pd.merge(left_df, right_df, how='inner', on='common_column')
```

### 2.1 Types of joins

| `how` | Kya karta hai |
|---|---|
| `'inner'` | Sirf wo rows jo **dono** mein hain (default) |
| `'left'` | Left ki saari rows + matching right |
| `'right'` | Right ki saari rows + matching left |
| `'outer'` | Dono ki saari rows (union) |
| `'cross'` | Cartesian product |

```python
students.merge(regs, how='inner', on='student_id')
students.merge(regs, how='left', on='student_id')
students.merge(regs, how='right', on='student_id')
students.merge(regs, how='outer', on='student_id')
```

### 2.2 Alag naam wale columns pe merge
```python
pd.merge(df1, df2, left_on='id', right_on='student_id')
```

### 2.3 Index pe merge
```python
pd.merge(df1, df2, left_index=True, right_index=True)
```

### 2.4 Suffixes (jab column names clash karein)
```python
pd.merge(df1, df2, on='id', suffixes=('_left', '_right'))
```

### 2.5 `indicator`
```python
pd.merge(df1, df2, how='outer', on='id', indicator=True)
# Ek '_merge' column aata hai: 'both', 'left_only', 'right_only'
```

---

## 3. Relationship Types

| Type | Example |
|---|---|
| **One to One** | Har student ka ek hi roll number |
| **One to Many** | Ek course mein kai students |
| **Many to Many** | Kai students, kai courses (registration table se) |

---

## 4. `df.join()`

Index pe merge karne ka shortcut.
```python
df1.join(df2)                       # index pe left join
df1.join(df2, how='inner')
df1.join(df2, on='key_column')
```

---

## 5. Merging — Practical Questions (Colab exercises)

Datasets: `courses.csv`, `students.csv`, `nov.csv`, `dec.csv`, `matches.csv`, `deliveries.csv`

```python
# 1. Kitne students ne kabhi enroll nahi kiya
students.merge(regs, how='left', on='student_id')['name'].isnull().sum()

# 2. Kaunse course mein koi student nahi
courses.merge(regs, how='left', on='course_id')['student_id'].isnull().sum()

# 3. Total revenue generated
regs.merge(courses, how='inner', on='course_id')['price'].sum()

# 4. Month by month revenue
temp = pd.concat([nov, dec], keys=['Nov', 'Dec']).reset_index()
temp.merge(courses, on='course_id').groupby('level_0')['price'].sum()

# 5. Har course ka revenue
regs.merge(courses, on='course_id').groupby('course_name')['price'].sum().sort_values(ascending=False)

# 6. Sabse zyada revenue dene wala student
regs.merge(students, on='student_id').merge(courses, on='course_id') \
    .groupby('name')['price'].sum().sort_values(ascending=False).head(1)

# 7. Har course ka average rating
# 8. Kaunsa course sabse popular hai
regs.merge(courses, on='course_id')['course_name'].value_counts()

# 9. IPL: har batsman ka total run
deliveries.merge(matches, left_on='match_id', right_on='id') \
    .groupby('batter')['batsman_runs'].sum().sort_values(ascending=False)

# 10. Kohli ne kis stadium mein sabse zyada run banaye
new = deliveries.merge(matches, left_on='match_id', right_on='id')
new[new['batter'] == 'V Kohli'].groupby('venue')['batsman_runs'].sum() \
   .sort_values(ascending=False).head(1)
```

---
---

# SESSION 21 — MULTIINDEX SERIES & DATAFRAMES

## 1. MultiIndex kyun chahiye?

Kabhi kabhi humein **higher-dimensional data** ko 2D table mein represent karna hota hai. MultiIndex (Hierarchical Index) se ek se zyada level ka index bana sakte hain.

**Example:** Har company ke har saal ka data → index = (year, company)

---

## 2. MultiIndex banana

### 2.1 `from_tuples()`
```python
index_val = [('cse', 2019), ('cse', 2020), ('cse', 2021), ('cse', 2022),
             ('ece', 2019), ('ece', 2020), ('ece', 2021), ('ece', 2022)]

multiindex = pd.MultiIndex.from_tuples(index_val)
multiindex.levels[0]     # ['cse', 'ece']
multiindex.levels[1]     # [2019, 2020, 2021, 2022]
```

### 2.2 `from_product()` (aasan tareeqa)
```python
pd.MultiIndex.from_product([['cse', 'ece'], [2019, 2020, 2021, 2022]])
```

---

## 3. MultiIndex Series

```python
s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=multiindex)
s['cse']              # cse ka poora data
s['cse'][2019]        # ek value
s[('cse', 2019)]      # same
```

---

## 4. `unstack()` aur `stack()` ⭐

- **`unstack()`** → inner index level ko **columns** bana deta hai (Series → DataFrame)
- **`stack()`** → columns ko wapas index bana deta hai (DataFrame → Series)

```python
temp = s.unstack()      # 2D table ban gaya
temp.stack()            # wapas MultiIndex Series
```

Ye ek dusre ka **inverse** hain.

---

## 5. MultiIndex DataFrame

```python
branch_df = pd.DataFrame(
    [
        [1, 2], [3, 4], [5, 6], [7, 8],
        [9, 10], [11, 12], [13, 14], [15, 16]
    ],
    index=multiindex,
    columns=['avg_package', 'students']
)
```

### Columns pe bhi MultiIndex
```python
branch_df = pd.DataFrame(
    [[1,2,0,0],[3,4,0,0],[5,6,0,0],[7,8,0,0],
     [9,10,0,0],[11,12,0,0],[13,14,0,0],[15,16,0,0]],
    index=multiindex,
    columns=pd.MultiIndex.from_product([['delhi','mumbai'], ['avg_package','students']])
)
```

Ab **rows aur columns dono** MultiIndex hain.

---

## 6. MultiIndex DataFrame ke saath kaam

```python
branch_df.head()
branch_df.shape
branch_df.info()
branch_df.duplicated().sum()
branch_df.isnull().sum()

# Extracting rows
branch_df.loc['cse']
branch_df.loc[('cse', 2022)]
branch_df.loc[('cse', 2019):('ece', 2020)]

# Extracting columns
branch_df['delhi']
branch_df['delhi']['students']
branch_df.iloc[:, 1:3]

# Transpose
branch_df.transpose()
```

### `sort_index()` on MultiIndex ⭐
```python
branch_df.sort_index(ascending=False)
branch_df.sort_index(ascending=[False, True])
branch_df.sort_index(level=0, ascending=False)
```

### `swaplevel()`
```python
branch_df.swaplevel(axis=0)      # index levels swap
branch_df.swaplevel(axis=1)      # column levels swap
```

---

## 7. Long vs Wide Data ⭐⭐

| Wide Format | Long Format |
|---|---|
| Har variable ka apna column | Ek 'variable' aur ek 'value' column |
| Insaan ke liye padhne mein aasan | Machines/plotting ke liye better |
| Excel jaisa | Database jaisa |

**Example — Wide:**
```
branch   2019   2020   2021
cse       1      3      5
ece       9     11     13
```

**Example — Long:**
```
branch   year   students
cse      2019      1
cse      2020      3
cse      2021      5
ece      2019      9
```

---

## 8. `melt()` — Wide se Long ⭐

```python
pd.melt(df, id_vars=['branch'], var_name='year', value_name='students')
```

**Real example — COVID data:**
```python
death = pd.read_csv('time_series_covid19_deaths_global.csv')
confirm = pd.read_csv('time_series_covid19_confirmed_global.csv')

death = death.melt(id_vars=['Province/State','Country/Region','Lat','Long'],
                   var_name='date', value_name='num_deaths')

confirm = confirm.melt(id_vars=['Province/State','Country/Region','Lat','Long'],
                       var_name='date', value_name='num_cases')

final = confirm.merge(death, on=['Province/State','Country/Region','Lat','Long','date'])

# Ab analysis aasan:
final.groupby('Country/Region')['num_cases'].max().sort_values(ascending=False)
```

---

## 9. `pivot_table()` ⭐⭐⭐

Long data ko Wide mein badalne ka aur summary banane ka sabse powerful tool. Excel ke Pivot Table jaisa.

### Syntax
```python
df.pivot_table(index=..., columns=..., values=..., aggfunc=...)
```

```python
import seaborn as sns
df = sns.load_dataset('tips')

# Har gender ka har day pe average bill
df.pivot_table(index='sex', columns='day', values='total_bill')

# aggfunc badlo
df.pivot_table(index='sex', columns='day', values='total_bill', aggfunc='sum')
df.pivot_table(index='sex', columns='day', values='total_bill', aggfunc='count')
df.pivot_table(index='sex', columns='day', values='total_bill', aggfunc='std')

# Multiple index / columns
df.pivot_table(index=['sex', 'smoker'], columns=['day', 'time'],
               values='total_bill', aggfunc='mean')

# margins (Grand Total row/column)
df.pivot_table(index='sex', columns='day', values='total_bill',
               aggfunc='sum', margins=True)

# Multiple values
df.pivot_table(index='sex', columns='day', values=['total_bill', 'tip'], aggfunc='mean')

# Missing values
df.pivot_table(index='sex', columns='day', values='tip', fill_value=0)
```

### Plotting a pivot table ⭐
```python
expense = pd.read_csv('expense_data.csv')
expense['Date'] = pd.to_datetime(expense['Date'])
expense['month'] = expense['Date'].dt.month_name()

expense.pivot_table(index='month', columns='Category',
                    values='INR', aggfunc='sum', fill_value=0).plot()
```

**`pivot()` vs `pivot_table()`:**
- `pivot()` — koi aggregation nahi karta, duplicate index-column combination pe error deta hai
- `pivot_table()` — aggregation karta hai, duplicates handle kar leta hai ✅

---
---

# SESSION 22 — VECTORIZED STRING OPERATIONS & DATETIME

# PART A — VECTORIZED STRING OPERATIONS (`.str`)

## 1. Problem kya hai?

Normal Python mein string functions ek string pe chalte hain. Poore column pe chalane ke liye loop lagana padta — **slow**.

Pandas mein `.str` accessor hai jo poore column pe **vectorized** string operation karta hai.

```python
s = pd.Series(['cat', 'mat', None, 'rat'])
s.str.startswith('c')
```

Setup:
```python
df = pd.read_csv('titanic.csv')
```

---

## 2. Common `.str` methods

### 2.1 Case change
```python
df['Name'].str.lower()
df['Name'].str.upper()
df['Name'].str.capitalize()      # Pehla letter capital
df['Name'].str.title()           # Har word ka pehla letter capital
df['Name'].str.swapcase()
```

### 2.2 `len()`
```python
df['Name'].str.len()                          # naam kitna lamba
df[df['Name'].str.len() == df['Name'].str.len().max()]   # sabse lamba naam
```

### 2.3 `strip()` — whitespace hatana
```python
df['Name'].str.strip()      # dono taraf se
df['Name'].str.lstrip()     # left
df['Name'].str.rstrip()     # right
```

### 2.4 `split()` aur `get()` ⭐
```python
# Titanic naam format: "Braund, Mr. Owen Harris"
df['lastname'] = df['Name'].str.split(',').str.get(0)
df['title'] = df['Name'].str.split(',').str.get(1).str.strip().str.split('.').str.get(0)

df['title'].value_counts()
# Mr, Miss, Mrs, Master, Dr, Rev...

# expand=True se alag columns mil jaate hain
df['Name'].str.split(',', expand=True)
```

### 2.5 `replace()` ⭐
```python
df['title'] = df['title'].str.replace('Ms', 'Miss')
df['title'] = df['title'].str.replace('Mlle', 'Miss')
```

### 2.6 Filtering ke liye string methods ⭐
```python
df[df['Name'].str.startswith('A')]
df[df['Name'].str.endswith('n')]
df[df['Name'].str.contains('Mr')]
df[df['Name'].str.contains('John|James')]     # regex OR
```

### 2.7 Check functions
```python
df['Name'].str.isdigit()
df['Name'].str.isalpha()
df['Name'].str.isalnum()
df['Name'].str.isupper()
df['Name'].str.islower()
df['Name'].str.isspace()
```

### 2.8 Slicing
```python
df['Name'].str[::-1]        # reverse
df['Name'].str[0:5]         # pehle 5 characters
df['Name'].str[0]           # pehla character
```

### 2.9 `cat()` — join karna
```python
df['Name'].str.cat(sep=', ')
```

### 2.10 Padding
```python
df['Name'].str.pad(width=50, side='left', fillchar='*')
df['Name'].str.zfill(10)
```

---

## 3. Regex ke saath `.str` ⭐

```python
# Jinke naam mein 'chidambaram' pattern hai
df[df['Name'].str.contains('^[A-Z][a-z]+$', regex=True)]

# Extract karna
df['Name'].str.extract(r'([A-Za-z]+)\.')      # title extract

# Kitni baar pattern aaya
df['Name'].str.count('a')

# Find
df['Name'].str.find('Mr')

# findall
df['Name'].str.findall(r'[A-Z]')
```

---

# PART B — DATETIME IN PANDAS

## 1. `Timestamp` object

Pandas ka datetime object.

```python
# Alag alag tareeqe
pd.Timestamp('2023/1/5')
pd.Timestamp('2023-1-5')
pd.Timestamp('2023, 1, 5')
pd.Timestamp('5th January 2023')
pd.Timestamp('5th January 2023 9:21AM')

import datetime as dt
pd.Timestamp(dt.datetime(2023, 1, 5, 9, 21, 56))
```

### Attributes nikalna
```python
x = pd.Timestamp('2023/1/5')
x.year        # 2023
x.month       # 1
x.day         # 5
x.hour
x.minute
x.second
x.dayofweek   # 0 = Monday
x.day_name()  # 'Thursday'
x.month_name()
x.is_month_start
x.is_month_end
x.quarter
```

### Current time
```python
pd.Timestamp('today')
pd.Timestamp('now')
dt.datetime.now()
```

---

## 2. `DatetimeIndex`

```python
dt_index = pd.DatetimeIndex(['2023/1/1', '2022/1/1', '2021/1/1'])
pd.Series([1, 2, 3], index=dt_index)
```

---

## 3. `date_range()` ⭐

Dates ki series generate karta hai.

```python
pd.date_range(start='2023/1/5', end='2023/2/28', freq='D')     # daily
pd.date_range(start='2023/1/5', end='2023/2/28', freq='3D')    # har 3rd din
pd.date_range(start='2023/1/5', end='2023/2/28', freq='B')     # business days
pd.date_range(start='2023/1/5', end='2023/2/28', freq='W')     # weekly (Sunday)
pd.date_range(start='2023/1/5', end='2023/2/28', freq='W-THU') # har Thursday
pd.date_range(start='2023/1/5', end='2023/2/28', freq='h')     # hourly
pd.date_range(start='2023/1/5', end='2023/12/31', freq='ME')   # month end
pd.date_range(start='2023/1/5', end='2023/12/31', freq='MS')   # month start
pd.date_range(start='2023/1/5', end='2030/12/31', freq='YS')   # year start

pd.date_range(start='2023/1/5', periods=25, freq='D')          # count se
```

**Common freq codes:** `D` day, `B` business day, `W` week, `h` hour, `min` minute, `s` second, `ME`/`MS` month end/start, `QE`/`QS` quarter, `YE`/`YS` year.

---

## 4. `to_datetime()` ⭐⭐

String column ko datetime column mein convert karta hai. **Real world mein sabse zyada use hota hai.**

```python
s = pd.Series(['2023/1/1', '2022/1/1', '2021/1/1'])
pd.to_datetime(s)

# Errors handle karna
pd.to_datetime(s, errors='coerce')       # galat values → NaT

# Format specify karna (fast + safe)
pd.to_datetime(s, format='%Y/%m/%d')

# read_csv ke waqt hi
pd.read_csv('data.csv', parse_dates=['date'])
```

---

## 5. `dt` accessor ⭐⭐

Ek datetime column se saari properties nikaalne ke liye.

```python
df = pd.read_csv('messages.csv')
df['date'] = pd.to_datetime(df['date'])

df['date'].dt.year
df['date'].dt.month
df['date'].dt.month_name()
df['date'].dt.day
df['date'].dt.day_name()          # Monday, Tuesday...
df['date'].dt.dayofweek
df['date'].dt.hour
df['date'].dt.minute
df['date'].dt.second
df['date'].dt.quarter
df['date'].dt.is_month_start
df['date'].dt.is_month_end
df['date'].dt.is_quarter_start
df['date'].dt.is_year_start
df['date'].dt.is_leap_year
df['date'].dt.days_in_month
df['date'].dt.date                # sirf date part
df['date'].dt.time                # sirf time part
```

---

## 6. Real-world DateTime Examples ⭐

```python
# Har mahine mein kitne messages
df['month_name'] = df['date'].dt.month_name()
df.groupby('month_name').size().plot(kind='bar')

# Har din ke messages
df['day_name'] = df['date'].dt.day_name()
df.groupby('day_name').size().plot(kind='bar')

# Weekend vs weekday
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])

# Saal ke hisaab se trend
df.groupby(df['date'].dt.year).size().plot()

# Stock data — sirf 2023 ka
stocks[stocks['date'].dt.year == 2023]

# Date range filter
mask = (df['date'] >= '2023-01-01') & (df['date'] <= '2023-06-30')
df[mask]
```

---

## 7. Timedelta — dates ka difference

```python
d1 = pd.Timestamp('2023/1/1')
d2 = pd.Timestamp('2023/6/15')

d2 - d1                     # Timedelta('165 days')
(d2 - d1).days              # 165

pd.Timedelta(days=10)
d1 + pd.Timedelta(days=30)

# DataFrame mein
df['duration'] = df['end_date'] - df['start_date']
df['duration_days'] = df['duration'].dt.days
```

---

## 8. Time Series ke special features

```python
# Date ko index bana do
df.set_index('date', inplace=True)

# Ab partial string indexing kaam karti hai
df['2023']              # poora 2023
df['2023-01']           # sirf January 2023
df['2023-01-05']        # ek din

# Resample ⭐ — time-based groupby
df['sales'].resample('ME').sum()      # monthly total
df['sales'].resample('W').mean()      # weekly average
df['sales'].resample('YE').sum()      # yearly

# Rolling window (moving average)
df['sales'].rolling(window=7).mean()  # 7-day moving average
df['sales'].expanding().mean()        # cumulative average
```

---

## Advanced Pandas — Quick Revision

| Kaam | Tool |
|---|---|
| Group-wise summary | `groupby().agg()` |
| Ek group nikaalna | `get_group()` |
| Group pe custom function | `groupby().apply()` |
| Tables jodna (stack) | `pd.concat()` |
| Tables jodna (SQL join) | `pd.merge()`, `df.join()` |
| Hierarchical index | `pd.MultiIndex`, `stack`, `unstack`, `swaplevel` |
| Wide → Long | `melt()` |
| Long → Wide + summary | `pivot_table()` |
| String operations | `.str.*` |
| Date operations | `pd.to_datetime()`, `.dt.*`, `date_range()` |
| Time-based grouping | `resample()`, `rolling()` |
