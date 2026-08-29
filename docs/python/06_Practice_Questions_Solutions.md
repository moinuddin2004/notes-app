---
title: Practice Questions & Solutions
tags:
  - python
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - practice
  - exercises
  - case-study
  - data-science
  - campusx
  - notes
date: 2026-08-29
---

# Practice Questions + Solutions
### CampusX Tasks (Task 13–22) aur Case Studies — sab ek jagah

Ye wo questions hain jo CampusX ke har session ke baad "Task" notebooks mein diye jaate hain. Solve karke dekho, phir solution check karo.

---

# PART A — NUMPY PRACTICE

```python
import numpy as np
```

**Q1. 3x3 ka array banao jismein 0 se 8 tak numbers hon.**
```python
np.arange(9).reshape(3, 3)
```

**Q2. 5x5 identity matrix banao.**
```python
np.identity(5)
```

**Q3. Ek array mein saari odd values ko -1 se replace karo (original change kiye bina).**
```python
a = np.arange(10)
out = np.where(a % 2 == 1, -1, a)
```

**Q4. Do arrays mein common elements dhoondo.**
```python
np.intersect1d(a, b)
```

**Q5. `a` mein se wo elements hatao jo `b` mein hain.**
```python
np.setdiff1d(a, b)
```

**Q6. Array mein 5 aur 10 ke beech ki saari values nikaalo.**
```python
a[(a >= 5) & (a <= 10)]
```

**Q7. 2D array ke do columns swap karo.**
```python
arr[:, [1, 0, 2]]
```

**Q8. 2D array ki rows reverse karo.**
```python
arr[::-1]
```

**Q9. 5 se 10 ke beech random decimals ka 5x3 array banao.**
```python
np.random.uniform(5, 10, size=(5, 3))
```

**Q10. Array ko 3 decimal places tak print karo.**
```python
np.set_printoptions(precision=3)
```

**Q11. Array mein sabse zyada aane wali value dhoondo.**
```python
vals, counts = np.unique(a, return_counts=True)
vals[np.argmax(counts)]
```

**Q12. Array ko normalize karo (0 se 1 ke beech).**
```python
(a - a.min()) / (a.max() - a.min())
```

**Q13. Array ke top 5 values ke indices nikaalo.**
```python
np.argsort(a)[-5:]
```

**Q14. Array mein NaN ki jagah column ka mean daalo.**
```python
col_mean = np.nanmean(a, axis=0)
idx = np.where(np.isnan(a))
a[idx] = np.take(col_mean, idx[1])
```

**Q15. Do arrays ka correlation nikaalo.**
```python
np.corrcoef(a, b)[0, 1]
```

**Q16. Array mein har row ka max element ka index nikaalo.**
```python
np.argmax(a, axis=1)
```

**Q17. Ek 1D array ko column vector banao.**
```python
a.reshape(-1, 1)
# ya
np.expand_dims(a, axis=1)
```

**Q18. Sigmoid function banao aur -10 se 10 tak plot karo.**
```python
x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y)
```

---

# PART B — PANDAS SERIES/DATAFRAME PRACTICE

```python
import pandas as pd
```

**Q1. Kohli ne kitne matches mein 50+ score kiya?**
```python
(vk >= 50).sum()
```

**Q2. Kohli kitni baar duck out (0 runs) hua?**
```python
(vk == 0).sum()
```

**Q3. Kohli ka highest score aur wo kis match mein?**
```python
vk.sort_values(ascending=False).head(1)
```

**Q4. Kitne din mein 200 se zyada subscribers mile?**
```python
(subs > 200).sum()
```

**Q5. Sabse zyada movies kis actor ne ki?**
```python
movies.value_counts().head(1)
```

**Q6. Kohli ka batting average kya hai? (assuming not-outs nahi)**
```python
vk.mean()
```

**Q7. IPL mein sabse zyada 'Player of the Match' kisne jeeta?**
```python
ipl['Player_of_Match'].value_counts().head(1)
```

**Q8. Toss jeetne wali team ne kitni baar match bhi jeeta?**
```python
(ipl['TossWinner'] == ipl['WinningTeam']).sum()
```

**Q9. Har season ka final winner nikaalo.**
```python
ipl.drop_duplicates(subset=['Season'], keep='last')[['Season', 'WinningTeam']]
```

**Q10. Kis city mein sabse zyada match hue?**
```python
ipl['City'].value_counts().head(1)
```

**Q11. Movies dataset mein 8+ rating aur 10000+ votes wali movies.**
```python
movies[(movies['imdb_rating'] > 8) & (movies['no_of_votes'] > 10000)]
```

**Q12. Har movie ka lead actor nikaalo.**
```python
movies['lead_actor'] = movies['actors'].str.split('|').str[0]
```

**Q13. Sabse lambi runtime wali movie.**
```python
movies.nlargest(1, 'Runtime')
```

**Q14. Har column mein kitne missing values hain?**
```python
movies.isnull().sum()
```

**Q15. Sabse zyada missing values wala column.**
```python
movies.isnull().sum().sort_values(ascending=False).head(1)
```

**Q16. Ek column ke missing values ko mean se bharo.**
```python
movies['Gross'].fillna(movies['Gross'].mean(), inplace=True)
```

**Q17. Duplicate rows kitni hain aur unko hata do.**
```python
movies.duplicated().sum()
movies.drop_duplicates(inplace=True)
```

**Q18. DataFrame ki memory usage optimize karo.**
```python
for col in df.select_dtypes(include='object'):
    if df[col].nunique() / len(df) < 0.5:
        df[col] = df[col].astype('category')

for col in df.select_dtypes(include='int64'):
    df[col] = pd.to_numeric(df[col], downcast='integer')
```

---

# PART C — GROUPBY PRACTICE

**Q1. Kaunsa genre sabse zyada earning karta hai?**
```python
movies.groupby('Genre')['Gross'].sum().sort_values(ascending=False).head(1)
```

**Q2. Kis genre ki average rating sabse high hai?**
```python
movies.groupby('Genre')['IMDB_Rating'].mean().sort_values(ascending=False).head(1)
```

**Q3. Har genre mein kitni movies hain?**
```python
movies.groupby('Genre').size().sort_values(ascending=False)
```

**Q4. Har genre ki top rated movie.**
```python
movies.loc[movies.groupby('Genre')['IMDB_Rating'].idxmax()]
```

**Q5. Har director ne kitni movies banayi aur unka total gross?**
```python
movies.groupby('Director').agg(
    movie_count=('Series_Title', 'count'),
    total_gross=('Gross', 'sum')
).sort_values('total_gross', ascending=False)
```

**Q6. Sabse successful director-actor jodi.**
```python
movies.groupby(['Director', 'Star1'])['Gross'].sum().sort_values(ascending=False).head(1)
```

**Q7. Har genre mein kitne unique directors hain?**
```python
movies.groupby('Genre')['Director'].nunique().sort_values(ascending=False)
```

**Q8. Har genre ki minimum aur maximum runtime.**
```python
movies.groupby('Genre')['Runtime'].agg(['min', 'max', 'mean'])
```

**Q9. Har genre ka contribution total gross mein (%).**
```python
g = movies.groupby('Genre')['Gross'].sum()
(g / g.sum() * 100).sort_values(ascending=False)
```

**Q10. Har year mein sabse zyada rated movie.**
```python
movies.loc[movies.groupby('Released_Year')['IMDB_Rating'].idxmax()]
```

---

# PART D — MERGING PRACTICE (Session 20 tasks)

Datasets: `courses.csv`, `students.csv`, `regs.csv`, `nov.csv`, `dec.csv`

**Q1. Kitne students ne kabhi enroll nahi kiya?**
```python
students.merge(regs, how='left', on='student_id')['course_id'].isnull().sum()
```

**Q2. Kaunse courses mein koi enrollment nahi hui?**
```python
temp = courses.merge(regs, how='left', on='course_id')
temp[temp['student_id'].isnull()]['course_name']
```

**Q3. Total revenue kitna generate hua?**
```python
regs.merge(courses, on='course_id')['price'].sum()
```

**Q4. Month-by-month revenue.**
```python
temp = pd.concat([nov, dec], keys=['Nov', 'Dec']).reset_index()
temp.merge(courses, on='course_id').groupby('level_0')['price'].sum()
```

**Q5. Har course ka revenue.**
```python
regs.merge(courses, on='course_id') \
    .groupby('course_name')['price'].sum().sort_values(ascending=False)
```

**Q6. Sabse zyada paise kharch karne wala student.**
```python
regs.merge(students, on='student_id') \
    .merge(courses, on='course_id') \
    .groupby('name')['price'].sum() \
    .sort_values(ascending=False).head(1)
```

**Q7. Sabse popular course.**
```python
regs.merge(courses, on='course_id')['course_name'].value_counts().head(1)
```

**Q8. Har partner ke kitne students hain?**
```python
students.groupby('partner').size().sort_values(ascending=False)
```

**Q9. Har partner ka average student enrollment.**
```python
students.merge(regs, on='student_id').groupby('partner').size()
```

**Q10. IPL: har batsman ka total run.**
```python
deliveries.merge(matches, left_on='match_id', right_on='id') \
    .groupby('batter')['batsman_runs'].sum() \
    .sort_values(ascending=False).head(10)
```

**Q11. Kohli ne kis venue pe sabse zyada run banaye?**
```python
new = deliveries.merge(matches, left_on='match_id', right_on='id')
new[new['batter'] == 'V Kohli'].groupby('venue')['batsman_runs'].sum() \
   .sort_values(ascending=False).head(1)
```

**Q12. Har season mein kaunsa batsman top scorer tha?**
```python
new = deliveries.merge(matches, left_on='match_id', right_on='id')
temp = new.groupby(['Season', 'batter'])['batsman_runs'].sum().reset_index()
temp.loc[temp.groupby('Season')['batsman_runs'].idxmax()]
```

---

# PART E — STRING & DATETIME PRACTICE (Session 22)

**Q1. Titanic mein sabse lamba naam kiska hai?**
```python
df[df['Name'].str.len() == df['Name'].str.len().max()]['Name']
```

**Q2. Har passenger ka title (Mr/Mrs/Miss...) extract karo.**
```python
df['title'] = df['Name'].str.split(',').str[1].str.strip().str.split('.').str[0]
df['title'].value_counts()
```

**Q3. Sirf wo passengers jinke naam 'A' se shuru hote hain.**
```python
df[df['Name'].str.startswith('A')]
```

**Q4. Naam mein 'John' hai aisi rows.**
```python
df[df['Name'].str.contains('John')]
```

**Q5. Last name alag column mein nikaalo.**
```python
df['lastname'] = df['Name'].str.split(',').str[0]
```

**Q6. Sabse common last name.**
```python
df['lastname'].value_counts().head(1)
```

**Q7. Date column ko datetime mein convert karo.**
```python
df['date'] = pd.to_datetime(df['date'])
```

**Q8. Har mahine ke messages count karo aur plot karo.**
```python
df['month'] = df['date'].dt.month_name()
df.groupby('month').size().plot(kind='bar')
```

**Q9. Weekend pe kitne messages aaye?**
```python
df[df['date'].dt.dayofweek.isin([5, 6])].shape[0]
```

**Q10. Kis hour mein sabse zyada activity hoti hai?**
```python
df['date'].dt.hour.value_counts().sort_index().plot(kind='bar')
```

**Q11. Sirf 2023 ka data filter karo.**
```python
df[df['date'].dt.year == 2023]
```

**Q12. Do dates ke beech ka difference days mein.**
```python
(df['end'] - df['start']).dt.days
```

---

# PART F — PIVOT TABLE PRACTICE

**Q1. Har gender ka har din ka average bill.**
```python
tips.pivot_table(index='sex', columns='day', values='total_bill')
```

**Q2. Smoker vs Non-smoker ka tip percentage.**
```python
tips['tip_pct'] = tips['tip'] / tips['total_bill'] * 100
tips.pivot_table(index='smoker', columns='time', values='tip_pct')
```

**Q3. Har din, har time slot mein kitne customers.**
```python
tips.pivot_table(index='day', columns='time', values='size', aggfunc='sum', margins=True)
```

**Q4. Multiple aggregations ek saath.**
```python
tips.pivot_table(index='day', values='total_bill', aggfunc=['mean', 'sum', 'count'])
```

**Q5. Monthly expense category-wise (plot ke saath).**
```python
expense['month'] = expense['Date'].dt.month_name()
expense.pivot_table(index='month', columns='Category', values='INR',
                    aggfunc='sum', fill_value=0).plot(kind='bar', stacked=True)
```

---

# PART G — VISUALIZATION PRACTICE

**Q1. Kohli aur Rohit ka career comparison line chart.**
```python
plt.figure(figsize=(12, 6))
plt.plot(df['index'], df['V Kohli'], marker='o', label='Virat Kohli')
plt.plot(df['index'], df['RG Sharma'], marker='s', label='Rohit Sharma')
plt.title('Career Comparison')
plt.xlabel('Season'); plt.ylabel('Runs')
plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

**Q2. Batting average vs strike rate ka scatter plot (bubble size = runs).**
```python
plt.figure(figsize=(10, 6))
plt.scatter(df['avg'], df['strike_rate'], s=df['runs']/10, alpha=0.5, c=df['runs'], cmap='viridis')
plt.colorbar(label='Runs')
plt.xlabel('Average'); plt.ylabel('Strike Rate')
plt.show()
```

**Q3. IPL teams ki titles ka bar chart.**
```python
plt.figure(figsize=(12, 6))
plt.bar(df['IPL Team'], df['Titles'], color='teal')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
```

**Q4. Kohli ke scores ka histogram.**
```python
plt.hist(vk['batsman_runs'], bins=20, edgecolor='black')
plt.xlabel('Runs'); plt.ylabel('Frequency')
```

**Q5. 2x2 dashboard banao.**
```python
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
ax[0,0].plot(x, y);          ax[0,0].set_title('Trend')
ax[0,1].scatter(x, y);       ax[0,1].set_title('Relationship')
ax[1,0].bar(cats, vals);     ax[1,0].set_title('Comparison')
ax[1,1].hist(data, bins=20); ax[1,1].set_title('Distribution')
fig.suptitle('Analysis Dashboard')
plt.tight_layout()
```

**Q6. Titanic ka complete EDA (Seaborn).**
```python
titanic = sns.load_dataset('titanic')

fig, ax = plt.subplots(2, 3, figsize=(18, 10))
sns.countplot(data=titanic, x='survived', ax=ax[0,0])
sns.countplot(data=titanic, x='pclass', hue='survived', ax=ax[0,1])
sns.countplot(data=titanic, x='sex', hue='survived', ax=ax[0,2])
sns.histplot(data=titanic, x='age', kde=True, ax=ax[1,0])
sns.boxplot(data=titanic, x='pclass', y='fare', ax=ax[1,1])
sns.violinplot(data=titanic, x='survived', y='age', hue='sex', split=True, ax=ax[1,2])
plt.tight_layout()
```

**Q7. Correlation heatmap.**
```python
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', center=0)
```

**Q8. Iris dataset ka pairplot species ke hisaab se.**
```python
sns.pairplot(sns.load_dataset('iris'), hue='species', diag_kind='kde')
```

**Q9. Har din ka bill distribution — violin plot with split.**
```python
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True)
```

**Q10. Facet grid — har day ka scatter alag plot mein.**
```python
sns.relplot(data=tips, x='total_bill', y='tip', hue='sex',
            col='day', col_wrap=2, height=4)
```

---

# PART H — MINI CASE STUDY: Indian Startup Funding

Ye CampusX ka classic end-of-Pandas project hai. Poore playlist ka revision ek jagah.

```python
df = pd.read_csv('startup_funding.csv')

# 1. Data cleaning
df.info()
df.isnull().sum()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Amount'] = df['Amount'].str.replace(',', '').replace('undisclosed', np.nan)
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['City'] = df['City'].str.strip().str.title()

# 2. Basic analysis
df['Startup'].nunique()
df['Amount'].sum()
df['Amount'].mean()

# 3. Top funded startups
df.groupby('Startup')['Amount'].sum().sort_values(ascending=False).head(10)

# 4. City wise funding
df.groupby('City')['Amount'].sum().sort_values(ascending=False).head(10).plot(kind='bar')

# 5. Sector wise funding
df.groupby('Vertical')['Amount'].sum().sort_values(ascending=False).head(10)

# 6. Month by month trend
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
temp = df.groupby(['year','month'])['Amount'].sum().reset_index()
temp['x_axis'] = temp['month'].astype(str) + '-' + temp['year'].astype(str)
plt.figure(figsize=(15,6))
plt.plot(temp['x_axis'], temp['Amount'])
plt.xticks(rotation=90)

# 7. Investor wise analysis
df.groupby('Investors')['Amount'].sum().sort_values(ascending=False).head(10)

# 8. Funding round types
df['Round'].value_counts().head(10).plot(kind='barh')

# 9. Year on year growth
df.groupby('year')['Amount'].sum().plot(kind='bar')

# 10. Heatmap: year vs month
pt = df.pivot_table(index='month', columns='year', values='Amount', aggfunc='sum')
sns.heatmap(pt, cmap='YlOrRd')
```

---

## 💡 Apni Practice ke liye Extra Resources

| Resource | Link |
|---|---|
| 100 NumPy exercises | `github.com/rougier/numpy-100` |
| 100 Pandas puzzles | `github.com/ajcr/100-pandas-puzzles` |
| CampusX official repos | `github.com/campusx-official` |
| Pandas docs (User Guide) | `pandas.pydata.org/docs/user_guide` |
| Seaborn gallery | `seaborn.pydata.org/examples` |
| Matplotlib gallery | `matplotlib.org/stable/gallery` |
| Kaggle Learn (Pandas, Data Viz) | `kaggle.com/learn` |
