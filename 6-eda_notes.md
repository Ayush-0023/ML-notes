# EDA — Univariate, Bivariate & Multivariate Analysis — Complete Notes

---

## What is EDA?

Exploratory Data Analysis (EDA) is the process of visually and statistically exploring data to understand its structure, patterns, and relationships before building any model.

**The goal of EDA is not to make pretty plots. The goal is to reach conclusions.**

After every plot ask:
- Does this feature help separate my target classes?
- Are there unexpected patterns suggesting data quality issues?
- Are two features so correlated they're saying the same thing?
- Does this feature need transformation?

### Three Types of EDA

| Type | Definition | Example |
|---|---|---|
| **Univariate** | Analysis of a single column at a time | Distribution of Age |
| **Bivariate** | Analysis of two columns at a time | Age vs Salary |
| **Multivariate** | Analysis of more than two columns at a time | Age, Salary, Gender together |

---

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load datasets
titanic = pd.read_csv('train.csv')
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')
```

---

# PART 1 — UNIVARIATE ANALYSIS

Understand each column individually before looking at relationships.

---

## Categorical Columns

### 1. Countplot

Shows the frequency of each category.

```python
# Basic countplot
sns.countplot(
    x='Survived',
    hue='Survived',
    data=titanic
)

# Order bars by frequency — more readable
sns.countplot(
    x='Embarked',
    data=titanic,
    order=titanic['Embarked'].value_counts().index
)
```

**What to look for:**
- **Class imbalance** — if target column has 80% zeros and 20% ones, your model will predict zero for everything and still look accurate
- **Rare categories** — categories with very few samples; model can't learn from 3 examples. Group into "Others"
- **Unexpected categories** — typos, inconsistent capitalisation (Male vs male)

### 2. Pie Chart

Shows proportion of each category as part of a whole.

```python
counts = titanic['Survived'].value_counts()
plt.pie(counts, labels=['Not Survived', 'Survived'], autopct='%.2f%%')
plt.show()
```

**When to use vs avoid:**
- Use pie charts only when showing **part of a whole** to a non-technical audience
- For analysis, **countplot is always better** — human eyes compare lengths (bars) far better than angles (pie slices)

---

## Numerical Columns

### 3. Histogram

Shows the frequency distribution of a numerical column.

```python
# Count-based histogram — Y axis is number of observations
sns.histplot(x='Age', data=titanic, kde=True, bins=20)

# Density histogram — Y axis is probability density
sns.histplot(x='Age', data=titanic, kde=True, bins=20, stat='density')
```

**Bin size matters:**
```python
sns.histplot(x='Age', data=titanic, bins=5)   # too few — hides pattern
sns.histplot(x='Age', data=titanic, bins=50)  # too many — too noisy
sns.histplot(x='Age', data=titanic, bins=20)  # usually a good starting point
```

**What to look for:**
- **Shape** — normal, right skewed, left skewed, bimodal
- **Range** — min and max values reasonable?
- **Outliers** — isolated bars far from main distribution
- **Gaps** — missing ranges of values

### 4. KDE Plot (Kernel Density Estimate)

Smoother version of histogram. No bin size decision needed.

```python
sns.kdeplot(x='Age', data=titanic, fill=True, bw_adjust=0.5)
# bw_adjust — controls smoothness. Lower = more detail, higher = smoother
```

**When to prefer KDE over histogram:**
- Comparing two distributions on the same plot — KDE overlays cleanly, histograms overlap messily
- When you want a smooth continuous representation of the distribution

### 5. Boxplot

Shows the 5-number summary and outliers simultaneously.

```python
sns.boxplot(x='Fare', data=titanic)
```

**The 5 numbers:**
- **Minimum** (excluding outliers) — left whisker end
- **Q1 (25th percentile)** — left edge of box
- **Median (50th percentile)** — line inside box
- **Q3 (75th percentile)** — right edge of box
- **Maximum** (excluding outliers) — right whisker end

**Outlier rule:** Any point below Q1 - 1.5×IQR or above Q3 + 1.5×IQR appears as individual dots.

**What to look for:**
- **Box position** — is median closer to Q1 or Q3? Tells you skewness direction
- **Whisker length** — long whisker = spread out data
- **Individual dots** — outliers
- **IQR (box width)** — measure of spread

### 6. Violin Plot

Combines boxplot + full distribution shape. Best of both worlds.

```python
sns.violinplot(x='Age', data=titanic)
# Use when distribution is bimodal or unusual — boxplot would miss this
```

### 7. Skewness and Kurtosis

```python
print(f"Skewness: {titanic['Age'].skew():.2f}")
# 0 = normal
# Positive = right skewed (tail on right)
# Negative = left skewed (tail on left)
# |skew| > 0.5 = skewed, |skew| > 1 = highly skewed

print(f"Kurtosis: {titanic['Age'].kurtosis():.2f}")
# High = heavy tails, more outliers than normal
# Low = light tails, fewer outliers
```

**Why skewness matters for ML:**
- Right skewed → apply log transform
- Left skewed → apply square transform
- Many algorithms assume normally distributed features

### 8. Q-Q Plot

Most reliable way to check normality.

```python
stats.probplot(titanic['Age'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()
# Points on 45° diagonal = normal distribution
# Deviations = where and how distribution differs from normal
```

---

## Complete Univariate Loop

```python
# Categorical columns
for col in titanic.select_dtypes(include='object').columns:
    print(f"\n{col} — {titanic[col].nunique()} unique values")
    print(titanic[col].value_counts())
    sns.countplot(x=col, data=titanic, order=titanic[col].value_counts().index)
    plt.title(col)
    plt.show()

# Numerical columns
for col in titanic.select_dtypes(include='number').columns:
    print(f"\n{col}:")
    print(f"  Skewness: {titanic[col].skew():.2f}")
    print(f"  Kurtosis: {titanic[col].kurtosis():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(x=col, data=titanic, kde=True, ax=axes[0])
    axes[0].set_title(f'{col} — Distribution')
    sns.boxplot(x=col, data=titanic, ax=axes[1])
    axes[1].set_title(f'{col} — Boxplot')
    plt.show()
```

---

# PART 2 — BIVARIATE ANALYSIS

Understand relationships between two columns at a time.

---

## Numerical vs Numerical

### 1. Scatterplot

```python
# Basic bivariate
sns.scatterplot(x='total_bill', y='tip', data=tips)

# Add correlation coefficient
from scipy import stats
r, p = stats.pearsonr(tips['total_bill'], tips['tip'])
print(f"Correlation: {r:.2f}, p-value: {p:.4f}")
# p-value < 0.05 = correlation is statistically significant
```

**What to look for:**
- **Direction** — positive or negative relationship
- **Strength** — tight cluster = strong, spread out = weak
- **Form** — linear or curved
- **Outliers** — points far from main cluster

### 2. Line Plot

Special case of scatterplot — connect the dots. Use when X-axis is time-based.

```python
# Aggregate first
new = flights.groupby('year')['passengers'].sum().reset_index()

sns.lineplot(x='year', y='passengers', data=new)
# Automatically shows mean + confidence interval if multiple y values per x
```

**What to look for:**
- **Trend** — increasing, decreasing, stable
- **Seasonality** — repeating patterns
- **Anomalies** — sudden spikes or drops

---

## Numerical vs Categorical

### 3. Barplot

Shows mean of numerical column for each category. Black line = confidence interval.

```python
sns.barplot(x='Pclass', y='Age', hue='Pclass', data=titanic)
# Black line = confidence interval — how certain we are about that mean
# Wide CI = few samples in that category (unreliable estimate)
# Narrow CI = many samples (reliable estimate)
```

**Important additions:**
```python
# Remove CI if just want bar heights
sns.barplot(x='Pclass', y='Age', data=titanic, ci=None)

# Change estimator — use median for skewed/outlier data
import numpy as np
sns.barplot(x='Pclass', y='Age', data=titanic, estimator=np.median)
```

### 4. Boxplot (Numerical vs Categorical)

Better than barplot when you care about the full distribution, not just the average.

```python
sns.boxplot(x='Sex', y='Age', data=titanic)
```

### 5. Violin Plot (Numerical vs Categorical)

```python
sns.violinplot(x='Sex', y='Age', data=titanic)
# Better than boxplot when distribution is bimodal or unusual
```

### 6. KDE / Distribution Plot Comparison

Most powerful tool for feature selection — shows overlap between classes.

```python
# Method 1 — separate filters
sns.histplot(x='Age', data=titanic[titanic['Survived']==0],
             kde=True, stat='density', label='Not Survived')
sns.histplot(x='Age', data=titanic[titanic['Survived']==1],
             kde=True, stat='density', label='Survived')
plt.legend()

# Method 2 — hue parameter (cleaner)
sns.histplot(x='Age', hue='Survived', data=titanic,
             kde=True, stat='density')

# Method 3 — KDE only
sns.kdeplot(x='Age', hue='Survived', data=titanic)
```

**The key insight:**
- **Distributions completely overlap** → feature cannot help model separate classes → probably not useful
- **Distributions clearly separated** → feature is powerful for classification
- **Which direction** → does higher Age mean more or less survival?

---

## Categorical vs Categorical

### 7. Crosstab Heatmap

```python
# Raw counts
pd.crosstab(titanic['Pclass'], titanic['Survived'])

# Row percentages — what % of each class survived (much more insightful)
pd.crosstab(titanic['Pclass'], titanic['Survived'], normalize='index') * 100

# Heatmap of survival rate
sns.heatmap(
    pd.crosstab(titanic['Pclass'], titanic['Survived'], normalize='index'),
    annot=True,
    fmt='.0%',
    cmap='RdYlGn'  # red = low survival rate, green = high
)
```

**Always normalise.** Raw counts are misleading when category sizes differ. Row percentages tell the real story.

### 8. GroupBy + Mean

```python
# Mean of binary target = rate/probability
titanic.groupby('Embarked')['Survived'].mean() * 100
# This gives survival rate by embarkation port in one line
# Mean of binary column = proportion = rate
```

### 9. Clustermap

Like heatmap but automatically reorders rows AND columns using hierarchical clustering to group similar ones together.

```python
sns.clustermap(
    pd.crosstab(titanic['Parch'], titanic['Survived']),
    annot=True,
    fmt='d'
)
# Reveals patterns you'd miss with fixed ordering
# Similar categories get grouped together automatically
```

**Heatmap vs Clustermap:**
- Heatmap — rows/columns in original order
- Clustermap — auto-reorders to reveal hidden groupings

---

# PART 3 — MULTIVARIATE ANALYSIS

Understand relationships between three or more columns simultaneously.

---

### 10. Scatterplot with Hue and Style

```python
sns.scatterplot(
    x='total_bill',
    y='tip',
    hue='sex',      # colour = third variable
    style='smoker', # marker shape = fourth variable
    data=tips
)
# Encodes 4 variables in one plot
```

### 11. Boxplot with Hue

```python
sns.boxplot(
    x='Sex',
    y='Age',
    hue='Survived',  # third variable
    data=titanic
)
# Distribution of Age by Sex, split by Survived
```

### 12. Pairplot

Bird's eye view of all pairwise relationships in one grid. Best for datasets with multiple numerical columns.

```python
sns.pairplot(iris, hue='species')
# Diagonal = individual distribution of each feature (univariate)
# Off-diagonal = pairwise relationships (bivariate)
# Colour by target = see which features separate classes
```

**What to look for:**
- Which feature pairs show clear separation between classes? → Most useful for classification
- Which scatter plots show linear relationship? → Those features are correlated → multicollinearity risk
- Which plots show no pattern? → Those features are independent

**Performance note:** Pairplot is slow on large datasets. Sample first:
```python
sns.pairplot(df.sample(500, random_state=42), hue='target')
```

### 13. Correlation Heatmap (Numerical Multivariate)

```python
plt.figure(figsize=(10, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    mask=np.triu(np.ones_like(df.corr(numeric_only=True)))  # lower triangle only
    # Upper triangle is mirror of lower — no need to show both
)
plt.title('Correlation Heatmap')
plt.show()
```

### 14. Pivot Heatmap — Time Series Patterns

Reveals seasonality — which months/years consistently behave differently.

```python
sns.heatmap(
    flights.pivot_table(values='passengers', index='month', columns='year'),
    cmap='YlOrRd',
    annot=True,
    fmt='d'
)
# Rows = months, columns = years
# Immediately shows which months are always busiest
```

### 15. Pivot Clustermap

```python
sns.clustermap(
    flights.pivot_table(values='passengers', index='month', columns='year'),
    cmap='YlOrRd'
)
# Groups similar months and similar years together automatically
```

---

## Pandas Profiling — Automated EDA

A tool that automates the entire EDA process in one line.

```python
from ydata_profiling import ProfileReport  # newer name for pandas_profiling

profile = ProfileReport(df, title="EDA Report", explorative=True)
profile.to_file("report.html")
```

**What it gives you automatically:**
- Overview statistics
- Per column analysis (distribution, missing values, unique values)
- Missing value patterns
- Correlations
- Duplicate detection
- Interactions between columns

**When to use:**
- Quick first look at an unfamiliar dataset
- Generating a shareable report

**When NOT to use:**
- As a replacement for thinking
- For targeted analysis — manual EDA gives more control and insight

---

## Complete EDA Toolkit Summary

```
Two numerical columns     → Scatterplot + correlation coefficient
Numerical + categorical   → Barplot (means) + Boxplot (distribution) + KDE (overlap)
Two categorical columns   → Crosstab heatmap (normalised) + groupby mean
Many numerical columns    → Pairplot + correlation heatmap
Time series               → Lineplot + pivot heatmap for seasonality
Three+ variables          → Scatterplot with hue/style + boxplot with hue
```

---

## What to Conclude From EDA — The Most Important Part

| What you see | Conclusion | Action |
|---|---|---|
| Class imbalance in target | Model will be biased toward majority class | SMOTE, class weights, use F1/precision/recall |
| KDE distributions completely overlap | Feature can't separate classes | Consider dropping feature |
| KDE distributions clearly separated | Feature is highly predictive | Keep and prioritise |
| Barplot shows large difference across categories | Category strongly affects target | Important feature |
| High correlation between two input features | Multicollinearity risk | Drop one or use Ridge |
| Bimodal distribution | Two subgroups in data | Investigate — might need separate models |
| Outliers in boxplot | Will distort distance/regression models | Apply capping or trimming |
| Right skewed histogram | Log transform needed for linear models | Apply FunctionTransformer(np.log1p) |
| Confidence interval very wide in barplot | Few samples for that category | Treat rare category carefully |

EDA without conclusions is just decoration.
