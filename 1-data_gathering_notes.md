# Data Gathering Methods — Complete Notes

---

## Why Data Gathering Matters

Before any ML happens, you need data. In real projects, data never comes in one clean CSV file. It comes from databases, APIs, web pages, Excel files — often all at once. Every method has its quirks, and knowing them saves hours of debugging.

The universal goal: get data into a **pandas DataFrame** as cleanly as possible, with correct types and identified missing values.

---

## 1. CSV Files

### What is a CSV?

CSV = Comma Separated Values. The simplest way to store tabular data — like a spreadsheet saved as plain text.

```
name,age,salary
John,25,50000
Mary,30,75000
```

Each line = one row. Each comma = column separator. First row = header (column names).

### Why CSV?

- Works everywhere — Excel, Python, R, any database
- Human readable — open in any text editor
- Lightweight — no special software needed

### Reading CSV

```python
import pandas as pd

# Basic read
df = pd.read_csv('data.csv')

# Full options
df = pd.read_csv(
    'data.csv',

    # SEPARATOR
    sep=',',            # default — comma
    # sep='\t'          # TSV (tab separated values)
    # sep=';'           # some European files use semicolon

    # HEADER
    header=0,           # first row is column names (default)
    # header=None       # no column names in file
    # skiprows=3        # skip first 3 rows of junk

    # COLUMN NAMES
    # names=['col1', 'col2', 'col3']  # provide your own names

    # MISSING VALUES
    na_values=['NA', 'null', 'NULL', '-', '', 'missing', 'N/A'],

    # DATA TYPES
    dtype={'age': int, 'salary': float, 'name': str},

    # DATES
    parse_dates=['date_column'],  # automatically parse dates

    # INDEX
    index_col='id',     # use this column as row index

    # ENCODING
    encoding='utf-8',   # default
    # encoding='latin-1' # use when you get UnicodeDecodeError

    # LOAD SUBSET
    nrows=1000,         # only read first 1000 rows
    usecols=['name', 'age', 'salary']  # only load specific columns
)
```

### The Encoding Problem

When you get this error:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

```python
# Try latin-1 first
df = pd.read_csv('data.csv', encoding='latin-1')

# If unsure, detect encoding automatically
import chardet

with open('data.csv', 'rb') as f:
    result = chardet.detect(f.read())
print(result['encoding'])
```

### Large Files — Chunking

When file is too large to load into RAM:

```python
chunks = []

for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    # Process each chunk of 10000 rows
    chunk = chunk[chunk['salary'] > 50000]  # filter
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
```

### TSV Files

Same as CSV but tab-separated. Common in NLP and bioinformatics:

```python
df = pd.read_csv('data.tsv', sep='\t')
```

### Writing CSV

```python
df.to_csv('output.csv', index=False)
# index=False — don't write row numbers as a column
```

---

## 2. JSON Files

### What is JSON?

JSON = JavaScript Object Notation. Stores data as key-value pairs like a Python dictionary. Standard format for APIs.

```json
[
    {"name": "John", "age": 25, "salary": 50000},
    {"name": "Mary", "age": 30, "salary": 75000}
]
```

### Why JSON?

- Standard format for APIs — every web API returns JSON
- Handles nested/hierarchical data — CSV can't
- Human readable

### Reading Flat JSON

```python
# List of records — most common
df = pd.read_json('data.json')

# Different orientations
df = pd.read_json('data.json', orient='records')   # list of dicts
df = pd.read_json('data.json', orient='columns')   # dict of lists
df = pd.read_json('data.json', orient='index')     # dict of dicts
```

### Reading Nested JSON

Real world JSON is rarely flat:

```json
[
    {
        "customer_id": 1,
        "name": "John",
        "address": {
            "city": "Mumbai",
            "state": "Maharashtra"
        },
        "orders": [
            {"order_id": 101, "amount": 500},
            {"order_id": 102, "amount": 300}
        ]
    }
]
```

```python
import json

with open('nested.json') as f:
    data = json.load(f)

# Flatten one level — address becomes address.city, address.state
df = pd.json_normalize(data)

# Flatten nested list — one row per order
df = pd.json_normalize(
    data,
    record_path=['orders'],        # expand this nested list
    meta=['customer_id', 'name']   # keep these parent fields
)
```

### Writing JSON

```python
df.to_json('output.json', orient='records', indent=2)
```

---

## 3. SQL Databases

### What is SQL?

SQL = Structured Query Language. Production data almost always lives in databases, not CSV files. SQL lets you query, filter, join, and aggregate tables.

### Two Ways to Connect

**sqlite3** — for local SQLite databases (single file, no server):

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()
```

**SQLAlchemy** — for production databases (MySQL, PostgreSQL, etc.):

```python
from sqlalchemy import create_engine

# PostgreSQL
engine = create_engine('postgresql://user:password@localhost:5432/mydb')

# MySQL
engine = create_engine('mysql+pymysql://user:password@localhost/mydb')

# SQLite
engine = create_engine('sqlite:///database.db')

df = pd.read_sql_query("SELECT * FROM customers", engine)
```

### Essential SQL for ML

```sql
-- Select specific columns
SELECT name, age, salary FROM customers;

-- Filter rows
SELECT * FROM customers WHERE age > 25 AND salary > 50000;

-- Join two tables
SELECT customers.name, orders.amount
FROM customers
JOIN orders ON customers.id = orders.customer_id;

-- Aggregate
SELECT city, AVG(salary) as avg_salary, COUNT(*) as count
FROM customers
GROUP BY city
HAVING COUNT(*) > 10
ORDER BY avg_salary DESC;

-- Handle missing values
SELECT * FROM customers WHERE salary IS NOT NULL;
```

### Writing Back to Database

```python
df.to_sql(
    'processed_data',     # table name
    engine,
    if_exists='replace',  # 'replace' / 'append' / 'fail'
    index=False
)
```

---

## 4. Fetching from APIs

### What is an API?

API = Application Programming Interface. You send a request to a URL. The server sends back JSON data.

### Basic API Call

```python
import requests

response = requests.get('https://api.example.com/data')

print(response.status_code)  # 200 = success
print(response.json())       # the actual data
```

### Status Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 201 | Created successfully |
| 400 | Bad request — your fault |
| 401 | Unauthorized — wrong API key |
| 403 | Forbidden — no permission |
| 404 | Not found |
| 429 | Too many requests — rate limited |
| 500 | Server error — their fault |

### Authentication

```python
# API Key in headers (most common)
headers = {'Authorization': 'Bearer your_api_key'}
response = requests.get(url, headers=headers)

# API Key as parameter
response = requests.get(url, params={'api_key': 'your_key'})

# Basic auth
response = requests.get(url, auth=('username', 'password'))
```

### Passing Parameters

```python
params = {
    'start_date': '2024-01-01',
    'limit': 100,
    'category': 'electronics'
}

response = requests.get('https://api.example.com/data', params=params)
# Becomes: https://api.example.com/data?start_date=2024-01-01&limit=100&...
```

### Rate Limiting — The Big Practical Issue

APIs cap how many requests you can make per minute. Exceed the cap → 429 error → temporary block.

```python
import time

all_data = []

for page in range(1, 101):
    response = requests.get(
        'https://api.example.com/data',
        params={'page': page, 'limit': 100},
        headers={'Authorization': 'Bearer your_key'}
    )

    if response.status_code == 429:
        print("Rate limited — waiting 60 seconds")
        time.sleep(60)
        continue  # retry same page

    if response.status_code != 200:
        print(f"Error on page {page}: {response.status_code}")
        break

    data = response.json()

    if not data['results']:  # empty = no more pages
        break

    all_data.extend(data['results'])
    time.sleep(0.5)  # small delay — be respectful

df = pd.DataFrame(all_data)
```

---

## 5. Web Scraping

### What is Web Scraping?

When there's no API, extract data directly from HTML of web pages.

### Two Types of Web Pages

| Type | Description | Tool |
|---|---|---|
| Static | HTML fully loaded when page opens | BeautifulSoup |
| Dynamic | Content loads via JavaScript after page opens | Selenium / Playwright |

### Static Scraping — BeautifulSoup

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Fetch page — pretend to be a browser
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get('https://example.com', headers=headers)

# Parse HTML
soup = BeautifulSoup(response.content, 'html.parser')

# Find elements
title = soup.find('h1').text                              # by tag
price = soup.find('span', class_='price').text            # by class
product = soup.find('div', id='product-info').text        # by id
all_prices = soup.find_all('span', class_='price')        # find all

# Extract table
table = soup.find('table')
rows = []
for tr in table.find_all('tr')[1:]:  # skip header row
    cols = [td.text.strip() for td in tr.find_all('td')]
    rows.append(cols)

df = pd.DataFrame(rows, columns=['name', 'price', 'rating'])
```

### Dynamic Scraping — Selenium

When content loads via JavaScript:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get('https://example.com')

# Wait for element to load
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'product')))

# Extract data
products = driver.find_elements(By.CLASS_NAME, 'product')
data = []

for product in products:
    name = product.find_element(By.CLASS_NAME, 'name').text
    price = product.find_element(By.CLASS_NAME, 'price').text
    data.append({'name': name, 'price': price})

driver.quit()
df = pd.DataFrame(data)
```

### Important Rules

- **Check robots.txt first:** `https://example.com/robots.txt` — tells you what you're allowed to scrape
- **Add delays** between requests — don't hammer servers
- **Don't scrape personal data** — legal and ethical issue

---

## 6. Excel Files

### Why Excel?

Business data almost always comes in Excel. Finance, HR, operations — every team uses it.

```python
# Read single sheet
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Read by index
df = pd.read_excel('data.xlsx', sheet_name=0)  # first sheet

# Read all sheets
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)
# Returns dict: {'Sheet1': df1, 'Sheet2': df2}

# With options
df = pd.read_excel(
    'data.xlsx',
    sheet_name='Sales',
    skiprows=2,
    na_values=['N/A', '-'],
    usecols='A:E'
)

# Write single sheet
df.to_excel('output.xlsx', sheet_name='Results', index=False)

# Write multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)
```

---

## 7. Cloud Storage

In production, data often lives in cloud storage rather than local files.

```python
# AWS S3
import boto3

s3 = boto3.client('s3')
obj = s3.get_object(Bucket='mybucket', Key='data.csv')
df = pd.read_csv(obj['Body'])

# Google Cloud Storage
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('mybucket')
blob = bucket.blob('data.csv')
blob.download_to_filename('local_data.csv')
df = pd.read_csv('local_data.csv')
```

---

## Complete Summary

| Method | Best For | Key Library | Common Gotcha |
|---|---|---|---|
| CSV | Flat tabular data | pandas | Encoding errors, wrong separator |
| TSV | Tab separated data | pandas | Remember sep='\t' |
| JSON | Nested/API data | pandas, json | Nested structure needs json_normalize |
| SQL | Production databases | sqlalchemy | Use SQLAlchemy not raw sqlite3 for production |
| API | Live/real-time data | requests | Rate limiting, authentication |
| Web scraping (static) | HTML tables/lists | BeautifulSoup | Websites block scrapers — send User-Agent |
| Web scraping (dynamic) | JS-rendered pages | Selenium | BeautifulSoup sees empty page for JS sites |
| Excel | Business data | pandas | Multiple sheets, skiprows for messy headers |
| Cloud | Production data | boto3, google-cloud | Credentials management |

**Everything ends up as a DataFrame** — that's the universal format for all ML preprocessing.
