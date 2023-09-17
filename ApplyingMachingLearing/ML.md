# 实用机器学习

***

## 数据获取

### 数据的发现

寻找现存的数据集

- MNIST：手写识别
- ImageNet：图片数据集
- AudioSet：油管声音切片
- Kinetics：油管视频切片，识别人的行为
- KITTI：无人驾驶数据集
- Amazon Review：用户评论数据集
- SQuAD：问题-答案对
- LibriSpeech：有声读物

**去哪里找数据？**

- Paperswithcodes Datasets：论文常见数据集
- Kaggle Datasets：数据科学家上传的机器学习数据集
- Google Dataset search：搜索引擎数据集
- Open Data on AWS：大量原始数据集

|                      | Pros                           | Cons                                                 |
| -------------------- | ------------------------------ | ---------------------------------------------------- |
| Academic datasets    | Clean,proper difficulty        | Limited choices, too simplified, usually small scale |
| Competition datasets | Closer to real ML applications | Still simplified, and only available for hot topics  |
| Raw Data             | Great flexibility              | Need a lot of effort to process                      |

### 数据融合

将不同表的数据融合起来，即table join的过程

### 生成数据

- GANS
- 数据增强：图像的翻转，切割等操作，文本数据的反向翻译

### 网页数据抓取

常用工具：curl、headless browser、大量的新IP去抓取网页

```python
# headless browser实例
from selenium import webdriver

chrome_options = webdriver.ChromeOptions()
chrome_options.headless = True
chrome = webdriver.Chorme(
	chrome_options=chrome_options)

page = chrome.get(url)


'''
假设已经获取并保存了html页面
'''
page = BeautifulSoup(open(html_path, 'r'))
links = [a['href'] for a in page.find_all('a', 'list-card-list')]
ids = [l.split('/')[-2].split('_')[0] for l in links] # 19506780_zpid
'''
The house detial page by ID
https://www.zillow.com/homedetails/19506780_zpid/

提取数据
根据前端的标签信息获取需要的数据
'''
sold_items = [a.text for a in page.find(
			'div', 'ds-home-details-chip').find('p').find_all('span')]
for item in sold_items:
    if 'Sold:' in item:
        result['Sold Price'] = item.split(' ')[1]
    if 'Sold on' in item:
        result['Sold on'] = item.split(' ')[-1]
```

![](./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-17%20111746.png)

### 数据标注

半监督学习、自学习、人工标注、弱监督学习（数据编程）

```python
'''
Data programming example
rules to check if YouTube comments are spam or ham
'''
def check_out(x):
    return SPAM if 'check out' in x.lower() else ABSTAIN
def sentiment(x):
    return HAM if sentiment_polarity(x) > 0.9 else ABSTAIN
def short_comment(x):
    return HAM if len(x.split()) < 5 else ABSTAIN
```



## 数据预处理

### 探索性数据分析

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 将绘制的图片设置为svg模式
from IPython import display
display.set_matplotlib_formats('svg')

data = pd.read_csv('house_sales.zip')

# check the data shape and the first a few examples
data.shape()
data.head()

# We drop columns that at least 30% values are null to simplify our EDA.
null_sum = data.isnull().sum()
data.columns[null_sum < len(data) * 0.3]  # columns will keep
data.drop(columns=data.columns[null_sum > len(data) * 0.3], inplace=True)

# Next we check the data types
data.dtypes

# Convert currency from string format such as $1,000,000 to float.
currency = ['Sold Price', 'Listed Price', 'Tax assessed value', 'Annual tax amount']
for c in currency:
    data[c] = data[c].replace(
        r'[$,-]', '', regex=True).replace(
        r'^\s*$', np.nan, regex=True).astype(float)
# Also convert areas from string format such as 1000 sqft and 1 Acres to float as well.
areas = ['Total interior livable area', 'Lot size']
for c in areas:
    acres = data[c].str.contains('Acres') == True
    col = data[c].replace(r'\b sqft\b|\b Acres\b|\b,\b','', regex=True).astype(float)
    col[acres] *= 43560
    data[c] = col

# Now we can check values of the numerical columns. You could see the min and max values for several columns do not make sense.
data.describe()

# We filter out houses whose living areas are too small or too hard to simplify the visualization later.
abnormal = (data[areas[1]] < 10) | (data[areas[1]] > 1e4)
data = data[~abnormal]
sum(abnormal)

# Let's check the histogram of the 'Sold Price', which is the target we want to predict.
ax = sns.histplot(np.log10(data['Sold Price']))
ax.set_xlim([3, 8])
ax.set_xticks(range(3, 9))
ax.set_xticklabels(['%.0e'%a for a in 10**ax.get_xticks()]);
```

<img src="./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-17%20164642.png" alt="价格分布" style="zoom:80%;" />

```python
# A house has different types. Here are the top types:
data['Type'].value_counts()[0:20]

# Price density for different house types.
types = data['Type'].isin(['SingleFamily', 'Condo', 'MultiFamily', 'Townhouse'])
sns.displot(pd.DataFrame({'Sold Price':np.log10(data[types]['Sold Price']),
                          'Type':data[types]['Type']}),
            x='Sold Price', hue='Type', kind='kde');
```

<img src="./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-17%20164715.png" alt="不同类别房子的价值" style="zoom:80%;" />

```python
# Another important measurement is the sale price per living sqft. Let's check the differences between different house types.
data['Price per living sqft'] = data['Sold Price'] / data['Total interior livable area']
ax = sns.boxplot(x='Type', y='Price per living sqft', data=data[types], fliersize=0)
ax.set_ylim([0, 2000]);
```

<img src="./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-17%20164728.png" alt="单位均价" style="zoom:80%;" />

```python
# We know the location affect the price. Let's check the price for the top 20 zip codes.
d = data[data['Zip'].isin(data['Zip'].value_counts()[:20].keys())]
ax = sns.boxplot(x='Zip', y='Price per living sqft', data=d, fliersize=0)
ax.set_ylim([0, 2000])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
```

<img src="./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-17%20164741.png" alt="不同地区的单位均价" style="zoom:80%;" />

```python
# Last, we visualize the correlation matrix of several columns.
_, ax = plt.subplots(figsize=(6,6))
columns = ['Sold Price', 'Listed Price', 'Annual tax amount', 'Price per living sqft', 'Elementary School Score', 'High School Score']
sns.heatmap(data[columns].corr(),annot=True,cmap='RdYlGn', ax=ax);
```

<img src="./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-17%20164800.png" alt="特征之间的关系，协方差矩阵" style="zoom:80%;" />



### 数据清理

### 数据变换

### 特征工程



