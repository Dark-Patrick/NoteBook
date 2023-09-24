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

数据错误类型：异常值，规则冲突，格式错误

```python
data["Type"].value_counts()[0:20]
```

基于规则的检测(Rule-based Detection)

基于模式的检测(Pattern-based Detection)

### 数据变换

|        Normalization for Real Value Columns        |                                               |
| :------------------------------------------------: | --------------------------------------------- |
|               Min-max normalization                | $x'_i = \frac{x_i-min_x}{max_x-min_x}(b-a)+a$ |
| Z-score normalization:0 mean, 1 standard deviation | $x'_i=\frac{x_i-mean(x)}{std(x)}$             |
|                  Decimal scaling                   | $x'_i=x_i/10^j$ smallest j s.t. max(\|x\|)<1  |
|                    Log scaling                     | $x'_i=log(x_i)$                               |



Image Transformations

下采样和裁剪，图片白化

Text Transformations

词根化、语法化

### 特征工程

**Tabular Data Features**

- int/float:直接使用或者分成n个bin
- Catagorical data: ont-hot encoding
- Date-time: 一个特征列表
  - eg:[year, month, day, day_of_year, week_of_year, day_of_week]
- Feature combination: 将不同的特征组合起来
  - [cat, dog] * [male, female] -> [(cat, male), (cat, female), (dog, male), (dog, female)]

**Text Features**

- Represent text as token features
  - Bag of words(BoW) model
  - Word Embeddings(eg:Word2vec)

- Pre-trained language models(eg: BERT, GPT-3)

**Image/Video Features**

- hand-craft
- pre-trained deep neural networks



## 机器学习模型

决策树模型

- 随机森林
- Gradient Boosting Decision Trees

线性模型



## 模型评估 

- Accuracy

```python
sum(y == y_hat) / y.size
```

- Precision（精度）

```python
sum((y_hat == 1) & (y == 1)) / sum(y_hat == 1)
```

- Recall（召回）

```python
sum((y_hat == 1) & (y == 1)) / sum(y == 1)
```

- F1:Balnce precision and recall

- AUC&ROC(常用于二分类问题)

  ![](./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-20%20213259.png)

### 过拟合、欠拟合

数据和模型的复杂度要匹配



## 集成学习

###  方差&偏差

$$
E[f] = f\ \   E[ε] = 0\ \ Var[ε] = σ^2\\

ε\ is\ independent\ of\  \hat{f}\\


E_D[(y - \hat{y}(x))^2] =E[((f-E[\hat{f}])+ε-(\hat{f}-E[\hat{f}]))^2]\\
					    =(f-E[\hat{f}])^2+E[ε^2]+E[(\hat{f}-E[\hat{f}])^2]\\
					    =Bias[\hat{f}]^2+Var[\hat{f}]+σ^2
$$

- 减少偏差
  - 使用更复杂的模型
  - Boosting
  - Stacking
- 减少方差
  - 使用更简单的模型
  - Regularization添加正则项
  - Bagging
  - Stacking
- 减少$σ^2$
  - 提升数据质量

==Ensemble learning：集成学习，通过用多个模型提升预测的性能==



### bagging

==Bootstrap AGGregatING==: train multiple learner on data by bootstrap sampling

```python
'''Bagging Code'''
class Bagging: 
 def __init__(self, base_learner, n_learners):
 	self.learners = [clone(base_learner) for _ in range(n_learners)]# 复制多个learner，并保存起来
    
 def fit(self, X, y):
 	for learner in self.learners: 
 		examples = np.random.choice( # 有放回的抽取
 			np.arange(len(X)), int(len(X)), replace=True)
 	learner.fit(X.iloc[examples, :], y.iloc[examples])
    
 def predict(self, X):
 	preds = [learner.predict(X) for learner in self.learners] 
 	return np.array(preds).mean(axis=0) # 对每一个learner取平均

```

**Bagging reduces more variance when base learners are unstable**



### Boosting

Combines weak learners into a strong one, primarily to reduce bias

```python
'''Gradient Boosting Code'''
class GradientBoosting: 
 def __init__(self, base_learner, n_learners, learning_rate):
 	self.learners = [clone(base_learner) for _ in range(n_learners)]
 	self.lr = learning_rate
    
 def fit(self, X, y):
 	residual = y.copy()# 残差初始为y本身
 	for learner in self.learners:
 		learner.fit(X, residual)
 		residual -= self.lr * learner.predict(X) 
    
 def predict(self,X):
 	preds = [learner.predict(X) for learner in self.learners]
 	return np.array(preds).sum(axis=0) * self.lr
```

==XGBoost== ==lightBoost==



### Stacking

Combine multiple base learners to reduce variance

![](./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-23%20104602.png)

Multi-layer Stacking

![](./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-23%20105021.png)

Overfitting in Multi-layer Stacking

- 不同的层用不同的训练数据以避免过拟合
- 重复的k折交叉验证



## 模型调参

tensorboard、weights&bias

自动模型调参：autoML

- Hyperparameter optimization(HPO)
- Neural architecture search(NAS)

### 超参数优化(HPO)

<img src="./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-23%20220833.png" style="zoom:80%;" />

**HPO常用算法**

- Black-box
- Multi-fidelity: modifies the training job to speed up the search
  - Train on subsampled datasets
  - Reduce model size(e.g. less layers, channels)
  - Stop bad configuration earlier

<img src="./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-23%20221507.png" style="zoom:80%;" />

**简单介绍**

- Grid search

```python
for config in search_space:
    train_and_eval(config)
return best_result
```

- Random search

```python
for _ in range(n):
    config = random_select(search_space)
    train_and_eval(config)
return best_result
```

- Bayesian Optimization(BO)
  - 学习超参数到精度的评估指标中间的一个函数
  - Surrogate model(调参模型)
  - Acquisition function(采样模型)
  - Limitation of BO

![](./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-23%20222712.png)

- Successive Halving
  1. 随机抽取n个超参数训练m轮
  2. 保留最好的n / 2个超参数训练m轮
  3. 保留最好的n / 4个超参数训练2m轮
  4. ...
- Hyperband: 跑多次SH，选取不同的n、m

### 网络架构搜索(NAS)

搜索空间->怎么在搜索空间搜索->衡量神经网络架构的好坏

NAS训练方法

- 强化学习

- One-shot
- Scaling CNNs: Compound depth, width, resolution scaling



## 深度学习网络架构

### 批量和层的归一化

**Batch Norm**

对于具有多层的深度学习网络，把中间的一些层也做标准化，使得函数更加平滑

<img src="./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-24%20101137.png" style="zoom:80%;" />

```python
'''Batch Normalization Code'''
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
	if not torch.is_grad_enabled(): # In prediction mode 
		X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps) # 使用训练时存下的全局均值和方差 
	else: 
		assert len(X.shape) in (2, 4) 
		if len(X.shape) == 2: 
			mean = X.mean(dim=0) 
			var = ((X - mean)**2).mean(dim=0) 
		else: 
			mean = X.mean(dim=(0, 2, 3), keepdim=True) 
			var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True) 
		X_hat = (X - mean) / torch.sqrt(var + eps) 
		moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
		moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta 
return Y, moving_mean, moving_var

```

[Full code of Batch Normalization](http://d2l.ai/chapter_convolutional-modern/batch-norm.html)



**Layer Norm**

主要用于循环神经网络

<img src="./assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-24%20105308.png" style="zoom:80%;" />

Batch Norm是按列(特征)求平均，即将通道中的数据求均值与方差(CNN)

Layer Norm是按行(样本)求平均，将样本的所有特征和时间步骤变成一个向量，求均值与方差(RNN)



**More Norm**

- Modify reshape:

  - InstanceNorm:n * c * w * h -> wh * cn

  - GroupNorm: n * c * w * h -> swh * gn with c = sg

  - CrossNorm: swap mean/std between a pair of features

- Modify normalize: whitening

- Modify recovery: replace γ，β with a dense layer

- Apply to weights or gradients

==梯度裁剪==



### 迁移学习

看Dive into AI

哪里寻找预训练模型

- [Tensorflow Hub](https://tfhub.dev)

- [TIMM](https://github.com/rwightman/pytorch-image-models)

```python
import timm
from torch import nn

model = timm.create_model("resnet18", pretrained=True)
model.fc = nn.Linear(model.fc.in_features, n_classes)
# Train model as a normal training job
```



**微调用于nlp**

自监督的预训练

预训练模型

- Word embeddings
- Transformer based pre-trained models
  - BERT: a transformer **encoder**
  - GPT: a transformer **decoder**
  - T5: a transformer **encoder-decoder**

哪里去找预训练模型

- HuggingFace

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
inputs = tokenizer(sentences, padding="max_length", truncation=True) 
model = AutoModelForSequenceClassification.from_pretrained(
	“bert-base-cased", num_labels=2) 
# Train model on inputs as a normal training job
```

