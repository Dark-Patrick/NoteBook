# pandas基础入门

***

## Series--一维，带标签的数组

```python
import pandas as pd

t = pd.Series([1, 2, 3, 12, 14, 31], index=list("abcdef"))

temp_dict = {"name":"LCH", "age":23, "tel":1341234}
t_2 = pd.Series(temp_dict)

# 字典推导式创建一个字典a
a = {string.ascii_uppercase[i]:i for i in range(10)}
pd.Series(a)
# 重新给其指定其他的索引后，如果能对应上，就取其值，如不能就为NaN
pd.Series(a, index=list(string.ascii_uppercase[5:15]))
# dtype: float64
# numpy中nan为float，pandas会自动根据数据类更改series的dtype类型

'''
Series切片和索引
'''
t.index # 获取标签 type : pandas.core.indexes.base.Index
t.values # 获取值 type: numpy.ndarray

temp_dict[1:2]
temp_dict["name"]
t[[2, 3, 6]]
t[t > 4]
```



## 读取外部数据

```python
df = pd.read_csv("name.csv")
df = pd.read_sql(sql_sentence, connection)
```



## DataFrame--二维

```python
pd.DataFrame(np.range(12).reshape(3, 4))

pd.DataFrame(np.range(12).reshape(3, 4), index=list("abc"), columns=list("WXYZ"))

temp_dict = {"name":["LCH", "Lee"], "age":[23, 24], "tel":{1341234, 3244}}
temp_dict = pd.DataFrame(temp_dict)
```

**DataFrame整体情况查询：**

- df.head(3): 显示头部几行，默认5行
- df.tail(3): 显示末尾几行，默认5行
- df.info(): 相关信息概览
- df.describe(): 快速综合统计结果：计数，均值，标准差，最大值，四分位数，最小值

**DataFrame常用函数**

- df.sort_values(by="col name", ascending=False)
  - by: 按指定列排序
  - ascending=Flase: 降序
- df.loc: 通过==标签==索引数据
- df.iloc[1:2, 0:2]: 通过==位置==获取数据
- pd.isnull(df)
- pd.notnull(df)
- df.dropna(axis=0, how="all / any", inplace=True)
  - how=all：全部为nan时在删除当前行(列)，any：有一个就删除
  - inplace=True: 原地修改

- df.fillna(df.mean())
  - df.mean(): 会忽略nan再计算均值


