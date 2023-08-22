# matploblib简易入门

***

## 折线图绘制

``````python
from matplotlib import pyplot as plt

plt.figure(figsize = (8, 20), dpi = 80) # 设置大小与清晰度

plt.plot(x, y) # 传入x和y，通过plot绘制出折线图

plt.xticks(range(2, 25), str, rotation) # 设置x轴刻度, str可选项，与索引一一对应

plt.xlable("")
plt.ylable("")
plt.title("")

plt.show() # 展示

plt.savefig("./name.png") # 保存, 可保存为svg矢量图
``````

#### 常用函数及参数

1. plt.xticks()

   * list 数值型数据

   * str 字符型数据， 与数值型一一对应
   * rotation 设置x标签旋转

2. plt.grid()

   * alpha = float 设置透明度
   * linestyle = ':' ; '-.' ; '--'

3. plt.plot()

   多次plot既可绘制多条折线图

   * label = str 标签
   * color = str 设置颜色
   * linestyle = ':' ; '-.' ; '--'
   * linewidth = int
   * alpha = float

4. plt.legend() 添加图例

   * loc = num or str 设置图例位置



## 散点图绘制

```python
from matplotlib import pyplot as plt

y_1 = []
y_2 = []

x_1 = range(1, 32)
x_2= range(51, 82)

plt.scatter(x_1, y_1, lable="") # 散点图
plt.scatter(x_2, y_2, lable="")

_x = list(x_1) + list(x_10)
_xtick_labels = ["3月{}日".fromat(i) i for i in x_1]
_xtick_labels += ["10月{}日".fromat(i - 50) i for i in x_2]
plt.xticks(_x[::2], _xtick_labels[::2], rotation=45)

plt.xlable("")
plt.ylable("")
plt.title("")

plt.legend(loc="upper left")
plt.show()
```



## 条形图绘制

```python
from matplotlib import pyplot as plt

a = ["猩球崛起3：终极之战","敦刻尔克","蜘蛛侠：英雄归来","战狼2"]
b_16 = [15746,312,4497,319]
b_15 = [12357,156,2045,168]
b_14 = [2358,399,2358,362]

bar_width = 0.2

x_14 = list(range(len(a)))
x_15 =  [i+bar_width for i in x_14]
x_16 = [i+bar_width*2 for i in x_14]

#设置图形大小
plt.figure(figsize=(20,8),dpi=80)

plt.bar(range(len(a)),b_14,width=bar_width,label="9月14日")
plt.bar(x_15,b_15,width=bar_width,label="9月15日")
plt.bar(x_16,b_16,width=bar_width,label="9月16日")

#设置图例
plt.legend(prop=my_font)

#设置x轴的刻度
plt.xticks(x_15,a,fontproperties=my_font)

plt.show()
```



## 直方图绘制

```python
from matplotlib import pyplot as plt

a=[131,  98, 125, 131, 124, 139, 131, 117, ...]

#计算组数
d = 3  #组距
num_bins = (max(a)-min(a))//d
print(max(a),min(a),max(a)-min(a))
print(num_bins)

#设置图形的大小
plt.figure(figsize=(20,8),dpi=80)
plt.hist(a,num_bins,normed=True) # normed:计算比例

#设置x轴的刻度
plt.xticks(range(min(a),max(a)+d,d))

plt.grid()

plt.show()
```

一般来说能够用plt.hist方法的是那些==没有统计过==的数据，经过统计之后的可用条形图绘制:

```python
from matplotlib import pyplot as plt

interval = [0,5,10,15,20,25,30,35,40,45,60,90]
width = [5,5,5,5,5,5,5,5,5,15,30,60]
quantity = [836,2737,3723,3926,3596,1438,3273,642,824,613,215,47]


print(len(interval),len(width),len(quantity))
# 12, 12, 12

#设置图形大小
plt.figure(figsize=(20,8),dpi=80)


plt.bar(interval,quantity,width=width)

#设置x轴的刻度
temp_d = [5]+ width[:-1]
_x = [i-temp_d[interval.index(i)]*0.5 for i in interval]

plt.xticks(_x,interval)

plt.grid(alpha=0.4)
plt.show()
```

