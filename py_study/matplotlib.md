# 折线图绘制

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







