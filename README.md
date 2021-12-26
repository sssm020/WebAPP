

## 1.设计目的以及需求

- 场景：目前新闻网站上大多数新闻并没有对应的摘要，为了更好的让用户快速浏览新闻，利用自然语言处理技术来对没有摘要的新闻文本进行自动摘要生成，来满足用户需求，节约用户时间。
- 问题定义：实现新闻文本的自动摘要功能

## 2.解决方案

根据目前对于文本摘要技术的普遍方法主要分为两种，一种为抽取式文本摘要，另一种为生成式文本摘要。本应用主要采用的是抽取式文本摘要，生成式文本摘要在之后的版本进行完善，并且分为两级内容摘要。抽取式主要进行最为简要的概括，用户可继续深入阅读，此时就采用生成式文本摘要对新闻内容进行全面且具体的摘要。

## 3.使用的模型以及方法

### 1.基于TextRank的新闻文本摘要

#####   1） 核心思想：PageRank算法的自然语言处理版本

- 链接数量：如果一个网页被越多的其他网页链接，则这个网页越重要
- 链接质量：如果一个网页被一个越高权值的网页链接，也表明这个网页越重要。
- 将文档视作一个词的网络，网络之间的链接表示词与词之间的语义关系
- 任意两个句子的相似性等价于网页转换概率
- 相似性存储在一个方形矩阵中，即PageRank算法中的矩阵M

##### 2）算法计算公式

$$
WS(V_i)=(1-d)+d*\sum_{Vj\in In(V_i)}\frac{W_{ji}}{\sum_{V_k\in Out（V_j）W_{jk}}}WS(V_j)
$$

- $WS(V_i)$表示句子i的权重
- $W_{ji}$表示两个句子的相似度
- $WS(V_j)$表示上次迭代出的句子j的权重
- d为阻尼系数一般取0.85

###### 3）改进

传统的TextRank算法只针对本文章进行关键词统计，为了提高准确度，在权重上多考虑两个方面。一个是词的位置权重，在段首，句首的词通常重要程度更高，第二是统计人民日报语料库中的关键词作为第三个考虑方面，提高整体精确度。

##### 4）系统的主要模块（完整代码见附件）

```python
def Set Text(self,title,text) # 设置读取需要自动摘要的文本
def SplitSentence（self）# 对文本进行预处理包含以下几个部分
"""
1.切分段落
2.对段落进行分词，并且在sentence中标记该词的位置，用于后续计算
"""
def CalcKeywords(self) # 计算词频取千20个作为关键词，其中分词与词性标注使用jieba库进行
def GetKeyWordWeight(self) #计算关键词权重
def GetLocalWeight(self)# 计算句子的位置权重First+2，Last+1
def GetCueWordsWeight(self)# 计算句子的线索词权重，其中关键词使用人民日报语料库中词频出现次数最多的前1000个作为关键词
def __CalcSentenceWeight(self)#计算总权重
def CaclSummary（self）#对权值进行排序，取前%10作为摘要
```



### 2.基于Seq2Seq结构与attention机制的beam search文本摘要

##### 1）核心思想

- Seq2seq结构

  Seq2seq由两个部分组成：encoder，decoder。encoder对输入的内容进行处理输出一个向量context，decoder接受context产生输出序列。

- 注意力机制

  将一个查询和键值对映射到输出的方法，Q、K、V均为向量，输出通过对V进行加权求和得到，权重就是Q、K相似度。

- Beam Search算法

  对于普通的seq2seq模型的decoder通常选取t-1时刻output经过softmax计算得到概率最大的word作为t时刻的input，是一种贪心策略无法得到最全局最优解。beam search则是保存最好的k个候选，每个step在k*n个选项中选出最好的k个。

##### 2）模型选择

RNN模型：

![image-20211226032252019](C:\Users\szy45\AppData\Roaming\Typora\typora-user-images\image-20211226032252019.png)

##### 3）系统主要文件与模块

以下内容参考博客：

[(28条消息) 文本摘要生成任务_frank_zhaojianbo的博客-CSDN博客_文本生成任务](https://blog.csdn.net/frank_zhaojianbo/article/details/106777304)

[(28条消息) 集束搜索（Beam Search Algorithm ）_DavidChen的博客-CSDN博客_beam search](https://blog.csdn.net/sdgihshdv/article/details/76737537)

使用keras以及tensorflow工具进行数据集的训练

model.py

parse.py #文本预处理

test.py  #模型预测

train.py #数据集训练

untils.py

目前仅实现parse.py，其他部分由于难度过大，且项目由个人完成还在实现中

### 4.数据获取，处理与分析

##### 1.在抽取式文本摘要中，由于新闻信息的模板化，其关键词信息并不需要特别大的数据采样，故直接采用之前用于学习的人民日报语料库，即可获得如 据报道等关键词信息。

##### 2.在生成式文本摘要中，需要大量的数据集进行模型的训练。预计使用搜狗新闻语料库进行模型训练，该语料库获取方式简单且数据样本充足，下载地址为[搜狗实验室（Sogou Labs）](http://www.sogou.com/labs/resource/ca.php)

### 5.APP窗口

将使用pywebio制作基于python的web窗口，后续可使用CSS+html优化界面，由于并未部署服务器，故直接弹出浏览器窗口进行调试，后续可使用webSocket部署

pyWebIO详细使用手册见[pywebio.platform — 应用部署 — PyWebIO 1.5.1 文档](https://pywebio.readthedocs.io/zh_CN/latest/platform.html)

### 6.版本号 1.0.0
