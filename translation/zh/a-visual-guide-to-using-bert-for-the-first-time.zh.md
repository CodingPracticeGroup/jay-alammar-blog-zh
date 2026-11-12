<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-sentence-classification.png"/>
  <br />
</div>

过去几年，处理语言的机器学习模型进展迅猛。这些进展已经走出实验室，开始为一些领先的数字产品提供动力。一个典型的例子是[最近的公告：BERT 模型如今已成为 Google 搜索背后的一支主力](https://www.blog.google/products/search/search-language-understanding-bert/)。Google 认为，这一步（或者说应用于搜索的自然语言理解进展）是"过去五年最大的飞跃，也是搜索史上最大的飞跃之一"。

本文是一篇简单的教程，讲如何用 BERT 的一个变体给句子做分类。这个例子作为入门足够基础，同时又足够进阶，能展示其中若干关键概念。

配合本文，我准备了一个 notebook。你可以在[这里](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)查看，或者[在 colab 上直接运行](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)。


## 数据集：SST2
本例使用的数据集是 [SST2](https://nlp.stanford.edu/sentiment/index.html)，其中是取自电影评论的句子，每句标注为正面（值为 1）或负面（值为 0）：

<table class="features-table">
  <tr>
    <th class="mdc-text-light-green-600">
    句子
    </th>
    <th class="mdc-text-purple-600">
    标签
    </th>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      a stirring , funny and finally transporting re imagining of beauty and the beast and 1930s horror films
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      apparently reassembled from the cutting room floor of any given daytime soap
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      they presume their audience won't sit still for a sociology lesson
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      this is a visually stunning rumination on love , memory , history and the war between art and commerce
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      jonathan parker 's bartleby should have been the be all end all of the modern office anomie films
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
</table>


## 模型：句子情感分类
我们的目标是做一个模型：输入一个句子（就像数据集中的那些），输出 1（表示句子情感正面）或 0（表示负面）。可以把它想象成这样：


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/sentiment-classifier-1.png"/>
  <br />
</div>


模型内部其实由两个模型组成。
 * [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5) 处理句子，并把从中提取的信息传给下一个模型。DistilBERT 是 [HuggingFace](https://huggingface.co/) 团队开发并开源的 BERT 精简版。它更轻、更快，性能却与 BERT 大致相当。
 * 第二个模型是 scikit-learn 的一个基础逻辑回归（Logistic Regression）模型。它接收 DistilBERT 的处理结果，把句子分类为正面或负面（分别对应 1 或 0）。

 两个模型之间传递的数据是一个 768 维向量。可以把它看作我们拿来分类的句子嵌入。


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/distilbert-bert-sentiment-classifier.png"/>
  <br />
</div>

如果你读过我之前的文章 [Illustrated BERT](/illustrated-bert/)，就会知道这个向量是第一个位置（输入为 [CLS] token 的位置）的输出结果。

## 模型训练
虽然我们使用两个模型，但只训练逻辑回归这一个模型。对 DistilBERT，我们直接使用一个已经预训练好、"懂"英语的模型。不过，这个模型既没有为句子分类做过训练，也没有为此做过微调。尽管如此，我们仍然能从 BERT 的训练目标所赋予的一般能力中获得一些句子分类能力——尤其是 BERT 第一个位置（对应 [CLS] token）的输出。我认为这要归功于 BERT 的第二个训练目标——下一句分类（Next sentence classification）。该目标似乎促使模型在第一个位置的输出中概括出整句话的意思。[transformers](https://github.com/huggingface/transformers) 库为我们提供了 DistilBERT 的实现以及预训练好的模型版本。


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/model-training.png"/>
  <br />
</div>

## 教程总览
教程的整体计划是：先用训练好的 DistilBERT 为 2,000 个句子生成句子嵌入。


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-tutorial-sentence-embedding.png"/>
  <br />
</div>

这一步之后就再也不动 DistilBERT 了，剩下的全是 Scikit-Learn 的事。对这个数据集做常规的 train/test 划分：

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-train-test-split-sentence-embedding.png"/>
  <br />
  对 distilBERT（模型 #1）的输出做 train/test 划分，得到用于训练和评估逻辑回归（模型 #2）的数据集。注意实际上 sklearn 的 train/test split 会先把样本打乱再切分，并不是按数据集顺序直接取前 75%。
</div>

然后在训练集上训练逻辑回归模型：

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-training-logistic-regression.png"/>
  <br />

</div>



## 一次预测是如何计算的
在深入代码、讲解如何训练模型之前，先看看训练好的模型如何计算一次预测。

试着给句子 "a visually stunning rumination on love" 分类。第一步，用 BERT 分词器把词拆成 token；然后加上句子分类所需的特殊 token（第一个位置放 [CLS]，句尾放 [SEP]）。

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-tokenization-1.png"/>
  <br />
</div>

分词器做的第三件事，是把每个 token 替换成它在嵌入表中对应的 id——嵌入表是我们随训练好的模型一起获得的组件。词嵌入的背景知识请阅读 [The Illustrated Word2vec](/illustrated-word2vec/)。

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-tokenization-2-token-ids.png"/>
  <br />
</div>
注意，分词器一行代码就完成了以上所有步骤：
```python
tokenizer.encode("a visually stunning rumination on love", add_special_tokens=True)
```

现在，输入句子的形状就可以送进 DistilBERT 了。

如果你读过 [Illustrated BERT](/illustrated-bert/)，这一步也可以用这种方式可视化：

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-input-tokenization.png"/>
  <br />
</div>


## 流经 DistilBERT

把输入向量送入 DistilBERT 的过程[与 BERT 完全一样](/illustrated-bert/)。输出是每个输入 token 对应的一个向量，每个向量由 768 个浮点数组成。

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-model-input-output-1.png"/>
  <br />
</div>

因为这是句子分类任务，除了第一个向量（与 [CLS] token 对应的那个）外，其余全部忽略。就把这一个向量作为输入送给逻辑回归模型。

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-model-calssification-output-vector-cls.png"/>
  <br />
</div>

接下来，根据训练阶段学到的东西对这个向量分类，就是逻辑回归模型的工作了。一次预测的计算过程可以想象成这样：

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-sentence-classification-example.png"/>
  <br />
</div>

训练将在下一节讨论，连同整个流程的代码。


## 代码
本节梳理训练这个句子分类模型的代码。包含全部代码的 notebook 在 [colab](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) 和 [github](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) 上都有。

先导入"吃饭的家伙"：

```python
import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
```


数据集在 github 上有一个[文件](https://github.com/clairett/pytorch-sentiment-classification/)，我们直接把它读进 pandas dataframe：

```python
df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
```

可以用 df.head() 看 dataframe 的前五行，了解数据长什么样：

```python
df.head()
```
输出如下：

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/sst2-df-head.png"/>
  <br />
</div>

### 导入预训练的 DistilBERT 模型与分词器

```python
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
```

现在可以给整个数据集分词了。注意这里和上面的例子略有不同：上面只对单个句子分词、处理；这里是把所有句子作为一个批次一起分词和处理（notebook 里出于资源考虑只处理较小的一组样本，比如 2000 个）。

### 分词
```python
tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
```

这把每个句子变成一个 id 列表。

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/sst2-text-to-tokenized-ids-bert-example.png"/>
  <br />
</div>

数据集现在是一个"列表的列表"（pandas Series/DataFrame）。DistilBERT 要把它作为输入处理，必须先让所有向量长度一致——用 token id 0 给较短的句子补齐（padding）。补齐步骤见 notebook，就是基础的 python 字符串和数组操作。

补齐之后，我们就有了一个可以直接传给 BERT 的矩阵/张量：

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-input-tensor.png"/>
  <br />
</div>



### 用 DistilBERT 处理

现在把补齐后的 token 矩阵做成输入张量，送入 DistilBERT：

```python
input_ids = torch.tensor(np.array(padded))

with torch.no_grad():
    last_hidden_states = model(input_ids)
```

运行这一步后，`last_hidden_states` 就装着 DistilBERT 的输出。它是一个元组，形状为（样本数，序列最大 token 数，DistilBERT 模型的隐藏单元数）。在本例中即 2000（我们只用了 2000 个样本）、66（这 2000 个样本里最长序列的 token 数）、768（DistilBERT 的隐藏单元数）。

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-output-tensor-predictions.png"/>
  <br />
</div>

### 解包 BERT 的输出张量

我们来拆解这个三维输出张量。先看看各维度的含义：

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-output-tensor.png"/>
  <br />
</div>

### 回顾一个句子的旅程
每一行都对应数据集中的一句话。回顾第一个句子的处理路径，可以把它想象成这样：

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-input-to-output-tensor-recap.png"/>
  <br />
</div>


### 只切出有用的部分
句子分类只关心 BERT 对 [CLS] token 的输出，所以切出立方体中的这一片，其余全部丢弃。

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-output-tensor-selection.png"/>
  <br />
</div>

从三维张量里切出我们需要的二维张量，代码是这样：

```python
 # Slice the output for the first position for all the sequences, take all hidden unit outputs
features = last_hidden_states[0][:,0,:].numpy()
```

现在 `features` 是一个二维 numpy 数组，装着数据集中所有句子的句子嵌入。

<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-output-cls-senteence-embeddings.png"/>
  <br />
  我们从 BERT 输出中切出的张量
</div>


## 供逻辑回归使用的数据集
有了 BERT 的输出，训练逻辑回归模型所需的数据集就组装好了：768 列是特征，标签直接取自最初的数据集。


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/logistic-regression-dataset-features-labels.png"/>
  <br />
  用于训练逻辑回归的带标签数据集。特征是上一图中切出的 BERT 对 [CLS] token（位置 #0）的输出向量。每行对应数据集里的一句话；每列对应 Bert/DistilBERT 模型顶层 Transformer Block 中前馈神经网络某个隐藏单元的输出。
</div>

做完机器学习的常规 train/test 划分后，就可以声明逻辑回归模型并在数据集上训练它：


```python
labels = df[1]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

```
这把数据集分成了训练/测试集：


<div class="img-div-any-width" markdown="0">
  <image src="/images/distilBERT/bert-distilbert-train-test-split-sentence-embedding.png"/>
  <br />
</div>

接着在训练集上训练逻辑回归模型。

```python
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
```

模型训练好之后，用测试集给它打分：

```python
lr_clf.score(test_features, test_labels)
```

结果显示模型准确率约 81%。

## 分数基准
作为参照，该数据集目前的最高准确率是 **96.8**。DistilBERT 可以被进一步训练来提升这项任务的成绩——这个过程叫微调（fine-tuning），即更新 BERT 的权重，让它在句子分类（所谓*下游任务*）上表现更好。微调后的 DistilBERT 能达到 **90.7** 的准确率，完整规模的 BERT 模型能达到 **94.9**。

## Notebook
直接看 [notebook](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)，或[在 colab 上运行](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb)。


就是这样！这是与 BERT 的一次很好的初次接触。下一步应该去读官方文档，亲自动手试试[微调](https://huggingface.co/transformers/examples.html#glue)。你也可以回头把 distilBERT 换成完整的 BERT，看看效果有何不同。

感谢 [Clément Delangue](https://twitter.com/ClementDelangue)、[Victor Sanh](https://twitter.com/SanhEstPasMoi) 和 HuggingFace 团队对本文早期版本提供的反馈。
