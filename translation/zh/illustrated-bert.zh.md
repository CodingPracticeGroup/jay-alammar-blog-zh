<strong>2021 年更新：</strong>我制作了一个简短且非常易懂的 BERT 视频入门

 <div style="text-align:center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/ioGry-89gqE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
style="
width: 100%;
max-width: 560px;"
allowfullscreen></iframe>
</div>


2018 年对于处理文本的机器学习模型（更准确地说，是自然语言处理，简称 NLP）来说是一个转折点。我们对于"如何以最能捕捉内在含义与关系的方式表示词和句子"的概念理解正在快速演进。此外，NLP 社区还推出了一系列极其强大的组件，你可以免费下载并用在自家的模型和流程中<span class="faded_text">（这被称为<a href="http://ruder.io/nlp-imagenet/">NLP 的 ImageNet 时刻</a>，指的是多年前类似的发展曾加速了计算机视觉任务中机器学习的进步）</span>。

<div class="img-div-any-width" markdown="0">
  <image src="/images/transformer-ber-ulmfit-elmo.png"/>

</div>



<span class="faded_text">（ULM-FiT 与甜饼怪 Cookie Monster 没有任何关系。但我实在想不到别的什么了……）</span>

这一浪潮中最新的里程碑之一，是 <a href="https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html">BERT 的发布</a>，这件事被<a href="https://twitter.com/lmthang/status/1050543868041555969">评价为</a>开启了 NLP 的一个新纪元。BERT 是一个刷新了多项语言任务表现纪录的模型。在介绍该模型的论文发布后不久，团队还开源了模型代码，并提供了已在海量数据集上预训练好的模型供下载。这是一个意义重大的进展：任何正在构建涉及语言处理的机器学习模型的人，都可以把这个"重型引擎"当作现成的组件来使用——省下了从零训练一个语言处理模型所需的时间、精力、知识和资源。


<div class="img-div-any-width" markdown="0">
  <img src="/images/bert-transfer-learning.png"/>
  <br />
  BERT 的两个研制步骤。你可以直接下载完成第 1 步（在未标注数据上训练）的预训练模型，只需操心第 2 步的微调。[书本图标<a href="https://commons.wikimedia.org/wiki/File:Documents_icon_-_noun_project_5020.svg">来源</a>]。
</div>


BERT 建立在近期 NLP 社区涌现的一系列巧妙想法之上——包括但不限于<a href="https://arxiv.org/abs/1511.01432">Semi-supervised Sequence Learning</a><span class="faded_text">（作者 <a href="https://twitter.com/iamandrewdai">Andrew Dai</a> 与 <a href="https://twitter.com/quocleix">Quoc Le</a>）</span>、<a href="https://arxiv.org/abs/1802.05365">ELMo</a><span class="faded_text">（作者 <a href="https://twitter.com/mattthemathman">Matthew Peters</a> 及来自 <a href="https://allenai.org/">AI2</a> 和 <a href="https://www.engr.washington.edu/about/bldgs/cse">UW CSE</a> 的研究人员）</span>、<a href="https://arxiv.org/abs/1801.06146">ULMFiT</a><span class="faded_text">（作者 fast.ai 创始人 <a href="https://twitter.com/jeremyphoward">Jeremy Howard</a> 与 <a href="https://twitter.com/seb_ruder">Sebastian Ruder</a>）</span>、<a href="https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf">OpenAI transformer</a><span class="faded_text">（作者 OpenAI 研究员 <a href="https://twitter.com/alecrad">Radford</a>、<a href="https://twitter.com/karthik_r_n">Narasimhan</a>、<a href="https://twitter.com/timsalimans">Salimans</a> 与 <a href="https://twitter.com/ilyasut">Sutskever</a>）</span>，以及 Transformer<span class="faded_text">（[Vaswani et al](https://arxiv.org/pdf/1706.03762.pdf)）</span>。

要真正理解 BERT 是什么，需要先了解若干概念。所以我们先来看 BERT 有哪些用法，然后再研究模型本身涉及的概念。

## 示例：句子分类

使用 BERT 最直接的方式，是拿它对一段文本进行分类。这样的模型看起来像这样：

<img src="/images/BERT-classification-spam.png"/>

要训练这样一个模型，你主要需要训练分类器，BERT 模型本身在训练阶段几乎不发生变化。这个训练过程叫做微调（Fine-Tuning），其根源可以追溯到<a href="https://arxiv.org/abs/1511.01432">Semi-supervised Sequence Learning</a> 和 ULMFiT。

对于不熟悉这个领域的人来说：既然说的是分类器，那我们就处于机器学习的监督学习范畴。这意味着我们需要一个带标注的数据集来训练这样的模型。以这个垃圾邮件分类器为例，标注数据集就是一批邮件消息，每条消息带一个标签（"spam" 或 "not spam"）。


<div class="img-div-any-width" markdown="0">
  <image src="/images/spam-labeled-dataset.png"/>
</div>

这类用例的其他例子包括：

* **情感分析**
   * 输入：电影/商品评论。输出：评论是正面还是负面？
   * 示例数据集：[SST](https://nlp.stanford.edu/sentiment/)
* **事实核查**
  * 输入：句子。输出："Claim"（声明）或 "Not Claim"（非声明）
  * 更宏大/更具未来感的例子：
     * 输入：声明句。输出："True"（真）或 "False"（假）
  * [Full Fact](https://fullfact.org/) 是一家为公众利益构建自动事实核查工具的组织。他们的流程中有一个分类器，负责阅读新闻文章并检测声明（把文本分类为"声明"或"非声明"），这些声明之后可以被事实核查（目前由人工完成，希望将来能由机器学习完成）。
  * 视频：[Sentence embeddings for automated factchecking - Lev Konstantinovskiy](https://www.youtube.com/watch?v=ddf0lgPCoSo)。


## 模型架构

现在你脑海中已经有了 BERT 用法的一个示例，接下来我们仔细看看它是如何工作的。

<img src="/images/bert-base-bert-large.png"/>

论文给出了 BERT 的两种规模：

 * BERT BASE——大小与 OpenAI Transformer 相当，便于进行性能对比
 * BERT LARGE——一个庞大得夸张的模型，取得了论文中报告的当时最先进的结果


BERT 基本上就是一个训练好的 Transformer 编码器堆叠。趁此机会，推荐你先阅读我之前的文章 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)，它解释了 Transformer 模型——这是 BERT 以及我们接下来讨论的概念的基础。

<img src="/images/bert-base-bert-large-encoders.png"/>

两种规模的 BERT 都有非常多的编码器层（论文中称之为 Transformer Blocks）——Base 版是 12 层，Large 版是 24 层。它们的前馈网络也更大（隐藏单元分别为 768 和 1024），注意力头也更多（分别为 12 和 16），都超过了原始论文 Transformer 参考实现的默认配置（6 层编码器、512 个隐藏单元、8 个注意力头）。

### 模型输入

<img src="/images/bert-input-output.png"/>

第一个输入 token 会被替换为一个特殊的 [CLS] token，原因稍后会明了。这里的 CLS 代表 Classification（分类）。

和标准的 Transformer 编码器一样，BERT 接收一个词序列作为输入，这些向量沿着堆叠逐层向上传递。每一层都施加 self-attention（自注意力），把结果通过前馈网络，再传给下一个编码器。


<img src="/images/bert-encoders-input.png"/>

就架构而言，到这里为止它与 Transformer 完全相同（除了规模，那只是我们可以设置的配置）。分歧首先出现在输出端。

### 模型输出

每个位置都会输出一个大小为 *hidden_size* 的向量（BERT Base 中为 768）。对于我们上面看过的句子分类示例，我们只关注第一个位置（即传入特殊 [CLS] token 的位置）的输出。

<img src="/images/bert-output-vector.png"/>

这个向量现在可以作为我们选择的任意分类器的输入。论文仅用一个单层神经网络作为分类器，就取得了很好的结果。

<img src="/images/bert-classifier.png"/>

如果你有多个标签（例如你是一家邮件服务商，要给邮件打上 "spam"、"not spam"、"social"、"promotion" 标签），只需把分类器网络改成更多的输出神经元，其后接 softmax 即可。

## 与卷积神经网络的类比

有计算机视觉背景的读者，应该会对这种"向量交接"的做法感到眼熟——在 VGGNet 这类网络中，卷积部分与网络末端的分类用全连接部分之间发生的正是类似的事情。

<img src="/images/vgg-net-classifier.png"/>


## 嵌入的新纪元

这些新进展带来了词编码方式的一次转变。在此之前，词嵌入一直是主流 NLP 模型处理语言的主要手段。Word2Vec 和 GloVe 等方法被广泛使用。我们先回顾这些方法的用法，再来看如今什么变了。

### 词嵌入回顾

词要被机器学习模型处理，就需要某种模型可用于计算的数值表示。Word2Vec 表明，我们可以用一个向量（一串数字）来表示词，并且这种表示能捕捉*语义*（meaning-related）关系（例如判断词是否相似、是否反义，或者 "Stockholm"（斯德哥尔摩）与 "Sweden"（瑞典）之间的关系，同 "Cairo"（开罗）与 "Egypt"（埃及）之间的关系相同），以及句法（grammar-based）关系（例如 "had" 与 "has" 之间的关系，同 "was" 与 "is" 之间的关系相同）。

这个领域很快就意识到：使用在海量文本上预训练好的嵌入，而不是在一个往往很小的数据集上随模型一起训练嵌入，是一个非常棒的主意。于是，下载一份由 Word2Vec 或 GloVe 预训练生成的词及其嵌入列表成为可能。下面是词 "stick" 的 GloVe 嵌入示例（嵌入向量大小为 200）：


<div class="img-div-any-width" markdown="0">
  <img src="/images/glove-embedding.png"/>
  <br />
  词 "stick" 的 GloVe 词嵌入——一个包含 200 个浮点数（四舍五入保留两位小数）的向量。它总共有二百个值。
</div>

由于这些向量很大且全是数字，我在文章插图中用下面这个简单的方框形状来表示向量：

<div class="img-div-any-width" markdown="0">
  <img src="/images/vector-boxes.png"/>
  <br />
</div>



### ELMo：上下文很重要

如果我们使用这个 GloVe 表示，那么无论在什么上下文里，词 "stick" 都会用这同一个向量表示。"等一下，"许多 NLP 研究者说道<span class="faded_text">（<a href="https://arxiv.org/abs/1705.00108">Peters et. al., 2017</a>、<a href="https://arxiv.org/abs/1708.00107">McCann et. al., 2017</a>，以及<a href="https://arxiv.org/pdf/1802.05365.pdf">在 ELMo 论文中再次提出的 Peters et. al., 2018</a>）</span>，"*stick*" 在不同使用场合有多个含义。为什么不根据它所处的上下文来给出嵌入呢——这样既能捕捉词在该上下文中的含义，也能利用其他上下文信息？于是，*上下文化（contextualized）*词嵌入诞生了。

<div class="img-div-any-width" markdown="0">
  <img src="/images/elmo-embedding-robin-williams.png"/>
  <br />
  上下文化词嵌入可以根据词在句子上下文中承载的含义不同，给出不同的嵌入。另外，<a href="https://www.youtube.com/watch?v=OwwdgsN9wF8">RIP Robin Williams</a>
</div>

ELMo 不使用每个词固定的嵌入，而是先通观整个句子，再为句中的每个词分配嵌入。它使用一个针对特定任务训练的双向 LSTM 来生成这些嵌入。

<div class="img-div-any-width" markdown="0">
  <img src="/images/elmo-word-embedding.png"/>
  <br />

</div>

ELMo 朝 NLP 语境下的预训练迈出了重要的一步。ELMo 的 LSTM 会在与我们数据集同语言的大规模数据集上训练，然后我们就可以把它当作组件用在其他需要处理语言的模型中。

ELMo 的秘诀是什么？

ELMo 的语言理解来自"预测词序列中的下一个词"的训练——这项任务叫*语言建模（Language Modeling）*。这很方便，因为我们有海量文本数据可供模型学习，而无需要求标签。


<div class="img-div-any-width" markdown="1">
  <img src="/images/Bert-language-modeling.png"/>
  <br />
  ELMo 预训练过程的一步：给定输入 "Let's stick to"，预测下一个最可能的词——一个*语言建模*任务。在大数据集上训练后，模型开始学到语言模式。它不太可能准确猜出这个例子的下一个词。更现实的情况是，在 "hang" 这样的词之后，它会给 "out"（组成 "hang out"）这类词分配比 "camera" 更高的概率。
</div>

我们可以看到，每个展开的 LSTM 步骤的隐藏状态都从 ELMo 的脑袋后面探出来。预训练完成后的嵌入过程要用到的正是这些隐藏状态。

ELMo 实际上更进一步，训练的是一个双向 LSTM——这样它的语言模型不仅感知下一个词，也感知前一个词。


<div class="img-div-any-width" markdown="0">
  <img src="/images/elmo-forward-backward-language-model-embedding.png"/>
  <br />
  关于 ELMo 的<a href="https://www.slideshare.net/shuntaroy/a-review-of-deep-contextualized-word-representations-peters-2018">很棒的讲义</a>
</div>

ELMo 通过以某种方式组合隐藏状态（以及初始嵌入）来得出上下文化嵌入（先拼接，再加权求和）。

<div class="img-div-any-width" markdown="0">
  <img src="/images/elmo-embedding.png"/>
</div>




## ULM-FiT：把迁移学习在 NLP 中做扎实
ULM-FiT 提出了一套方法，能有效利用模型在预训练中学到的大量内容——不只是嵌入，也不只是上下文化嵌入。ULM-FiT 提出了一个语言模型以及一整套流程，把这个语言模型高效地微调到各种任务上。

NLP 终于有了一种迁移学习方法，其效果可能和计算机视觉领域的一样好。

## Transformer：超越 LSTM
Transformer 论文和代码的发布，以及它在机器翻译等任务上取得的成绩，让领域中的一些人开始把它看作 LSTM 的替代品。再加上 Transformer 处理长期依赖（long-term dependencies）的能力本来就比 LSTM 更好，这一趋势就更明显了。

Transformer 的编码器-解码器结构非常适合机器翻译。但如果用它来做句子分类呢？用它来预训练一个可以微调到其它任务的语言模型呢？（学界把这些使用预训练模型或组件的监督学习任务称为*下游（downstream）*任务。）

## OpenAI Transformer：为语言建模预训练 Transformer 解码器
事实证明，要在 NLP 任务上采用迁移学习和可微调的语言模型，我们并不需要完整的 Transformer。只用 Transformer 的解码器就够了。解码器是个好选择，因为它做语言建模（预测下一个词）是天生的——它本来就会掩码（mask）未来的 token，而这是它逐词生成翻译时的宝贵特性。

<div class="img-div-any-width" markdown="0">
  <img src="/images/openai-transformer-1.png"/>
  <br />
  OpenAI Transformer 由 Transformer 的解码器堆叠构成
</div>

这个模型堆叠了 12 层解码器。由于这套结构中没有编码器，这些解码器层不存在标准 Transformer 解码器层中的 encoder-decoder attention 子层。但它仍然保留自注意力层（经过掩码，使其不会偷看未来的 token）。

有了这个结构，我们就可以用同样的语言建模任务来训练模型：在海量（无标注）数据集上预测下一个词。把 7,000 本书的文本丢给它，让它学起来！书籍非常适合这类任务，因为它能让模型学会把相关信息即使在相隔大量文本的情况下也关联起来——用推文或文章训练时就得不到这种效果。


<div class="img-div-any-width" markdown="0">
  <img src="/images/openai-transformer-language-modeling.png"/>
  <br />
  OpenAI Transformer 现在已准备好在一个由 7,000 本书组成的数据集上训练，学习预测下一个词。
</div>

## 迁移到下游任务

现在 OpenAI Transformer 已经完成了预训练，其各层已经能较好地处理语言，我们可以开始把它用于下游任务了。先看句子分类（把邮件消息分类为"spam" 或 "not spam"）：

<div class="img-div-any-width" markdown="0">
  <img src="/images/openai-transformer-sentence-classification.png"/>
  <br />

  如何使用预训练的 OpenAI Transformer 做句子分类
</div>

OpenAI 的论文列出了若干种输入变换，用于处理不同类型任务的输入。下图来自论文，展示了针对不同的任务所采用的模型结构与输入变换。


<div class="img-div-any-width" markdown="0">
  <img src="/images/openai-input transformations.png"/>
  <br />
</div>


是不是很巧妙？

## BERT：从解码器到编码器
OpenAI Transformer 给了我们一个基于 Transformer 的、可微调的预训练模型。但从 LSTM 转向 Transformer 的过程中，有些东西丢了。ELMo 的语言模型是双向的，而 OpenAI Transformer 只训练了单向（前向）语言模型。我们能不能构建一个基于 Transformer 的模型，让它的语言模型同时向前和向后看（用专业术语说——"同时以左右上下文为条件"）？

"扶好我的啤酒。"带着 R 级的 BERT 说道。

### 掩码语言模型（Masked Language Model）

"我们会用 Transformer 编码器。"BERT 说。

"这太疯狂了，"Ernie 回答道，"谁都知道，多层上下文中的双向条件化会让每个词间接'看到'它自己。"

"我们会用掩码。"BERT 自信地说。

<div class="img-div-any-width" markdown="0">
  <img src="/images/BERT-language-modeling-masked-lm.png"/>
  <br />
  BERT 巧妙的语言建模任务把输入中 15% 的词掩码掉，然后要求模型预测被藏起来的词。
</div>

要为堆叠的 Transformer 编码器找到合适的训练任务是个棘手的难题，BERT 通过借鉴早期文献中的"掩码语言模型"思想（那里称之为 Cloze task，完形填空任务）解决了它。

除了掩码 15% 的输入，BERT 还会做一点随机扰动来改善模型后续微调的效果。它有时会把某个词随机替换成另一个词，然后要求模型预测该位置上正确的词。

### 双句任务

回头看看上文 OpenAI Transformer 为处理不同任务所做的输入变换，你会注意到有些任务要求模型对两个句子做出有见地的判断（例如：两个句子是否只是彼此的换一种说法？输入一篇维基百科条目，再把关于该条目的一个问题作为另一个输入，我们能否回答这个问题？）。

为了让 BERT 更擅长处理多个句子之间的关系，预训练过程中加入了一项额外任务：给定两个句子（A 和 B），B 是不是 A 的下一句？

<div class="img-div-any-width" markdown="0">
  <img src="/images/bert-next-sentence-prediction.png"/>
  <br />
  BERT 预训练的第二项任务是一个双句分类任务。这张图里的分词被过度简化了：BERT 实际上使用 WordPiece 作为 token 而不是词——因此有些词会被拆成更小的片段。
</div>




### 任务专用模型
BERT 论文展示了把 BERT 用于不同任务的多种方式。

<div class="img-div-any-width" markdown="0">
  <img src="/images/bert-tasks.png"/>
  <br />
</div>



### 用 BERT 做特征提取
微调并不是使用 BERT 的唯一方式。和 ELMo 一样，你可以用预训练的 BERT 来生成上下文化词嵌入，然后把这些嵌入喂给你已有的模型——论文表明，这样做在命名实体识别等任务上的结果与微调 BERT 相差不远。

<div class="img-div-any-width" markdown="0">
  <img src="/images/bert-contexualized-embeddings.png"/>
  <br />
</div>

哪种向量最适合做上下文化嵌入？我认为这取决于具体任务。论文考察了六种选择（相比之下，微调后的模型得分是 96.4）：

<div class="img-div-any-width" markdown="0">
  <img src="/images/bert-feature-extraction-contextualized-embeddings.png"/>
  <br />
</div>


## 带 BERT 出去遛一遛
体验 BERT 的最佳方式是通过托管在 Google Colab 上的 [BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb) notebook。如果你从未用过 Cloud TPU，这也是一个很好的起点——BERT 代码同样支持在 TPUs、CPU 和 GPU 上运行。

下一步是阅读 [BERT 仓库](https://github.com/google-research/bert)中的代码：

* 模型在 [modeling.py](https://github.com/google-research/bert/blob/master/modeling.py)（```class BertModel```）中构建，与标准的 Transformer 编码器基本完全一致。
* [run_classifier.py](https://github.com/google-research/bert/blob/master/run_classifier.py) 是微调过程的示例。它还构建了监督学习模型的分类层。如果你想自己构建分类器，可以看看该文件中的 ```create_model()``` 方法。

* 提供了多个预训练模型供下载。这些模型涵盖 BERT Base 和 BERT Large，也涵盖英语、中文等语言，以及一个在维基百科上训练的、覆盖 102 种语言的多语言模型。

* BERT 并不把词当作 token，而是使用 WordPiece。[tokenization.py](https://github.com/google-research/bert/blob/master/tokenization.py) 就是那个把你的词转换成适合 BERT 的 wordPiece 的分词器。

你还可以看看 [BERT 的 PyTorch 实现](https://github.com/huggingface/pytorch-pretrained-BERT)。[AllenNLP](https://github.com/allenai/allennlp) 库使用了这个实现，从而可以在任意模型中[使用 BERT 嵌入](https://github.com/allenai/allennlp/pull/2067)。

## 致谢

感谢 [Jacob Devlin](https://github.com/jacobdevlin-google)、[Matt Gardner](https://twitter.com/nlpmattg)、[Kenton Lee](https://github.com/kentonl)、[Mark Neumann](https://twitter.com/markneumannnn) 和 [Matthew Peters](https://twitter.com/mattthemathman) 对本文早期草稿提供的反馈。
