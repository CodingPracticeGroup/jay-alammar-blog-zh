<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/word2vec.png"/>
  <br />
</div>


<blockquote class='subtle'>
  “<strong>万物之中都存在一种作为我们宇宙一部分的模式。它有对称、优雅与和谐</strong>——这些品质你总能在真正的艺术家所捕捉到的事物中找到。你能在四季的轮转中找到它，在沙脊上沙粒拖出的纹路中找到它，在杂酚灌木丛的枝条簇拥中，或在它叶片的纹理中找到它。<br /><br />

  我们试图在我们的生活和社会中复刻这些模式，追寻那些令人安心的节奏、舞步与形式。
  然而，在追寻终极完美的过程中，也可能看见危险。很显然，终极的模式包含它自身的僵化。在这样的完美中，万物都趋向死亡。”
  ~《沙丘》（1965）
</blockquote>


我觉得"嵌入（embeddings）"这个概念是机器学习中最迷人的想法之一。如果你用过 Siri、Google Assistant、Alexa、Google 翻译，甚至只是带下一个词预测功能的手机键盘，那么你很可能已经受益于这个如今已成为自然语言处理模型核心思想的概念。用嵌入来构建神经模型在过去二十年里有了长足的发展（近期的进展包括上下文化词嵌入，催生了像 [BERT](https://jalammar.github.io/illustrated-bert/) 和 GPT2 这样的前沿模型）。

<iframe width="560" height="315" src="https://www.youtube.com/embed/ISPId9Lhc1g" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"  style="
width: 100%;
max-width: 560px;" allowfullscreen></iframe>

Word2vec 是一种高效创建词嵌入的方法，自 2013 年问世至今。但除了作为词嵌入方法的实用性之外，它的一些概念也被证明在构建推荐系统、以及理解序列数据方面非常有效，甚至可以用在商业上的非语言任务中。[Airbnb](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb)、[阿里巴巴](https://www.kdd.org/kdd2018/accepted-papers/view/billion-scale-commodity-embedding-for-e-commerce-recommendation-in-alibaba)、[Spotify](https://www.slideshare.net/AndySloane/machine-learning-spotify-madison-big-data-meetup) 和 [Anghami](https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484) 等公司都从中受益——它们把这台从 NLP 世界里拆解出来的精妙机器搬到生产环境中，支撑起新一代推荐系统。

在这篇文章中，我们会讲解嵌入的概念，以及用 word2vec 生成嵌入的机制。不过我们先从一个例子入手，熟悉"用向量表示事物"的做法。你知道吗，一串五个数字（一个向量）就能表达关于你性格的这么多信息？

# 性格嵌入：你是一个什么样的人？
<blockquote class='subtle'>
“我给你介绍沙漠变色龙——它融入背景的能力，告诉了你关于生态学根源与个人身份基础所需要知道的一切” ~《沙丘之子》
</blockquote>

以 0 到 100 分计，你在内向/外向这个维度上得多少分（0 表示最内向，100 表示最外向）？
你做过 MBTI 之类的性格测试吗——或者更好的，[大五人格特质（Big Five Personality Traits）](https://en.wikipedia.org/wiki/Big_Five_personality_traits)测试？如果没有：这类测试会问你一串问题，然后在若干维度上给你打分，内向/外向就是其中之一。

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/big-five-personality-traits-score.png"/>
  <br />
  大五人格特质测试的结果示例。它真的能告诉你很多关于你自己的信息，并且被证明在<a href="http://psychology.okstate.edu/faculty/jgrice/psyc4333/FiveFactor_GPAPaper.pdf">学业</a>、<a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1744-6570.1999.tb00174.x">个人生活</a>和<a href="https://www.massgeneral.org/psychiatry/assets/published_papers/soldz-1999.pdf">职业成功</a>方面具有预测能力。<a href="https://projects.fivethirtyeight.com/personality-quiz/">这里</a>可以测出你自己的结果。
</div>


假设我的内向/外显得分是 38/100，我们可以这样把它画出来：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/introversion-extraversion-100.png"/>
</div>

我们把取值范围改成从 -1 到 1：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/introversion-extraversion-1.png"/>
</div>

仅凭这一条信息，你觉得你对一个人的了解有多少？不多。人是复杂的。那我们再加一个维度——测试中另一个特质的得分。

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/two-traits-vector.png"/>
  <br />
  我们可以把这两个维度表示为图上的一个点，或者更好的做法：表示为从原点到该点的一个向量。我们拥有处理向量的强大工具，马上就会派上用场。
</div>

我故意隐去了画的是哪两个特质，好让你习惯"不知道每个维度代表什么，却依然能从一个人的性格向量表示中获得大量价值"这种感觉。

现在我们可以说，这个向量部分地代表了我的性格。这种表示的用处体现在：当你想把我跟其他人做比较的时候。假设我出门被一辆 ```bus```（公交车）撞了，需要找一个性格相似的人来顶替我。下图中，这两个人谁跟我更像？


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/personality-two-persons.png"/>
</div>

处理向量时，计算相似度分数的一种常用方法是[余弦相似度（cosine_similarity）](https://en.wikipedia.org/wiki/Cosine_similarity)：


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/cosine-similarity.png"/>
  <br />
  <span class="encoder">Person #1</span> 的性格与我更相似。指向相同方向的向量（长度也会起作用）余弦相似度分数更高。
</div>


还是那句话，两个维度不足以捕捉人与人之间的差异。几十年的心理学研究归纳出了五大主要特质（以及大量次级特质）。那我们就把这五个维度都用上：


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/big-five-vectors.png"/>
  <br />
</div>

五个维度的问题在于：我们没法再在二维平面上优雅地画小箭头了。这是机器学习中常见的挑战——我们经常不得不在高维空间中思考。不过好消息是，cosine_similarity 依然适用，它对任意数量的维度都有效：


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/embeddings-cosine-personality.png"/>
  <br />
  cosine_similarity 适用于任意数量的维度。这些分数要好得多，因为它们是基于被比较对象的更高分辨率表示计算出来的。
</div>

本节结束时，我希望带走两个核心思想：

1. 我们可以把人（以及事物）表示为数值的向量（这对机器来说再合适不过了！）。
2. 我们可以方便地计算向量之间的相似度。


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/section-1-takeaway-vectors-cosine.png"/>
  <br />
</div>


# 词嵌入

<blockquote class='subtle'>
“语言的馈赠，是欺骗与幻象的馈赠” ~《沙丘之子》
</blockquote>

有了上述理解，我们就可以来看训练好的词向量示例（也叫词嵌入），并观察它们一些有趣的性质。

这是词 "king" 的一个词嵌入（在维基百科上训练的 GloVe 向量）：

<code class="small_code">
[ 0.50451 ,  0.68607 , -0.59517 , -0.022801,  0.60046 , -0.13498 ,
 -0.08813 ,  0.47377 , -0.61798 , -0.31012 , -0.076666,  1.493   ,
 -0.034189, -0.98173 ,  0.68229 ,  0.81722 , -0.51874 , -0.31503 ,
 -0.55809 ,  0.66421 ,  0.1961  , -0.13495 , -0.11476 , -0.30344 ,
  0.41177 , -2.223   , -1.0756  , -1.0783  , -0.34354 ,  0.33505 ,
  1.9927  , -0.04234 , -0.64319 ,  0.71125 ,  0.49159 ,  0.16754 ,
  0.34344 , -0.25663 , -0.8523  ,  0.1661  ,  0.40102 ,  1.1685  ,
 -1.0137  , -0.21585 , -0.15155 ,  0.78321 , -0.91241 , -1.6106  ,
 -0.64426 , -0.51042 ]
 </code>

这是一个 50 个数字的列表。光看数值我们说不出什么名堂。但我们可以稍微把它可视化一下，以便和其他词向量做比较。把这一串数字排成一行：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/king-white-embedding.png"/>
  <br />
</div>

按数值给格子着色（接近 2 的标红色，接近 0 的留白色，接近 -2 的标蓝色）：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/king-colored-embedding.png"/>
  <br />
</div>

接下来我们忽略具体数字，只看颜色来代表格子的取值。现在把 "King" 和其他词对比一下：


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/king-man-woman-embedding.png"/>
  <br />
</div>

看到没有："Man" 和 "Woman" 彼此的相似度，远高于它们各自与 "king" 的相似度？这说明了些什么。这些向量表示确实捕捉到了这些词的大量信息/含义/联想。

这里还有另一组例子（沿着列竖向扫视，找颜色相似的列）：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/queen-woman-girl-embeddings.png"/>
  <br />
</div>

指出几点：

1. 有一条笔直的红色列贯穿所有这些不同的词。它们在那个维度上很相似（尽管我们并不知道每个维度编码的是什么）
2. 你可以看到 "woman" 和 "girl" 在很多地方彼此相似。"man" 和 "boy" 也一样
3. "boy" 和 "girl" 也有彼此相似、但与 "woman" 或 "man" 不同的地方。这些维度会不会在编码某种模糊的"年轻"概念？有可能。
4. 除最后一个词外，其余都是表示人的词。我特意加了一个物体（water）来展示类别之间的差异。例如你可以看到那一整条蓝色列一路延伸，到 "water" 的嵌入之前就停了。
5. "king" 和 "queen" 之间有明显的相似之处，并且与其他所有词都不同。这些维度会不会在编码某种模糊的"王权"概念？

## 类比

<blockquote class='subtle'>
"词语可以承载我们希望加诸的任何分量。所需的只是共识，以及一个可供建造的传统。" ~《沙丘神帝》
</blockquote>

展示嵌入这一惊人性质的著名例子，是"类比（analogies）"的概念。我们可以对词嵌入做加减法，得到有趣的结果。最有名的例子是这个公式："king" - "man" + "woman"：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/king-man+woman-gensim.png"/>
  <br />
  使用 Python 的 <a href="https://radimrehurek.com/gensim/">Gensim</a> 库，我们可以对词向量做加减法，它会找出与结果向量最相似的词。图中列出了最相似的词，每个都带有其余弦相似度。
</div>

我们可以像之前那样把这个类比可视化：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/king-analogy-viz.png"/>
  <br />
  "king-man+woman" 得到的向量并不严格等于 "queen"，但在这个集合包含的 40 万个词嵌入中，"queen" 是与它最接近的词。
</div>

看完了训练好的词嵌入，让我们来了解训练过程。但在讲 word2vec 之前，我们需要先看词嵌入的概念上的"前身"：神经语言模型。

# 语言建模

<blockquote class='subtle'>
  “先知不会被过去、现在与未来的幻象带偏。<strong>语言的固定性决定了这类线性的区分。</strong>先知握有开启语言中那把锁的钥匙。<br /> <br />

  这不是一个机械的宇宙。事件的线性进程是观察者强加的。因与果？根本不是那么回事。<strong>先知说出的是命运之语。</strong>你瞥见某件'注定会发生'的事。但预言之刻释放出的，是具有无限预兆与力量的东西。宇宙经历了一次幽灵般的位移。” ~《沙丘神帝》
</blockquote>

如果要举一个 NLP 应用的例子，最好的例子之一就是手机键盘的"下一个词预测"功能。这是一个每天被数十亿人使用数百次的功能。

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/swiftkey-keyboard.png"/>
  <br />
</div>

下一个词预测是*语言模型*可以解决的任务。语言模型可以接收一串词（比如说两个词），并尝试预测紧随其后的词。

在上面这张截图中，我们可以把模型想象成接收了那两个绿色的词（<code class="plain_code mdc-text-green-600">thou shalt</code>），然后返回一个候选词列表（"not" 是概率最高的那个）：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/thou-shalt-_.png"/>
  <br />
</div>

<br />

我们可以把这个模型想成这样一个黑盒：

<br />

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/language_model_blackbox.png"/>
  <br />
</div>

<br />

但实际中，模型并不会只输出一个词。它实际上会输出它认识的所有词（模型的"词表"，规模可以从几千个词到超过一百万个词）的概率分数。键盘应用随后要找出分数最高的那些词，把它们呈现给用户。

<br />

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/language_model_blackbox_output_vector.png"/>
  <br />
  神经语言模型的输出是一个向量，包含模型认识的每个词的概率分数。这里我们把概率说成百分比，但在输出向量中 40% 实际表示为 0.4。
</div>

<br />

早期的神经语言模型（[Bengio 2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)）训练完成后，会分三步计算一次预测：

<br />

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/neural-language-model-prediction.png"/>
  <br />
</div>

<br />

当我们讨论嵌入时，第一步与我们最相关。训练过程的产出之一，就是这样一个矩阵：词表中的每个词在其中都有一个嵌入。预测时，我们只需查表取出输入词的嵌入，用它们来计算预测：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/neural-language-model-embedding.png"/>
  <br />
</div>

现在让我们转向训练过程，看看这个嵌入矩阵是怎么来的。


# 语言模型的训练

<blockquote class='subtle'>
“理解一个过程，不能靠让它停下来。理解必须随着过程的流动而流动，必须汇入其中，与之同流。” ~《沙丘》
</blockquote>

语言模型相对大多数其他机器学习模型有一个巨大优势：我们能拿连续的文本（running text）来训练它——而文本我们多得很。想想我们手边有多少书籍、文章、维基百科内容和其他形式的文本数据。对比之下，许多其他机器学习模型需要人工设计的特征和专门采集的数据。

> "View a word by its company, and you shall know it."——你该通过一个词交往的伙伴来认识这个词。 J.R. Firth

词之所以能获得嵌入，靠的是观察它们倾向于和哪些词一起出现。具体机制是：

1. 我们获取大量文本数据（比如全部维基百科条目）。然后
2. 我们有一个窗口（比如三个词宽），在整个文本上滑动。
3. 滑动的窗口为模型生成训练样本

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/wikipedia-sliding-window.png"/>
  <br />
</div>

随着窗口在文本上滑动，我们（虚拟地）生成了一个数据集，用它来训练模型。为了看清具体做法，我们看看滑动窗口如何处理这句话：

> “Thou shalt not make a machine in the likeness of a human mind”（你不可造出一台貌似人类心智的机器）~《沙丘》

一开始，窗口罩在句子的前三个词上：

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/lm-sliding-window.png"/>
  <br />
</div>

<br />  

我们把前两个词当作特征，第三个词当作标签：

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/lm-sliding-window-2.png"/>
  <br />
  现在我们生成了数据集的第一个样本，稍后用它来训练语言模型。
</div>

<br />  

然后把窗口滑到下一个位置，生成第二个样本：

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/lm-sliding-window-3.png"/>
  <br />
  第二个样本就这样生成了。
</div>

<br />  

很快，我们就有了一个更大的数据集，记录了不同的词对后面倾向于出现什么词：

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/lm-sliding-window-4.png"/>
  <br />  
</div>

<br />  

实际中，模型往往是边滑动窗口边训练的。但我认为把"数据集生成"阶段与训练阶段在逻辑上分开来更清楚。除了基于神经网络的语言建模方法，一种叫 N-grams 的技术也常被用于训练语言模型（参见：[Speech and Language Processing](http://web.stanford.edu/~jurafsky/slp3/) 第 3 章）。想了解从 N-grams 到神经模型的转变如何体现在真实产品上，[这里有一篇 Swiftkey 2015 年的博客文章](https://blog.swiftkey.com/neural-networks-a-meaningful-leap-for-mobile-typing/)——我最喜欢的 Android 键盘——介绍了他们的神经语言模型，并与之前的 N-gram 模型作比较。我喜欢这个例子，因为它展示了嵌入的算法特性可以被描述成营销话术。

## 两边都看

<blockquote class='subtle'>
“悖论是一根手指，指向它之外的事物。如果悖论让你不安，那暴露出的是你对绝对之物的深切渴望。相对主义者只把悖论当作有趣的东西——也许可笑，甚至，可怕地，有教育意义。” ~《沙丘神帝》
</blockquote>

运用你在文章前面学到的东西，来填空：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/jay_was_hit_by_a_.png"/>
  <br />  
</div>

我这里给出的上下文是空格前的五个词（外加前文提过的 "bus"）。我猜大多数人会认为空格里应该填 ```bus```。但如果我再给你多一条信息——空格后面的一个词——这会改变你的答案吗？


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/jay_was_hit_by_a_bus.png"/>
  <br />  
</div>

这完全改变了空格处该填什么。现在 ```red``` 是最可能的词。我们从中得知：某个词前后的词都携带信息价值。事实证明，同时考虑两个方向（我们要猜的词左边和右边的词）能带来更好的词嵌入。让我们看看如何调整模型的训练方式以利用这一点。


# Skipgram

<blockquote class='subtle'>
  “智能，是在一个数据有限、而且错误不仅可能而且必要的竞技场上押注。” ~《沙丘母族》
</blockquote>

我们可以不只看向目标词前面的两个词，也可以看它后面的两个词。

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/continuous-bag-of-words-example.png"/>
  <br />  
</div>

如果这样做，我们虚拟构建并用来训练模型的数据集就会是这样：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/continuous-bag-of-words-dataset.png"/>
  <br />  
</div>

这被称为**连续词袋（Continuous Bag of Words，CBOW）**架构，在[word2vec 论文之一](https://arxiv.org/pdf/1301.3781.pdf) [pdf] 中有描述。另一种同样表现出色的架构做法略有不同。

它不是根据上下文（前后的词）来猜一个词，而是试图用当前词去猜它邻近的词。我们可以把它在训练文本上滑动的窗口想象成这样：


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/skipgram-sliding-window.png"/>
  <br />  
  绿色槽里的词是输入词，每个粉色格子都是一个可能的输出。
</div>

粉色格子深浅不一，因为这个滑动窗口实际上在训练数据集中创建了四个独立样本：

<br />  

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/skipgram-sliding-window-samples.png"/>
  <br />  
</div>

<br />  

这种方法称为 **skipgram** 架构。我们可以把滑动窗口想象成在做下面的事：

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/skipgram-sliding-window-1.png"/>
  <br />  
</div>

<br />  

这会给训练数据集添加这四个样本：

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/skipgram-sliding-window-2.png"/>
  <br />  
</div>

接着把窗口滑到下一个位置：

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/skipgram-sliding-window-3.png"/>
  <br />  
</div>
<br />  

生成下一批四个样本：

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/skipgram-sliding-window-4.png"/>
  <br />  
</div>

再滑动几个位置，我们就有了多得多的样本：

<br />  

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/skipgram-sliding-window-5.png"/>
  <br />  
</div>


## 重新审视训练过程

<blockquote class="subtle">
  "穆阿迪布学得快，是因为他接受的第一项训练就是'如何学习'。而所有课程的第一课，是相信自己能够学习这一基本信念。有多少人并不相信自己能学习，又有更多人认为学习是困难的——这令人震惊。" ~《沙丘》
</blockquote>

现在我们已经从现成的连续文本中抽出了 skipgram 训练数据集，来看看如何用它在"预测相邻词"这一任务上训练一个基本的神经语言模型。



<br />  

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/skipgram-language-model-training.png"/>
  <br />  
</div>

我们从数据集中的第一个样本开始。取出特征词，喂给未训练的模型，让它预测一个合适的相邻词。

<br />  


<div class="img-div" markdown="0">
  <image src="/images/word2vec/skipgram-language-model-training-2.png"/>
  <br />  
</div>

模型执行上面说的三个步骤，输出一个预测向量（为词表中的每个词分配一个概率）。由于模型尚未训练，它此刻的预测必然是错的。但这没关系。我们知道它本该猜哪个词——就是当前训练行的标签/输出单元格：

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/skipgram-language-model-training-3.png"/>
  <br />  
  '目标向量'是这样一种向量：目标词的概率为 1，其他所有词的概率为 0。
</div>

<br />  

模型错得有多远？把两个向量相减，得到一个误差向量：

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/skipgram-language-model-training-4.png"/>
  <br />  
</div>

<br />  

这个误差向量现在可以用来更新模型，使下次输入 <code class="plain_code mdc-text-green-500">not</code> 时，它更有可能猜出 <code class="plain_code mdc-text-pink-500">thou</code>。

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/skipgram-language-model-training-5.png"/>
  <br />  
</div>

<br />  

这样就完成了训练的第一步。我们继续对数据集中的下一个样本做同样的事，然后是下一个，直到处理完数据集里的全部样本。这就完成了一个训练*轮次（epoch）*。如此重复若干轮，我们就得到了训练好的模型，可以从里面提取嵌入矩阵，用于任何下游应用。

虽然这加深了我们对过程的理解，但这仍然不是 word2vec 实际的训练方式。我们还缺两个关键思想。

# 负采样

<blockquote class='subtle'>
“不了解穆阿迪布一生的死敌——哈克南人，就谈不上理解他；就像不知黑暗就想看见光明，不知伪妄就想窥见真理。那是不可能的。” ~《沙丘》
</blockquote>

回顾这个神经语言模型计算预测的三步：
<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/language-model-expensive.png"/>
  <br />  
</div>

<br />  

第三步在计算上非常昂贵——尤其是考虑到数据集里的每个训练样本我们都要做一次（轻松就是几千万次）。我们需要做点什么来提升性能。

一种思路是把目标拆成两步：

1. 生成高质量的词嵌入（先不管下一个词预测）。
2. 用这些高质量嵌入去训练一个语言模型（做下一个词预测）。

本篇聚焦于嵌入，所以只关注第 1 步。要用高性能的模型生成高质量嵌入，我们可以把模型的任务从"预测相邻词"：

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/predict-neighboring-word.png "/>
  <br />  
</div>

换成这样一个模型：它接收输入词和输出词，输出一个分数，表示两者是否是邻居（0 表示"不是邻居"，1 表示"是邻居"）。


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/are-the-words-neighbors.png "/>
  <br />  
</div>

这一简单的转换，把所需的模型从神经网络降为一个逻辑回归模型——计算因此大大简化、大大加速。

这个转换要求我们改变数据集的结构——标签现在是一列新列，取值为 0 或 1。目前它们全是 1，因为我们加入的都是邻居词。

<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/word2vec-training-dataset.png "/>
  <br />  
</div>

<br />  

现在可以以极快的速度来训练了——几分钟内就能处理上百万个样本。但有一个漏洞需要补上。如果我们所有的样本都是正样本（目标为 1），那就存在一个偷懒的模型次次都返回 1 的可能——准确率 100%，却什么都没学到，生成的嵌入全是垃圾。


<div class="img-div" markdown="0">
  <image src="/images/word2vec/word2vec-smartass-model.png "/>
  <br />  
</div>

为了解决这个问题，我们需要向数据集中引入*负样本（negative samples）*——即不是邻居的词。模型对这些样本必须返回 0。这才是一个值得模型努力去解决的挑战——而且依然能保持极快的速度。


<br />  

<div class="img-div" markdown="0">
  <image src="/images/word2vec/word2vec-negative-sampling.png "/>
  <br />  
  为数据集中的每个样本，我们添加<strong>负样本</strong>：同样的输入词，但标签为 0。
</div>

那输出词填什么？我们从词表中随机采样一些词。

<br />  


<div class="img-div" markdown="0">
  <image src="/images/word2vec/word2vec-negative-sampling-2.png "/>
  <br />  
</div>

这个思想来源于[噪声对比估计（Noise-contrastive estimation）](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) [pdf]。我们把真实信号（邻居词的正样本）与噪声（随机选取的、并非邻居的词）作对比。这在计算效率与统计效率之间取得了极佳的平衡。

# 带负采样的 Skipgram（SGNS）

到现在为止，我们已经讲完了 word2vec 的两个核心思想：它们合在一起，就叫"带负采样的 skipgram（skipgram with negative sampling）"。

<div class="img-div" markdown="0">
  <image src="/images/word2vec/skipgram-with-negative-sampling.png "/>
  <br />  
</div>



# Word2vec 训练过程

<blockquote class="subtle">
“机器无法预见对人类重要的每一个问题。这是离散的比特与连续的无限可分之间的区别。我们拥有后者；机器只能困在前者之中。” ~《沙丘神帝》
</blockquote>

现在 skipgram 与负采样这两个核心思想都立起来了，我们可以更仔细地看 word2vec 实际是怎么训练的。

训练开始之前，我们要对用来训练的文本做预处理。在这一步，我们确定词表的大小（记作 <code class="plain_code mdc-text-amber-700">vocab_size</code>，假设它是 10,000）以及哪些词属于词表。

训练阶段开始时，我们创建两个矩阵——<code class="plain_code mdc-text-green-500">Embedding</code>（输入词嵌入）矩阵和 <code class="plain_code mdc-text-purple-500">Context</code>（上下文词嵌入）矩阵。这两个矩阵都为词表中的每个词准备一个嵌入（所以 <code class="plain_code mdc-text-amber-700">vocab_size</code> 是它们的一个维度）。另一个维度是我们希望每个嵌入有多长（<code class="plain_code mdc-text-amber-900">embedding_size</code>——常用值是 300，但本文前面我们看过 50 维的例子）。

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/word2vec-embedding-context-matrix.png "/>
  <br />  
</div>

训练开始时，我们用随机值初始化这两个矩阵。然后开始训练。每个训练步，我们取一个正样本及其关联的负样本。就拿第一组来说：


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/word2vec-training-example.png "/>
  <br />  
</div>

现在我们手里有四个词：输入词 <code class="plain_code mdc-text-green-500">not</code>，以及输出/上下文词：<code class="plain_code mdc-text-purple-500">thou</code>（真正的邻居）、<code class="plain_code mdc-text-purple-500">aaron</code> 和 <code class="plain_code mdc-text-purple-500">taco</code>（负样本）。接着我们去查它们的嵌入——输入词查 <code class="plain_code mdc-text-green-500">Embedding</code> 矩阵，上下文词查 <code class="plain_code mdc-text-purple-500">Context</code> 矩阵（尽管两个矩阵里其实都有每个词的嵌入）。

<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/word2vec-lookup-embeddings.png "/>
  <br />  
</div>

然后，把输入词的嵌入分别与每个上下文词的嵌入做点积。每种情况都会得到一个数，这个数表示输入嵌入与上下文嵌入的相似度。


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/word2vec-training-dot-product.png "/>
  <br />  
</div>

现在需要想办法把这些分数变成看起来像概率的东西——要求全为正、取值在 0 到 1 之间。这正是 [sigmoid](https://jalammar.github.io/feedforward-neural-networks-visual-interactive/#sigmoid-visualization)（[logistic 函数](https://en.wikipedia.org/wiki/Logistic_function)）大显身手的地方。


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/word2vec-training-dot-product-sigmoid.png "/>
  <br />  
</div>

现在我们可以把 sigmoid 的输出当作模型对这些样本的预测。可以看到，无论 sigmoid 之前还是之后，<code class="plain_code mdc-text-purple-500">taco</code> 的分数都最高，<code class="plain_code mdc-text-purple-500">aaron</code> 都最低。

既然未训练的模型已经给出了预测，而我们手里又有真正的目标标签可以对照，那就来计算模型预测的误差。做法很简单：用目标标签减去 sigmoid 分数。


<div class="img-div-any-width" markdown="0">
  <image src="/images/word2vec/word2vec-training-error.png "/>
  <br />  
  <code class="plain_code mdc-text-yellow-800">error</code> = <code class="plain_code mdc-text-pink-400">target</code> - <code class="plain_code mdc-text-grey-900">sigmoid_scores</code>
</div>

<br />

接下来就是"机器学习"里"学习"的部分。我们可以用这个误差来调整 <code class="plain_code mdc-text-green-500">not</code>、<code class="plain_code mdc-text-purple-500">thou</code>、<code class="plain_code mdc-text-purple-500">aaron</code> 和 <code class="plain_code mdc-text-purple-500">taco</code> 的嵌入，使下次做同样计算时，结果更接近目标分数。

<div class="img-div" markdown="0">
  <image src="/images/word2vec/word2vec-training-update.png "/>
  <br />  
</div>

这一个训练步就结束了。结束时，本步涉及的词（<code class="plain_code mdc-text-green-500">not</code>、<code class="plain_code mdc-text-purple-500">thou</code>、<code class="plain_code mdc-text-purple-500">aaron</code> 和 <code class="plain_code mdc-text-purple-500">taco</code>）的嵌入都略有改善。接着进入下一步（下一个正样本及其关联的负样本），重复同样的过程。

<div class="img-div" markdown="0">
  <image src="/images/word2vec/word2vec-training-example-2.png "/>
  <br />  
</div>

随着我们把整个数据集反复遍历多轮，嵌入不断被改进。然后我们可以停止训练，丢弃 <code class="plain_code mdc-text-purple-500">Context</code> 矩阵，把 <code class="plain_code mdc-text-green-500">Embeddings</code> 矩阵当作预训练嵌入，交给下一个任务使用。

# 窗口大小与负样本数量

word2vec 训练过程中的两个关键超参数是窗口大小和负样本数量。

<div class="img-div" markdown="0">
  <image src="/images/word2vec/word2vec-window-size.png "/>
  <br />  
</div>

不同的任务适合不同的窗口大小。一个[经验法则](https://youtu.be/tAxrlAVw-Tk?t=648)是：较小的窗口（2-15）产生的嵌入中，两个嵌入相似度高意味着这两个词*可互换*（注意：如果只看上下文，反义词往往是可互换的——比如 *good* 和 *bad* 经常出现在相似的语境中）。较大的窗口（15-50 甚至更大）产生的嵌入中，相似度更多指示的是词之间的*关联性（relatedness）*。实践中，你往往需要提供[标注](https://youtu.be/ao52o9l6KGw?t=287)来引导嵌入过程，使相似度对你的任务有用。Gensim 默认的窗口大小是 5（输入词前五个词和后五个词，外加输入词本身）。


<div class="img-div" markdown="0">
  <image src="/images/word2vec/word2vec-negative-samples.png "/>
  <br />  
</div>

负样本数量是训练过程中的另一个因素。原论文建议 5-20 个负样本是比较合适的数量，并指出当数据集足够大时，2-5 个似乎就够了。Gensim 的默认值是 5 个负样本。

# 结语

<blockquote class="subtle">
“如果它超出了你的标尺，那么你所进行的才是智能，而非自动化。” ~《沙丘神帝》
</blockquote>

希望你现在对词嵌入和 word2vec 算法有了整体的认识。也希望今后当你读到提到"带负采样的 skipgram"（SGNS）的论文（比如本文开头提到的那些推荐系统论文）时，对这些概念能有更清楚的把握。一如既往，欢迎任何反馈 <a href="https://twitter.com/JayAlammar">@JayAlammar</a>。

# 参考文献与延伸阅读
* [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) [pdf]
* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) [pdf]
* [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) [pdf]
* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)（Dan Jurafsky 与 James H. Martin 著）是 NLP 领域的权威资料。第 6 章讲解 Word2vec。
* [Neural Network Methods in Natural Language Processing](https://www.amazon.com/Language-Processing-Synthesis-Lectures-Technologies/dp/1627052984)（[Yoav Goldberg](https://twitter.com/yoavgo) 著）是了解神经 NLP 主题的绝佳读物。
* [Chris McCormick](http://mccormickml.com/) 写过多篇关于 Word2vec 的精彩博客文章。他刚发布了 [The Inner Workings of word2vec](https://www.preview.nearist.ai/paid-ebook-and-tutorial)，一本聚焦 word2vec 内部机制的电子书。
* 想读代码？有两个选择：
  * Gensim 的 word2vec [python 实现](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py)
  * Mikolov 最初的 [C 语言实现](https://github.com/tmikolov/word2vec/blob/master/word2vec.c)——更好的选择是 Chris McCormick 整理的[带详细注释的版本](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c)。
* [Evaluating distributional models of compositional semantics](http://sro.sussex.ac.uk/id/eprint/61062/1/Batchkarov,%20Miroslav%20Manov.pdf)
* [On word embeddings](http://ruder.io/word-embeddings-1/index.html)，[第二篇](http://ruder.io/word-embeddings-softmax/)
* [《沙丘》（Dune）](https://www.amazon.com/Dune-Frank-Herbert/dp/0441172717/)


<br />
<br />
<br />
<br />
