**5 月 25 日更新：** 更换了新的图示（RNN 动画、词嵌入图），加入颜色编码，并把最后的 attention 示例讲得更细。

**提示：** 下面的动画都是视频。触摸或用鼠标悬停在它们上面会出现播放控制条，必要时可以暂停。

Sequence-to-sequence（序列到序列，seq2seq）模型是一类深度学习模型，在机器翻译、文本摘要、图像描述等任务上取得了很大成功。Google 翻译从 2016 年底开始在生产环境中[使用](https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/)这样的模型。这类模型在两篇开创性论文中有详细说明（[Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)、[Cho et al., 2014](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)）。

不过我发现，要把模型理解到能动手实现的程度，需要把一串层层递进的概念逐一理清。而其中不少想法如果用视觉化的方式表达，会更容易理解——这正是本文想做的。阅读本文需要你具备一些深度学习基础。希望它能成为你阅读上述论文（以及后文链接的 attention 论文）时的有用伴侣。


序列到序列模型接收一序列元素（词、字母、图像特征等……）并输出另一个序列元素。训练好的模型工作起来是这样的：
<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq_1.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>





<br />

在神经机器翻译中，"序列"就是一串依次处理的词；输出同样是一串词：

<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq_2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## 看看引擎盖底下

模型内部由一个 <span class="encoder">编码器（encoder）</span> 和一个 <span class="decoder">解码器（decoder）</span> 组成。

<span class="encoder">编码器</span>处理输入序列中的每个元素，把它捕获的信息汇编成一个向量（称为<span class="context">上下文向量（context）</span>）。处理完整个输入序列后，<span class="encoder">编码器</span>把<span class="context">上下文向量</span>交给 <span class="decoder">解码器</span>，后者开始逐条生成输出序列。

<video width="100%" height="auto" loop autoplay  controls>
  <source src="/images/seq2seq_3.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>



<br />

机器翻译的情形也是如此。

<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq_4.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


对机器翻译而言，<span class="context">上下文向量</span>就是一个向量（本质上是一串数字）。<span class="encoder">编码器</span>和<span class="decoder">解码器</span>通常都是循环神经网络（RNN，想了解 RNN 入门，推荐 Luis Serrano 的 [A friendly introduction to Recurrent Neural Networks](https://www.youtube.com/watch?v=UNmqTiOnRfg)）。

<div class="img-div" markdown="0">
    <img src="/images/context.png" />
    <span class="context">上下文向量</span>是一个浮点数向量。本文后面会用颜色可视化向量：数值越大的格子颜色越亮。
</div>

构建模型时你可以自行设定<span class="context">上下文向量</span>的大小——它本质上就是<span class="encoder">编码器</span> RNN 的隐藏单元数量。这些可视化里画的是 4 维向量，但真实应用中<span class="context">上下文向量</span>通常是 256、512 或 1024 维。

<br />

RNN 的设计决定了它在每个时间步接收两个输入：一个输入（对编码器来说，就是输入句子中的一个词）和一个隐藏状态（hidden state）。而词必须先表示成向量。把词变成向量靠的是"词嵌入（word embedding）"这一类[方法](https://machinelearningmastery.com/what-are-word-embeddings/)——它们把词变成能捕捉大量词义/语义信息的向量空间（例如 [king - man + woman = queen](http://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html)）。

<br />

<div class="img-div" markdown="0">
    <img src="/images/embedding.png" />
    处理输入词之前要先把它们变成向量，这一步用<a href="https://en.wikipedia.org/wiki/Word_embedding">词嵌入</a>算法完成。可以用<a href="http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/">预训练嵌入</a>，也可以在自己的数据集上训练嵌入。嵌入向量常用 200 或 300 维；为简单起见，图中只画了 4 维。
</div>

主要向量/张量都介绍完了，接下来回顾 RNN 的运作机制，并建立一套描述这些模型的视觉语言：

<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/RNN_1.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<br />

下一个 RNN 步骤接收第二个输入向量和隐藏状态 #1，产出该时间步的输出。后文我们会用类似的动画来描述神经机器翻译模型内部的向量。

<br />

在接下来的可视化里，<span class="encoder">编码器</span>或<span class="decoder">解码器</span>的每次"跳动"，就是该 RNN 处理输入并生成当前时间步输出的过程。由于<span class="encoder">编码器</span>和<span class="decoder">解码器</span>都是 RNN，它们在每一步做完处理后，都会根据当前输入和之前见过的输入更新自己的<span class="context">隐藏状态</span>。

我们来看<span class="encoder">编码器</span>的<span class="context">隐藏状态</span>。注意最后一个<span class="context">隐藏状态</span>正是我们传给<span class="decoder">解码器</span>的那个<span class="context">上下文向量</span>。


<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq_5.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


<br />

<span class="decoder">解码器</span>同样维护着一个逐时间步传递的<span class="decoder">隐藏状态</span>。这张图里没把它画出来，是因为我们现在只关心模型的主体结构。


再来看另一种可视化序列到序列模型的方式。这个动画能帮你更容易看懂后面描述这些模型的静态图。这就是所谓的"展开（unrolled）"视角：不再画同一个<span class="decoder">解码器</span>，而是为每个时间步画一份拷贝，从而能观察每个时间步的输入和输出。

<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq_6.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<br />



# 现在该关注 Attention 了

<span class="context">上下文向量</span>成了这类模型的瓶颈：它让模型处理长句变得困难。[Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) 和 [Luong et al., 2015](https://arxiv.org/abs/1508.04025) 提出了解决方案。这些论文引入并完善了名为 "Attention（注意力）" 的技术，大幅提升了机器翻译系统的质量。Attention 让模型能够在需要时聚焦于输入序列的相关部分。


<img src="/images/attention.png" />

<div class="img-div" markdown="0">
    在第 7 个时间步，attention 机制让 <span class="decoder">解码器</span> 在生成英文译文之前先聚焦于词 "étudiant"（法语的"学生"）。这种放大输入序列相关部分信号的能力，使 attention 模型的表现优于不带 attention 的模型。
</div>

<br />

我们继续在这个抽象层次上观察 attention 模型。attention 模型与经典序列到序列模型有两个主要区别：

首先，<span class="encoder">编码器</span>要传给<span class="decoder">解码器</span>多得多的数据：不再只传编码阶段最后的隐藏状态，而是把编码器的*全部*<span class="context">隐藏状态</span>都传给<span class="decoder">解码器</span>：

<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq_7.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


<br />

其次，attention <span class="decoder">解码器</span>在产出输出之前会多做一步。为了聚焦于与当前解码时间步相关的输入部分，<span class="decoder">解码器</span>会：

 1. 观察收到的这组<span class="context">编码器隐藏状态</span>——每个<span class="context">编码器隐藏状态</span>都与输入句子中的某个词最相关
 1. 给每个<span class="context">隐藏状态</span>打分（打分怎么做的，先按下不表）
 1. 把每个<span class="context">隐藏状态</span>乘以它经 softmax 归一化后的分数——高分的<span class="context">隐藏状态</span>被放大，低分的被淹没


<video width="100%" height="auto" loop autoplay controls>
   <source src="/images/attention_process.mp4" type="video/mp4">
   Your browser does not support the video tag.
</video>

<br />
<br />

这个打分动作在<span class="decoder">解码器</span>的每个时间步都要做一遍。

下面用一张完整的可视化把所有环节串起来，看 attention 流程如何运转：

1. attention 解码器 RNN 接收 <span class="embedding">&lt;END&gt;</span> token 的嵌入和一个<span class="decoder">初始解码器隐藏状态</span>。
1. RNN 处理输入，产出一个输出和<span class="decoder">新的隐藏状态</span>向量（<span class="decoder">h</span><span class="step_no">4</span>）。这个输出被丢弃。
1. Attention 步骤：用<span class="context">编码器隐藏状态</span>和<span class="decoder">h</span><span class="step_no">4</span>向量计算本时间步的上下文向量（<span class="step_no">C</span><span class="decoder">4</span>）。
1. 把 <span class="decoder">h</span><span class="step_no">4</span> 和 <span class="step_no">C</span><span class="decoder">4</span> 拼接成一个向量。
1. 把这个向量送进<span class="ffnn">前馈神经网络</span>（与整个模型联合训练）。
1. <span class="logits_output">前馈网络的输出</span>即为本时间步的输出词。
1. 对后续时间步重复以上过程。

<video width="100%" height="auto" loop autoplay controls>
   <source src="/images/attention_tensor_dance.mp4" type="video/mp4">
   Your browser does not support the video tag.
</video>

<br />
<br />

还可以从另一个角度看每个解码步骤在关注输入句子的哪个部分：


<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq_9.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

注意，模型并不是机械地把输出的第一个词对到输入的第一个词。它是从训练阶段*学会*了这个语言对（我们的例子是法语和英语）中词该如何对齐。上面列出的 attention 论文里有一个展示这种机制能有多精确的例子：


<div class="img-div" markdown="0">
<img src="/images/attention_sentence.png" />
    可以看到，模型在输出 "European Economic Area" 时的 attention 完全正确：这些词在法语里的顺序与英语相反（"européenne économique zone"），而句子里其他所有词的顺序则与英语一致。
</div>


<br />

如果你觉得已经准备好上手实现了，推荐 TensorFlow 的 [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)。

---

<br />

希望本文对你有帮助。这些图示是我在 Udacity [自然语言处理 Nanodegree 课程](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892)中 attention 一节课的早期版本。课程里讲得更深，包括各种应用，以及更新的 attention 方法——比如 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 里的 Transformer 模型。

可以看看 [NLP Nanodegree 课程](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892)的预告片：

<div class="videoWrapper">
<iframe width="560" height="349"  src="https://www.youtube.com/embed/FggN0VtWYTk" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
</div>

我还在 Udacity 的[机器学习 Nanodegree 课程](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t)里做了若干节课，主题包括无监督学习，以及一个用协同过滤做电影推荐的 jupyter notebook。

欢迎任何反馈，请在 [@JayAlammmar](https://twitter.com/JayAlammar) 联系我。
