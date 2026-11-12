<table>
 <tr>
  <td style="vertical-align: middle; min-width:250px">
<a href="https://www.llm-book.com"><img src="https://github.com/user-attachments/assets/a471dfff-00cc-4cb4-8df5-123e195bcc71" width="200" /></a>
  </td>
  <td style="vertical-align: middle"><b>更新：</b>本文现在已经成为了一本书！欢迎阅读 <a href="https://llm-book.com">LLM-book.com</a>，其中的（第 3 章）是本文的更新扩充版本，讲述了最新的 Transformer 模型，以及自原始 Transformer 问世以来这七年里它们是如何演进的（例如 Multi-Query Attention 与 RoPE 位置编码）。
  </td>
 </tr>
</table>

在[上一篇文章中，我们探讨了 Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)——这是现代深度学习模型中一种无处不在的方法。Attention 这一概念有助于提升神经机器翻译应用的性能。在这篇文章里，我们将研究 **Transformer**——一个利用 attention 来提升模型训练速度的模型。Transformer 在某些任务上的表现超过了 Google 神经机器翻译模型。但它最大的优势，其实在于天然适合并行化计算。事实上，Google Cloud 正是推荐把 Transformer 作为使用其 [Cloud TPU](https://cloud.google.com/tpu/) 服务的参考模型。那么，就让我们把这个模型拆开，看看它是如何工作的。

Transformer 由论文 [Attention is All You Need](https://arxiv.org/abs/1706.03762) 提出。它的 TensorFlow 实现包含在 [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) 软件包中。哈佛大学的 NLP 团队编写了一份[用 PyTorch 实现逐段注释原论文的指南](http://nlp.seas.harvard.edu/2018/04/03/attention.html)。在这篇文章里，我会尝试把事情讲得尽可能简化，并逐一介绍各个概念，希望能让没有深入背景知识的读者也容易理解。

**2025 年更新**：我们制作了一门<a href="https://bit.ly/4aRnn7Z">免费短课程</a>，用动画把本文的内容更新到了最新进展：

 <div style="text-align:center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/k1ILy23t89E?si=UEqgMCVxYfMBkEtj" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"  style="
 width: 100%;
 max-width: 560px;"
allowfullscreen></iframe>
</div>
## 宏观视角

我们先把模型当作一个黑盒来看待。如果是在机器翻译应用中，它会输入一种语言的一个句子，并输出它在另一种语言中的翻译。


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/the_transformer_3.png" />
</div>



打开这个"擎天柱"般的东西，我们会看到一个编码器组件、一个解码器组件，以及两者之间的连接。

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/The_transformer_encoders_decoders.png" />
</div>

编码器组件是编码器的堆叠（原论文中堆叠了六个——数字六并没有什么神奇之处，你完全可以尝试其他配置）。解码器组件则是相同数量的解码器的堆叠。


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/The_transformer_encoder_decoder_stack.png" />
</div>

所有编码器的结构都完全相同（但它们并不共享权重）。每个编码器可以细分为两个子层：


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/Transformer_encoder.png" />
</div>

编码器的输入首先流经一个 self-attention（自注意力）层——该层的作用是，在编码器对某个特定词进行编码时，帮助它观察输入句子中的其他词。关于自注意力，我们在文章后面会详细讲解。

自注意力层的输出会被送入一个前馈神经网络。同一个前馈网络会被独立地、逐一地应用于每个位置。

解码器同时拥有这两个层，但在它们之间还多了一个 attention 层，帮助解码器关注输入句子中相关的部分（这与 [seq2seq 模型](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)中 attention 的作用类似）。

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/Transformer_decoder.png" />
</div>


## 引入张量

现在我们已经看到了模型的主要组件，接下来看看各种向量/张量，以及它们如何在这些组件之间流动，把训练好的模型的输入转换为输出。

和一般的 NLP 应用一样，我们首先用[嵌入算法（embedding algorithm）](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)把每个输入词转换为一个向量。

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/embeddings.png" />
  <br />
  每个词都会被嵌入为一个 512 维的向量。我们用这些简单的方框来表示这些向量。
</div>

嵌入只在最底层的编码器中发生。所有编码器共享的一个抽象是：它们接收的都是一个列表，列表中每个元素都是 512 维的向量——在最底层的编码器中，这个列表是词的嵌入；而在其他编码器中，它则是下方编码器的输出。这个列表的长度是一个我们可以设置的超参数——基本上它就是训练数据集中最长句子的长度。

对输入序列中的词做完嵌入之后，每个词都会流经编码器的两个层中的每一个。



<div class="img-div-any-width" markdown="0">
  <img src="/images/t/encoder_with_tensors.png" />
  <br />

</div>

从这里我们开始看到 Transformer 的一个关键特性：每个位置上的词在编码器中都沿着自己的路径流动。这些路径之间在自注意力层存在依赖关系。而前馈层则没有这种依赖，因此在流经前馈层时，各条路径可以并行执行。

接下来，我们换一个更短的句子作为例子，看看编码器的每个子层里发生了什么。

## 现在开始编码！

正如前面提到的，编码器接收一个向量列表作为输入。它处理这个列表的方式是：把这些向量传入"自注意力"层，再传入前馈神经网络，然后把输出向上发送给下一个编码器。


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/encoder_with_tensors_2.png" />
  <br />
  每个位置上的词都会经过一个自注意力过程。然后，它们各自经过一个前馈神经网络——是同一个网络，每个向量分别独立地流过它。
</div>

## 自注意力概览

别被我随手抛出的"自注意力"这个词唬住，好像人人都该熟悉它一样。我个人也是在读到 *Attention is All You Need* 这篇论文之前从未接触过这个概念。让我们来提炼一下它的原理。

假设下面这个句子是我们想要翻译的输入句子：

"```The animal didn't cross the street because it was too tired```"

这句话里的 "it" 指的是什么？是指街道（street）还是指动物（animal）？对人来说这是个简单的问题，但对算法而言就没这么简单了。

当模型处理单词 "it" 时，自注意力机制使它能把 "it" 与 "animal" 关联起来。

当模型处理每个词（输入序列中的每个位置）时，自注意力允许它查看输入序列中的其他位置，以寻找有助于为该词生成更好编码的线索。

如果你熟悉 RNN，可以想想隐藏状态（hidden state）是如何让 RNN 把它对之前处理过的词/向量的表示与当前正在处理的表示结合起来的。自注意力正是 Transformer 用来把其他相关词的"理解"融入当前正在处理的词的方法。


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self-attention_visualization.png" />
  <br />
  当我们在编码器 #5（堆叠中最顶层的编码器）中对单词 "it" 进行编码时，attention 机制的一部分注意力聚焦在 "The Animal" 上，并把它的表示的一部分融入到了 "it" 的编码中。
</div>


一定要去看看 [Tensor2Tensor notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)，你可以在那里加载一个 Transformer 模型，并用这个交互式可视化来研究它。



## 自注意力详解

我们先看看如何用向量计算自注意力，然后再看它实际是如何实现的——使用矩阵。

计算自注意力的**第一步**，是为编码器输入中的每个向量（本例中即每个词的嵌入）创建三个向量。也就是说，对每个词，我们都创建一个 Query 向量、一个 Key 向量和一个 Value 向量。这些向量是把嵌入乘以三个在训练过程中学到的矩阵得到的。

注意，这些新向量的维度比嵌入向量小：它们的维度是 64，而嵌入以及编码器的输入/输出向量维度是 512。它们并非非得小不可——这是一种架构上的选择，目的是让多头注意力（multiheaded attention）的计算量（大致）保持不变。


<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self_attention_vectors.png" />
  <br />
  把 <span class="encoder">x1</span> 乘以 <span class="decoder">WQ</span> 权重矩阵得到 <span class="decoder">q1</span>，即与该词关联的"query"向量。最终，我们为输入句子中的每个词都创建了一个"query"、一个"key"和一个"value"投影。
</div>

<br />
<br />

"query"、"key"、"value" 向量到底是什么？
<br />
<br />
它们是为了计算和方便理解 attention 而引入的抽象。等你读完下文 attention 的计算方式后，基本就能了解这些向量各自扮演什么角色了。

计算自注意力的**第二步**是计算分数（score）。假设我们要计算这个例子中第一个词 "Thinking" 的自注意力。我们需要对输入句子中的每个词相对于这个词打分。分数决定了：在对某个位置上的词进行编码时，我们要在多大程度上关注输入句子的其他部分。

分数的计算方式是：取 <span class="decoder">query 向量</span>与我们正在打分的对应词的 <span class="context">key 向量</span>的点积。所以，如果我们在处理位置 <span class="encoder">#1</span> 上那个词的自注意力，那么第一个分数就是 <span class="decoder">q1</span> 与 <span class="context">k1</span> 的点积，第二个分数则是 <span class="decoder">q1</span> 与 <span class="context">k2</span> 的点积。


<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self_attention_score.png" />
  <br />

</div>

<br />


**第三和第四步**是把分数除以 8（论文中所用 key 向量维度 64 的平方根。这样做能得到更稳定的梯度。这里也可以用其他值，但这是默认做法），然后把结果通过一个 softmax 运算。softmax 会把分数归一化，使它们全为正数且相加等于 1。


<br />


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention_softmax.png" />
  <br />

</div>

这个 softmax 分数决定了每个词在该位置会被表达多少。显然，本位置的那个词会得到最高的 softmax 分数，但有时关注与当前词相关的其他词也是很有用的。

<br />


**第五步**是把每个 value 向量乘以它的 softmax 分数（为下一步求和做准备）。这里的直觉是：保留我们想要关注的词的 value 值，同时"淹没"无关的词（比如把它们乘上 0.001 这样的小数）。

**第六步**是对加权后的 value 向量求和。这就产生了自注意力层在该位置（即第一个词）的输出。

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention-output.png" />
  <br />
</div>

自注意力的计算到此结束。得到的向量就是我们可以送入前馈神经网络的东西。不过在实现中，为了更快的计算，这一整套运算都是以矩阵形式完成的。既然我们已经从词级角度理解了计算直觉，现在来看看矩阵形式。


## 自注意力的矩阵计算

**第一步**是计算 Query、Key、Value 矩阵。做法是把嵌入打包成矩阵 <span class="encoder">X</span>，再乘以我们训练得到的权重矩阵（<span class="decoder">WQ</span>、<span class="context">WK</span>、<span class="step_no">WV</span>）。

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention-matrix-calculation.png" />
  <br />
  <span class="encoder">X</span> 矩阵的每一行对应输入句子中的一个词。我们再次看到嵌入向量（512 维，即图中的 4 个方框）与 q/k/v 向量（64 维，即图中的 3 个方框）在尺寸上的差别。
</div>

<br />

**最后**，既然我们处理的都是矩阵，就可以把第二步到第六步合并成一个公式，来计算自注意力层的输出。

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/self-attention-matrix-calculation-2.png" />
  <br />
  矩阵形式的自注意力计算
</div>

<br />


<br />

## 多头巨兽

论文还通过引入"多头"（multi-headed）注意力机制对自注意力层做了进一步改进。这从两个方面提升了 attention 层的性能：

 1. 它扩展了模型关注不同位置的能力。是的，在上面的例子中，z1 里包含了其他每个编码的一点点信息，但它可能被词本身所主导。如果我们在翻译 "The animal didn't cross the street because it was too tired" 这样的句子，知道 "it" 指的是哪个词会很有帮助。

 1. 它给 attention 层提供了多个"表示子空间"（representation subspaces）。如下文所见，多头注意力不止拥有一组，而是多组 Query/Key/Value 权重矩阵（Transformer 使用 8 个注意力头，因此每个编码器/解码器都有 8 组）。这些集合全部随机初始化，训练之后，每一组都会把输入的嵌入（或来自下层编码器/解码器的向量）投影到一个不同的表示子空间。


 <div class="img-div-any-width" markdown="0">
   <img src="/images/t/transformer_attention_heads_qkv.png" />
   <br />
   在多头注意力中，我们为每个注意力头维护独立的 Q/K/V 权重矩阵，从而得到不同的 Q/K/V 矩阵。和之前一样，我们把 X 乘以 WQ/WK/WV 矩阵来得到 Q/K/V 矩阵。
 </div>

 <br />
如果把我们前面描述的那个自注意力计算用 8 组不同的权重矩阵各做一遍，就会得到 8 个不同的 Z 矩阵。

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_attention_heads_z.png" />
  <br />

</div>

 <br />

这带来了一个小问题。前馈层并不期望收到 8 个矩阵——它期望的是单个矩阵（每个词一个向量）。所以我们需要一种办法把这 8 个矩阵压缩成一个矩阵。

该怎么做呢？把这些矩阵拼接起来，再乘以一个额外的权重矩阵 WO。


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_attention_heads_weight_matrix_o.png" />
  <br />

</div>

多头自注意力基本就是这些了。我知道矩阵确实有点多。我试着把它们全部画在一张图里，方便大家一次性看清。

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_multi-headed_self-attention-recap.png" />
  <br />

</div>

<br />

现在我们已经了解了注意力头，让我们回到之前的例子，看看在对例句中的单词 "it" 编码时，不同的注意力头分别在哪里聚焦：

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self-attention_visualization_2.png" />
  <br />
  在对单词 "it" 编码时，一个注意力头最关注 "the animal"，另一个则关注 "tired"——从某种意义上说，模型对 "it" 这个词的表示中融入了 "animal" 和 "tired" 两者表示的一部分。
</div>

<br />

不过，如果要把所有注意力头都画进图里，事情就变得难以解读了：

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_self-attention_visualization_3.png" />
  <br />
</div>



## 用位置编码表示序列顺序

到目前为止我们描述的模型还缺少一点东西：一种体现输入序列中词序的方法。

为了解决这个问题，Transformer 向每个输入嵌入添加一个向量。这些向量遵循模型学习到的特定模式，帮助模型确定每个词的位置，或序列中不同词之间的距离。这里的直觉是：给嵌入加上这些数值后，当它们被投影为 Q/K/V 向量并做点积注意力时，嵌入向量之间就具有了有意义的距离。

<br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_positional_encoding_vectors.png" />
  <br />
  为了让模型感知词的顺序，我们加上位置编码向量——它们的数值遵循特定的模式。
</div>
  <br />


如果我们假设嵌入的维度是 4，实际的位置编码看起来会是这样：

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_positional_encoding_example.png" />
  <br />
  嵌入大小为 4 的玩具示例下，位置编码的真实样例
</div>

  <br />

这种模式可能长什么样？

在下图中，每一行对应一个向量的位置编码。所以第一行就是我们会加到输入序列第一个词嵌入上的向量。每一行包含 512 个数值——每个值都在 1 和 -1 之间。我们用颜色标记了它们，以便让模式可见。



<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_positional_encoding_large_example.png" />
  <br />
  20 个词（行）、嵌入大小为 512（列）的位置编码真实样例。你会注意到图样在中间一分为二：这是因为左半部分的数值由一个函数（使用正弦）生成，右半部分由另一个函数（使用余弦）生成，再把它们拼接起来形成各个位置编码向量。
</div>


位置编码的公式在论文中给出（3.5 节）。你可以在 [```get_timing_signal_1d()```](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py) 中看到生成位置编码的代码。位置编码并非只有这一种方法。不过这种方法的优点是能够外推到未见过的序列长度（例如，当训练好的模型被要求翻译一个比训练集中任何句子都长的句子时）。

**2020 年 7 月更新：**
上面展示的位置编码来自 Transformer 的 Tensor2Tensor 实现。论文中给出的方法略有不同：它不是直接把两个信号拼接，而是把它们交错编织（interweave）在一起。下图展示了那样的效果。[这是生成它的代码](https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb)：

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/attention-is-all-you-need-positional-encoding.png" />
  <br />
</div>

## 残差连接

在继续之前，编码器架构中还有一个细节必须提到：编码器中的每个子层（自注意力、前馈网络）外部都有一条残差连接（residual connection），其后还跟一个[层归一化（layer-normalization）](https://arxiv.org/abs/1607.06450)步骤。

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_resideual_layer_norm.png" />
  <br />
</div>

如果把与自注意力相关的向量以及层归一化操作可视化出来，将会是这样：

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_resideual_layer_norm_2.png" />
  <br />
</div>

解码器的子层同样如此。如果把我们考虑的是一个由 2 层编码器和 2 层解码器堆叠而成的 Transformer，它看起来会是这样：

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_resideual_layer_norm_3.png" />
  <br />
</div>


## 解码器一侧

既然编码器一侧的大部分概念我们都讲过了，解码器各组件的工作原理基本也就清楚了。但让我们看看它们如何协同工作。

编码器先处理输入序列。最顶层编码器的输出会被转换为一组 attention 向量 K 和 V。它们将被每个解码器的"encoder-decoder attention"层使用，帮助解码器关注输入序列中合适的位置：


<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_decoding_1.gif" />
  <br />
  编码阶段结束后，我们开始解码阶段。解码阶段的每一步输出序列中的一个元素（本例中即英语翻译句子的一个词）。
</div>

下面的步骤会不断重复这一过程，直到遇到特殊的 \<end of sentence\> 符号，表示 Transformer 解码器已完成输出。每一步的输出会被下一个时间步送入最底层的解码器，各解码器就像编码器那样把自己的解码结果逐层向上传递。并且和编码器输入一样，我们也会对这些解码器输入做嵌入并加上位置编码，以表示每个词的位置。

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_decoding_2.gif" />
  <br />

</div>

解码器中的自注意力层的工作方式与编码器中的略有不同：

在解码器中，自注意力层只允许关注输出序列中较早的位置。做法是在自注意力计算的 softmax 步骤之前，把未来的位置掩码掉（将它们设为 ```-inf```）。

"Encoder-Decoder Attention"层的工作原理与多头自注意力完全一样，只是它的 Queries 矩阵来自下面一层的输出，而 Keys 和 Values 矩阵则来自编码器堆叠的输出。

## 最后的 Linear 层与 Softmax 层

解码器堆叠输出的是一个浮点数向量。我们如何把它变成一个词？这就是最后的 Linear 层及其后续 Softmax 层的任务。

Linear 层是一个简单的全连接神经网络，它把解码器堆叠产生的向量投影到一个大得多的向量中，称为 logits 向量。

假设我们的模型从训练数据集中学到了 10,000 个独特的英语单词（模型的"输出词表"）。这样 logits 向量就会有 10,000 个格子——每个格子对应一个词的得分。这就是我们对 Linear 层之后模型输出的解读方式。

随后 softmax 层把这些分数转换成概率（全为正数，总和为 1.0）。概率最高的格子被选中，与其对应的词就作为该时间步的输出被产生出来。

  <br />

<div class="img-div-any-width" markdown="0">
  <img src="/images/t/transformer_decoder_output_softmax.png" />
  <br />
  这张图从底部开始，即解码器堆叠输出的向量，然后它被转换成一个输出词。
</div>

  <br />

## 训练过程回顾

既然我们已经讲完了训练好的 Transformer 整个前向传播过程，再简单看看模型训练过程的直觉会很有帮助。

在训练时，未训练的模型同样要走完全一样的前向传播。但由于我们是在带标注的数据集上训练它，我们可以把它的输出与实际的正确输出进行比较。

 为了便于可视化，假设我们的输出词表只包含六个词（"a"、"am"、"i"、"thanks"、"student" 和 "\<eos\>"，即 'end of sentence' 的缩写）。

 <div class="img-div" markdown="0">
   <img src="/images/t/vocabulary.png" />
   <br />
   模型的输出词表是在开始训练之前的预处理阶段创建的。
 </div>

定义好输出词表之后，我们可以用一个宽度相同的向量来表示词表中的每个词，这就是所谓的一词编码（one-hot encoding）。例如，我们可以用下面的向量来表示单词 "am"：

<div class="img-div" markdown="0">
  <img src="/images/t/one-hot-vocabulary-example.png" />
  <br />
  示例：输出词表的一词编码
</div>

完成这段回顾之后，我们来讨论模型的损失函数——训练阶段我们要优化的指标，以期得到一个训练完成、 hopefully 精确得令人惊叹的模型。

## 损失函数

假设我们正在训练模型。这是训练阶段的第一步，我们在一个简单的例子上训练它——把 "merci" 翻译成 "thanks"。

这意味着我们希望输出是一个指示单词 "thanks" 的概率分布。但由于模型尚未经过训练，这目前还不太可能。

<div class="img-div" markdown="0">
  <img src="/images/t/transformer_logits_output_and_label.png" />
  <br />
  由于模型的全部参数（权重）都是随机初始化的，这个（未训练的）模型会为每个格子/词产生一个取值随机的概率分布。我们可以把它与实际输出比较，然后用反向传播调整模型的所有权重，使输出更接近期望输出。
</div>

<br />

如何比较两个概率分布？我们只需把两者相减即可。更多细节请参考 [cross-entropy（交叉熵）](https://colah.github.io/posts/2015-09-Visual-Information/)和 [Kullback–Leibler divergence（KL 散度）](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)。

但请注意这是个过度简化的例子。更现实的情况是，我们会使用比单个词更长的句子。例如——输入："je suis étudiant"，期望输出："i am a student"。这真正的含义是：我们希望模型依次输出满足以下条件的概率分布：

 * 每个概率分布用一个宽度为 vocab_size 的向量表示（我们的玩具示例中是 6，但现实中会是 30,000 或 50,000 这样的数字）
 * 第一个概率分布在单词 "i" 对应的格子上具有最高概率
 * 第二个概率分布在单词 "am" 对应的格子上具有最高概率
 * 依此类推，直到第五个输出分布指示 '```<end of sentence>```' 符号——它在 10,000 元素的词表中同样有自己的格子。


 <div class="img-div" markdown="0">
   <img src="/images/t/output_target_probability_distributions.png" />
   <br />
   对于这一个句子样本，我们将在训练样本中对模型输出的目标概率分布。
 </div>

<br />

在足够大的数据集上训练足够长的时间之后，我们希望模型产生的概率分布会是这样的：

  <div class="img-div" markdown="0">
    <img src="/images/t/output_trained_model_probability_distributions.png" />
    <br />
    希望训练之后，模型能输出我们期望的正确翻译。当然，如果这个短语本身就在训练数据集中，那就说明不了什么（参见：<a href="https://www.youtube.com/watch?v=TIgfjmp-4BA">交叉验证</a>）。注意，每个位置都会分到一点点概率，即使它不太可能作为该时间步的输出——这是 softmax 非常有用的一个性质，有助于训练过程。
</div>


现在，由于模型是一个词一个词地输出，我们可以假设模型是在从那个概率分布中选择概率最高的词，而丢弃其余部分。这是一种做法（称为贪心解码，greedy decoding）。另一种做法是保留前两名（比如 'I' 和 'a'），然后在下一步，把模型跑两遍：一遍假设第一个输出位置是 'I'，另一遍假设第一个输出位置是 'a'，综合考虑位置 #1 和 #2，错误更少的那个版本被保留。我们对位置 #2、#3……依此类推。这种方法称为"beam search（束搜索）"，在我们的例子中，beam_size 为 2（意味着任何时刻都在内存中保留两个未完成的候选翻译），top_beams 也为 2（意味着最终返回两个翻译）。这两个都是可以尝试调整的超参数。



## 大胆去 Transform 吧

希望这篇文章能成为你了解 Transformer 主要概念的一个有用的起点。如果你想进一步深入，我建议接下来的步骤是：

* 阅读 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 论文、Transformer 博客文章（[Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)），以及 [Tensor2Tensor 公告](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html)。
* 观看 [Łukasz Kaiser 的演讲](https://www.youtube.com/watch?v=rBCqOTEfxvg)，他会带你逐一过一遍模型及其细节
* 玩一玩 [Tensor2Tensor 仓库提供的 Jupyter Notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
* 探索 [Tensor2Tensor 仓库](https://github.com/tensorflow/tensor2tensor)。

后续相关工作：

* [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059)
* [One Model To Learn Them All](https://arxiv.org/abs/1706.05137)
* [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797)
* [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)
* [Image Transformer](https://arxiv.org/abs/1802.05751)
* [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)
* [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
* [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382)
* [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

## 致谢

感谢 <a href="https://twitter.com/ilblackdragon">Illia Polosukhin</a>、<a href="http://jakob.uszkoreit.net/">Jakob Uszkoreit</a>、<a href="https://www.linkedin.com/in/llion-jones-9ab3064b">Llion Jones </a>、<a href="https://ai.google/research/people/LukaszKaiser">Lukasz Kaiser</a>、<a href="https://twitter.com/nikiparmar09">Niki Parmar</a> 和 <a href="https://dblp.org/pers/hd/s/Shazeer:Noam">Noam Shazeer</a> 对本文早期版本提供的反馈。

如有任何勘误或建议，请在 <a href="https://twitter.com/JayAlammar">Twitter</a> 上联系我。
