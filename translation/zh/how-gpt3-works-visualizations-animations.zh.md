科技圈正为 GPT-3 疯狂[刷屏](https://www.theverge.com/21346343/gpt-3-explainer-openai-examples-errors-agi-potential)。像 GPT-3 这样的大型语言模型开始用能力给人惊喜。虽然它们对大多数企业来说还不足以直接摆到客户面前，但这些模型已经闪现出的灵光，注定会加速自动化进程、拓展智能计算机系统的想象空间。让我们揭开笼罩在 GPT-3 周围的魔法光环，看看它是如何训练的、如何工作的。


 <div style="text-align:center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/MQnJZuBGmSQ" title="YouTube video player" frameborder="0" style="
 width: 100%;
 max-width: 560px;" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

训练好的语言模型会生成文本。

我们可以（可选地）给它输入一些文本，来影响它的输出。

输出源自模型在训练阶段"学到"的东西——那时它扫过了海量文本。


<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/01-gpt3-language-model-overview.gif" />
  <br />

</div>



训练，就是让模型大量接触文本的过程。这个过程已经完成。你现在看到的所有实验，都出自这同一个训练好的模型。据估算，训练耗费了 355 GPU·年，成本 460 万美元。




<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/02-gpt3-training-language-model.gif" />
  <br />

</div>


这个 3000 亿 token 的文本数据集用来为模型生成训练样本。例如，上面那一句话就能生成图中这三个训练样本。

你可以看到，只要把窗口在全部文本上滑动，就能造出海量样本。



<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/gpt3-training-examples-sliding-window.png" />
  <br />

</div>

把一条样本喂给模型：我们只给它输入部分，让它预测下一个词。

模型的预测会是错的。我们计算它预测中的误差，更新模型，让它下一次预测得更好。

重复数百万次。

<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/03-gpt3-training-step-back-prop.gif" />
  <br />

</div>




现在带着更多细节再看一遍这些步骤。

GPT-3 实际上是一次生成一个 token（这里先假设一个 token 就是一个词）。



<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/04-gpt3-generate-tokens-output.gif" />
  <br />

</div>


请注意：本文描述的是 GPT-3 的工作原理，不讨论它新在哪里（新意主要在于大得离谱的规模）。其架构是基于这篇论文的 Transformer 解码器模型：https://arxiv.org/pdf/1801.10198.pdf




GPT-3 庞大得惊人。它把训练所学编码进 1750 亿个数字（称为参数）。每次运行时，正是这些数字用于计算该生成哪个 token。

未训练的模型以随机参数起步。训练要找的，是能让预测更好的那些取值。




<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/gpt3-parameters-weights.png" />
  <br />

</div>


这些数字是模型内部数百个矩阵的一部分。预测过程基本上就是大量的矩阵乘法。

我在 YouTube 的 [AI 入门](https://youtube.com/watch?v=mSTCzNgDJy4)里展示过一个只有 1 个参数的简单 ML 模型——想拆解这头 1750 亿参数的巨兽，那是个好起点。


要弄清这些参数如何分布、如何使用，我们得打开模型往里面看。

GPT-3 有 2048 个 token 宽，这就是它的"上下文窗口"（context window）。也就是说，它有 2048 条轨道，token 沿轨道被处理。

<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/05-gpt3-generate-output-context-window.gif" />
  <br />

</div>


我们沿着紫色轨道跟踪：一个系统是怎么处理 "robotics" 这个词并产出 "A" 的？

高层次步骤：

1. 把词转换为[一个代表该词的向量（一串数字）](https://jalammar.github.io/illustrated-word2vec/)
2. 计算预测
3. 把结果向量转换回词


<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/06-gpt3-embedding.gif" />
  <br />

</div>



GPT-3 的核心计算发生在它 96 层 Transformer 解码器层的堆叠内部。

看到这些层了吗？这就是"深度学习"里"深度"的含义。

每一层都有自己的 18 亿参数来完成计算。"魔法"就发生在这里。下面是这个过程的高层视图：



<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/07-gpt3-processing-transformer-blocks.gif" />
  <br />

</div>


解码器内部每个部分的详细讲解，见我的博客 [The Illustrated GPT2](https://jalammar.github.io/illustrated-gpt2/)。

GPT-3 与 GPT-2 的区别在于交替排布的稠密层与[稀疏自注意力层](https://arxiv.org/pdf/1904.10509.pdf)。






这是输入与回复（"Okay human"）穿过 GPT-3 的一张"X 光片"。注意每个 token 都会流经整个层堆叠。前面那些词的输出我们并不关心；输入结束后，我们才开始关心输出。然后我们把生成的每个词再喂回模型。


<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/08-gpt3-tokens-transformer-blocks.gif" />
  <br />

</div>


在[那个 React 代码生成示例](https://twitter.com/sharifshameem/status/1284421499915403264)里，我认为输入提示（绿色部分）是文字描述，另外还附加了几个"描述 => 代码"的示例。React 代码就会像图中粉色 token 那样逐词生成出来。


我的推测是：引导示例（priming examples）和描述被拼接为输入，用特定的 token 分隔示例与结果，然后送入模型。

<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/09-gpt3-generating-react-code-example.gif" />
  <br />

</div>


事情居然就以这样的方式跑通了，令人惊叹。等你等到 GPT-3 推出微调（fine-tuning）再看——可能性会更加惊人。

微调会真正更新模型的权重，使模型在某类任务上变得更好。

<div class="img-div-any-width" markdown="0">
  <img src="/images/gpt3/10-gpt3-fine-tuning.gif" />
  <br />

</div>
