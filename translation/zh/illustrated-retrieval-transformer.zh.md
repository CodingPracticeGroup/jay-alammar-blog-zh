**摘要**：最新一批语言模型可以做得小得多，却能凭借查询数据库或搜索网络获取信息的能力，达到接近 GPT-3 的性能。这是一个重要信号：把模型越做越大，并不是提升性能的唯一途径。

## 视频
<iframe width="560" height="315" src="https://www.youtube.com/embed/sMPq4cVS4kg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"  style="
width: 100%;
max-width: 560px;" allowfullscreen></iframe>

<hr />

过去几年见证了大语言模型（LLM）的崛起——这类机器学习模型正在迅速改变机器处理与生成语言的方式。2017 年以来的部分亮点包括：

*   原始的 [Transformer](/illustrated-transformer/) 打破了机器翻译此前的性能纪录。
* [BERT](/illustrated-bert/) 让"先预训练、再微调"的流程以及基于 Transformer 的上下文化词嵌入得到普及，并迅速开始为 [Google 搜索](https://blog.google/products/search/search-language-understanding-bert/)和 [Bing 搜索](https://azure.microsoft.com/en-us/blog/bing-delivers-its-largest-improvement-in-search-experience-using-azure-gpus/)提供支持。
* [GPT-2](/illustrated-gpt2/) 展示了机器写作可以达到与人类相当的水平。
* 先是 [T5](https://arxiv.org/abs/1910.10683)，然后是 [T0](https://huggingface.co/bigscience/T0pp)，把迁移学习（在一个任务上训练模型，然后让它在其他相邻任务上也表现出色）的边界不断向前推进，并把大量不同的任务统一表述为文本到文本的任务。
* [GPT-3](/how-gpt3-works-visualizations-animations/) 表明，生成式模型的大规模扩展可以带来令人震撼的涌现应用（业界仍在训练更大的模型，如 [Gopher](https://deepmind.com/research/publications/2021/scaling-language-models-methods-analysis-insights-from-training-gopher)、[MT-NLG](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) 等）。

有一段时间，把模型越做越大似乎成了提升性能的主要途径。而这一领域的最新进展——DeepMind 的 [RETRO Transformer](https://deepmind.com/research/publications/2021/improving-language-models-by-retrieving-from-trillions-of-tokens) 和 OpenAI 的 [WebGPT](https://openai.com/blog/improving-factual-accuracy/)——扭转了这个趋势：如果我们为较小的生成式语言模型配上一个搜索/查询信息的手段，它们就能与大模型并驾齐驱。

本文拆解 DeepMind 的 RETRO（**R**etrieval-**E**nhanced **TR**ansf**O**rmer，检索增强 Transformer）及其工作原理。该模型的规模只有 GPT-3 的 4%（75 亿参数，而 GPT-3 Da Vinci 是 1850 亿），性能却与 GPT-3 相当。

<div class="img-div" markdown="0">
  <img src="/images/retro/deepmind-retro-retrieval-transformer.png" />
  <br />
  RETRO 将检索自数据库的信息纳入模型，让宝贵的参数不必充当昂贵的"事实与世界知识仓库"。
</div>

RETRO 发表在论文 [Improving Language Models by Retrieving from Trillions of Tokens](https://arxiv.org/abs/2112.04426) 中。它延续并建立于研究社区大量检索相关工作之上（[一](http://www.crm.umontreal.ca/2018/Langue18/pdf/Cheung.pdf)、[二](https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)、[三](https://openreview.net/forum?id=HklBjCEKvH)、[四](https://arxiv.org/abs/2102.02557)、[五](https://openreview.net/forum?id=B184E5qee)）。本文解释的是模型本身，而非它特别新颖的地方。

## 为什么这很重要：把语言信息与世界知识信息分开

语言建模训练模型预测下一个词——本质上就是在句子末尾做填空。

填空有时需要事实性知识（例如人名或日期）。比如：

<div class="img-div" markdown="0">
  <img src="/images/retro/prompt-1.png" />
  <br />
  输入提示：The Dune film was released in ....（《沙丘》电影上映于……）
</div>

另一些时候，仅凭对语言的熟悉程度就足以猜出该填什么。比如：

<div class="img-div" markdown="0">
  <img src="/images/retro/prompt-2.png" />
  <br />
  输入提示：its popularity spread by word-of-mouth to allow Herbert to start working full ....（它的名声靠口口相传，使赫伯特得以开始全职……）
</div>

这个区分很重要，因为 LLM 把它们知道的一切都编码进了模型参数。对语言信息来说这说得通，但对事实与世界知识信息来说非常低效。

通过在语言模型中加入检索方法，模型可以小得多。一个"神经数据库"能在文本生成过程中帮它检索所需的事实信息。

<div class="img-div-any-width" markdown="0">
  <img src="/images/retro/Large-GPT-vs-Retro-transformer-world-knowledge-information.png" />
  <br />
  用检索方法辅助语言模型，能减少语言模型为了出色地完成文本生成而必须编码进参数里的信息量。
</div>

小型语言模型的训练也因此更快，因为需要"背下来"的训练数据减少了。任何人都可以把这些模型部署在更小、更便宜的 GPU 上，并按需调整它们。

从结构上看，RETRO 就是一个像原始 Transformer 那样的编码器-解码器模型。只不过它借助检索数据库来增强输入序列：模型在数据库中找到最可能的序列，把它们加入输入，然后 RETRO 施展魔法生成输出预测。

<div class="img-div" markdown="0">
  <img src="/images/retro/dune-prompt-into-retro-transformer-4.png" />
  <br />
  RETRO 用数据库增强输入提示：先用提示从数据库检索相关信息。
</div>

在研究模型架构之前，我们先深入看看检索数据库。

检查 RETRO 的检索数据库
-------------------------------------

数据库是一个键值存储（key-value store）。

键（key）是一个标准的 BERT 句子嵌入。

值（value）是文本，分两部分：

1.  邻居（Neighbor），用来计算键的那段文本；

2.  续文（Completion），原始文档中该文本的后续内容。

RETRO 的数据库基于 *MassiveText* 数据集，包含 2 万亿个多语言 token。邻居块与续文块的长度都不超过 64 个 token。

<div class="img-div-any-width" markdown="0">
  <img src="/images/retro/database-key-value-examples.png" />
  <br />
  RETRO 数据库内部一瞥：展示了数据库中键值对的示例。值包含一个邻居块和一个续文块。
</div>

RETRO 把输入提示切成多个块（chunk）。为简单起见，我们只关注一个块如何被检索到的文本增强。模型实际会对输入提示中的每个块（第一个块除外）执行这个过程。

数据库查找
-------------------

在到达 RETRO 之前，输入提示先送入 BERT。输出的上下文化向量取平均，构造出一个句子嵌入向量。然后用这个向量去查询数据库。

<div class="img-div-any-width" markdown="0">
  <img src="/images/retro/bert-sentence-embedding.png" />
  <br />
  用 BERT 处理输入提示，得到上下化的 token 嵌入；对它们取平均，得到句子嵌入。
</div>

然后用这个句子嵌入做近似最近邻搜索（<https://github.com/google-research/google-research/tree/master/scann>）。

检索出两个最近邻，它们的文本成为 RETRO 输入的一部分。

<div class="img-div" markdown="0">
  <img src="/images/retro/neighbor-retrieval-from-retro-neural-database-with-bert-embeddings.png" />
  <br />
  用 BERT 句子嵌入从 RETRO 的神经数据库中检索最近邻，然后把它们加入语言模型的输入。
</div>

这就是 RETRO 现在的输入：输入提示，加上数据库中它的两个最近邻（以及它们的续文）。

从这里开始，Transformer 块和 RETRO 块会把这些信息纳入自己的处理过程。

<div class="img-div" markdown="0">
  <img src="/images/retro/input-prompt-and-retrieved-text-retro-transformer.png" />
  <br />
  检索到的邻居被加入语言模型的输入。不过在模型内部，它们会被稍作不同的处理。
</div>

RETRO 架构概览
----------------------------------

RETRO 的架构是一个编码器栈加一个解码器栈。

<div class="img-div" markdown="0">
  <img src="/images/retro/Retro-transformer-encoder-decoder-stacks-2.png" />
  <br />
  一个 RETRO Transformer 由编码器栈（处理邻居）和解码器栈（处理输入）组成
</div>

编码器由标准的 Transformer 编码器块（自注意力 + FFNN）构成。据我理解，RETRO 使用的编码器由两个 Transformer 编码器块组成。

解码器栈交替使用两种解码器块：

* 标准 Transformer 解码器块（ATTN + FFNN）
* RETRO 解码器块（ATTN + 分块交叉注意力（Chunked cross attention，CCA）+ FFNN）

<div class="img-div" markdown="0">
  <img src="/images/retro/retro-transformer-blocks-4.png" />
  <br />
  构成 RETRO 的三种 Transformer 块
</div>

先看编码器栈：它处理检索到的邻居，得到之后供注意力使用的 KEYS 和 VALUES 矩阵（复习可看 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)）。

<div class="img-div" markdown="0">
  <img src="/images/retro/retro-encoder-block-keys-values-2.png" />
  <br />
  编码器栈处理检索到的邻居，得到 KEYS 和 VALUES 矩阵
</div>

解码器块像 GPT 一样处理输入文本：先对提示 token 施加（因果的，即只关注之前 token 的）自注意力，再通过 FFNN 层。

<div class="img-div" markdown="0">
  <img src="/images/retro/retro-transformer-decoders-2.png" />
  <br />
  输入提示经过包含自注意力和 FFNN 层的标准解码器块
</div>

只有走到 RETRO 解码器块时，才开始纳入检索到的信息。从第 9 层开始，每隔三层就有一个 RETRO 块（它允许自己的输入关注邻居）。也就是说，第 9、12、15……32 层是 RETRO 块。（两个较小的 RETRO 模型和 Retrofit 模型则从第 6 层开始，而不是第 9 层。）

<div class="img-div" markdown="0">
  <img src="/images/retro/retro-decoder-attention-2.png" />
  <br />
  输入提示到达 RETRO 解码器块，信息检索开始
</div>

所以实际上，就是在这一步，检索到的信息可以"瞥一眼"完成提示所需要的日期等资料。

<div class="img-div" markdown="0">
  <img src="/images/retro/retro-decoder-chunked-cross-attention.png" />
  <br />
  RETRO 解码器块通过分块交叉注意力（Chunked Cross-Attention）从最近邻块中检索信息
</div>

## 相关工作

用检索技术辅助语言模型一直是活跃的研究方向。此前的一些工作包括：

-   [Improving Neural Language Models with a Continuous Cache](https://openreview.net/forum?id=B184E5qee)
-   [Generalization through Memorization: Nearest Neighbor Language Models](https://openreview.net/forum?id=HklBjCEKvH)
-   阅读 Meta AI 的 [Retrieval Augmented Generation](https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/) 博客，以及 Jackie Chi Kit Cheung 关于 [Leveraging External Knowledge in Natural Language Understanding Systems](http://www.crm.umontreal.ca/2018/Langue18/pdf/Cheung.pdf) 的讲义
-   SPALM：[Adaptive Semiparametric Language Models](https://arxiv.org/abs/2102.02557)
-   DPR：[Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2020.emnlp-main.550/)
-   [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
-   FiD：[Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://aclanthology.org/2021.eacl-main.74/)
-   EMDR：[End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering](https://arxiv.org/abs/2106.05346)
-   BlenderBot 2.0：[Internet-Augmented Dialogue Generation](https://arxiv.org/abs/2107.07566)

如有任何勘误或反馈，请在[这个帖子](https://github.com/jalammar/jalammar.github.io/discussions/21)留言，或在 <a href="https://twitter.com/JayAlammar">Twitter</a> 上联系我。
