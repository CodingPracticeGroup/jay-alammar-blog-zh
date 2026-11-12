(**2022 年 11 月 V2**：更新了若干图片，以更准确地描述前向扩散过程；本版还增加了几张图)

AI 图像生成是近期最让人们（包括我）大开眼界的 AI 能力。从文字描述创造出生动画面，这种能力有一种魔力，也清晰地指向人类创作艺术方式的转变。[Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) 的发布是这一浪潮中一座明显的里程碑，因为它把一个高性能模型送到了大众手中（这个"性能"既指图像质量，也指速度和相对较低的资源/内存要求）。

<iframe width="560" height="315" src="https://www.youtube.com/embed/MXmacOUJUaw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" style="
 width: 100%;
 max-width: 560px;" allowfullscreen></iframe>

玩了一会儿 AI 图像生成之后，你可能会开始好奇它是如何工作的。

这是一篇关于 Stable Diffusion 工作原理的轻松入门。



<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-text-to-image.png" />
  <br />

</div>

Stable Diffusion 很全能，可以用在很多不同的玩法上。我们先聚焦最基础的"仅从文字生成图像"（text2img）。上图展示了一个文字输入的示例及生成的图像（完整的原始提示词见这里）。除了文生图，它的另一个主要用法是让它修改图像（即输入为文字 + 图像）。


<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-img2img-image-to-image.png" />
  <br />

</div>

我们来看看到它的引擎盖底下。这有助于理解各个组件、它们如何相互作用，以及图像生成的那些选项/参数到底意味着什么。

## Stable Diffusion 的组件

Stable Diffusion 是一个由若干组件和模型组成的系统，并不是一个铁板一块的单一大模型。 

往内部看，我们能得到的第一个观察是：存在一个"文字理解"组件，它把文字信息翻译成一种数值表示，以捕捉文字中的含义。 

<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-text-understanding-component-image-generation.png" />
  <br />

</div>

我们从一个高层次的视角出发，文章后面会进入更具体的机器学习细节。不过现在可以说：这个文本编码器是一个特殊的 Transformer 语言模型（技术上说：CLIP 模型的文本编码器）。它接收输入文本，输出文本中每个词/token 的数值表示列表（每个 token 一个向量）。

这些信息随后被交给"图像生成器"，而图像生成器本身又由几个组件构成。

<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/Stable-diffusion-text-info-to-image-generator.png" />
  <br />

</div>

图像生成器分两个阶段：

1- **图像信息创造器（Image information creator）**

这个组件是 Stable Diffusion 的独门秘方，相对此前模型的性能提升有很多就来自这里。 

它以多步迭代来生成图像信息。这就是 Stable Diffusion 界面和库里那个 *steps* 参数，通常默认为 50 或 100。

图像信息创造器完全在*图像信息空间*（即*潜空间（latent）*）中工作，这后文会展开。这一特性使它比在像素空间工作的早期扩散模型更快。技术上说，这个组件由一个 UNet 神经网络和一个调度（scheduling）算法组成。

"扩散（diffusion）"一词描述的就是这个组件里发生的事：信息一步一步地被加工，最终（由下一个组件，图像解码器）生成一张高质量图像。

<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/Stable-diffusion-image-generator-information-creator.png" />
  <br />

</div>

2- **图像解码器（Image Decoder）**

图像解码器根据信息创造器提供的信息"作画"。它在流程末尾只运行一次，产出最终的像素图像。


<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-cliptext-unet-autoencoder-decoder.png" />
  <br />

</div>

到这里，我们就看到了构成 Stable Diffusion 的三个主要组件（各自都有自己的神经网络）：

- **ClipText**：文本编码。<br />
  输入：文本。<br />
  输出：77 个 token 嵌入向量，每个 768 维。
  
- **UNet + 调度器（Scheduler）**：在信息（潜）空间中逐步加工/扩散信息。<br />
  输入：文本嵌入，以及一个由噪声构成的初始多维数组（结构化的数字列表，也叫*张量（tensor）*）。<br />
  输出：加工后的信息数组。

- **自编码器解码器（Autoencoder Decoder）**：用加工后的信息数组画出最终图像。<br />
输入：加工后的信息数组（维度：(4,64,64)）  <br />
  输出：最终图像（维度：(3, 512, 512)，即（红/绿/蓝，宽，高））



<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-components-and-tensors.png" />
  <br />

</div>

## 话说，扩散到底是什么？

扩散就是发生在粉色"图像信息创造器"组件内部的过程。它持有代表输入文本的 token 嵌入，以及一个随机的起始*图像信息数组*（也叫 *latents*），然后产出一个信息数组，供图像解码器绘制最终图像。

<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-diffusion-process.png" />
  <br />

</div>


这个过程一步一步地进行，每一步都加入更多相关信息。要建立对过程的直觉，我们可以检查一下那个随机的 latents 数组，看看它对应到的其实是一片视觉噪声。这里的"视觉检查"就是把它送进图像解码器看看。



<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-latent-space-pixel-space.png" />
  <br />

</div>

扩散分多步进行，每一步接收一个 latents 数组作为输入，产出另一个 latents 数组——它更接近输入文本，也更接近模型从其训练过的所有图像中学到的视觉信息。



<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-unet-steps.png" />
  <br />

</div>

我们可以把一系列 latents 可视化出来，看看每一步加入了什么信息。


<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-denoising-steps-latents.png" />
  <br />

</div>


这个过程看起来相当震撼。

<div class="img-div-any-width" markdown="0">
<video height="auto" loop autoplay  controls>
  <source src="/images/stable-diffusion/diffusion-steps-all-loop.webm" type="video/webm">
  Your browser does not support the video tag.
</video>
</div>

这个例子里，第 2 步到第 4 步之间发生的事情特别迷人：轮廓仿佛从噪声中浮现出来。


<div class="img-div-any-width" markdown="0">
<video height="auto" loop autoplay  controls>
  <source src="/images/stable-diffusion/stable-diffusion-steps-2-4.webm" type="video/webm">
  Your browser does not support the video tag.
</video>
</div>

### 扩散的原理
用扩散模型生成图像的核心思想，建立在"我们拥有强大的计算机视觉模型"这一事实上。只要有足够大的数据集，这些模型就能学会复杂的操作。扩散模型把图像生成问题重新表述为：

假设我们有一张图像，我们生成一些噪声，把它加到图像上。


<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-forward-diffusion-training-example.png" />
  <br />

</div>
这就成了一个训练样本。我们再用同样的公式生成大量训练样本，来训练我们图像生成模型的核心组件。 

<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-forward-diffusion-training-example-2.png" />
  <br />

</div>


这个例子展示的加噪幅度是从原图（幅度 0，无噪声）到完全噪声（幅度 4，全是噪声）的几个值。我们可以精确控制给图像加多少噪声，因此可以把噪声幅度铺开到几十步上，为训练集中的每张图像造出几十个训练样本。

<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-u-net-noise-training-examples-2.png" />
  <br />

</div>

有了这个数据集，我们就可以训练"噪声预测器"，得到一个出色的噪声预测器——以某种特定配置运行时，它竟然能创造出图像。如果你有机器学习经验，下面这个训练步骤应该不陌生：


<div class="img-div" markdown="0">
<a href="/images/stable-diffusion/stable-diffusion-u-net-noise-training-step.png">
  <img src="/images/stable-diffusion/stable-diffusion-u-net-noise-training-step.png" />
  </a><br />

</div>

接下来看看它如何用来生成图像。

### 靠去噪来作画

训练好的噪声预测器可以接收一张带噪声的图像和去噪步的步数，预测出一片噪声。 
<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-denoising-step-1v2.png" />
  <br />

</div>

预测这片噪声的目标是：把它从图像中减掉后，得到的图像更接近模型训练时所见过的那些图像——不是精确的某些图，而是那个*分布*：天空通常是蓝的、地面在天上之下、人有两只眼睛、猫是某种模样（尖耳朵、一脸"不过如此"）的像素排列世界。

<div class="img-div" markdown="0">
  <a href="/images/stable-diffusion/stable-diffusion-denoising-step-2v2.png"><img src="/images/stable-diffusion/stable-diffusion-denoising-step-2v2.png" /></a>
  <br />

</div>

如果训练数据集是审美上令人愉悦的图像（例如 Stable Diffusion 所用的 [LAION Aesthetics](https://laion.ai/blog/laion-aesthetics/)），那么生成的图像就会倾向于同样赏心悦目。如果我们在 logo 图上训练，就会得到一个 logo 生成模型。

<div class="img-div" markdown="0">
<a href="/images/stable-diffusion/stable-diffusion-image-generation-v2.png">
  <img src="/images/stable-diffusion/stable-diffusion-image-generation-v2.png" />
</a>
  <br />

</div>

至此，扩散模型生成图像的描述就基本完成了，主要内容来自 [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)。有了这套对扩散的直觉，你不只是理解了 Stable Diffusion——Dall-E 2 和 Google 的 Imagen 的主要组件你也懂了。

注意：到目前为止我们描述的扩散过程完全没用任何文本数据。也就是说，直接部署这个模型，它能生成好看的图像，但我们无法控制它生成的是金字塔还是猫。下一节我们说明如何把文本纳入这个过程，以控制模型生成什么类型的图像。

## 提速：在压缩（潜）数据而不是像素图像上做扩散

为了加快图像生成过程，Stable Diffusion 论文把扩散过程不是放在像素图像上跑，而是放在图像的压缩版本上跑。[论文](https://arxiv.org/abs/2112.10752)称之为"迈向潜空间（Departure to Latent Space）"。  

这个压缩（以及之后的解压/作画）通过自编码器（autoencoder）完成。自编码器用编码器把图像压缩进潜空间，再仅凭压缩后的信息用解码器重建图像。

<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-autoencoder.png" />
  <br />

</div>

现在前向扩散过程在压缩后的 latents 上进行。预测的"噪声片"是施加在这些 latents 上的噪声，而不是像素图像上的噪声。所以噪声预测器实际上是在压缩表示（潜空间）中学会预测噪声的。

<div class="img-div" markdown="0">
<a href="/images/stable-diffusion/stable-diffusion-latent-forward-process-v2.png">
  <img src="/images/stable-diffusion/stable-diffusion-latent-forward-process-v2.png" />
</a>
  <br />

</div>

前向过程（用自编码器的编码器）是我们为噪声预测器生成训练数据的方式。训练好之后，我们就能通过运行逆过程（用自编码器的解码器）来生成图像。


<div class="img-div" markdown="0">
<a href="/images/stable-diffusion/stable-diffusion-forward-and-reverse-process-v2.png">
  <img src="/images/stable-diffusion/stable-diffusion-forward-and-reverse-process-v2.png" />
</a>
  <br />

</div>


这两条流程，就是 LDM/Stable Diffusion 论文图 3 所展示的内容：

<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/article-Figure3-1-1536x762.png" />
  <br />

</div>

这张图还画出了"条件化（conditioning）"组件，在本例中即描述"模型应该生成什么样的图像"的文本提示。那我们来研究一下文本部分。

### 文本编码器：一个 Transformer 语言模型

语言理解组件用的是一个 Transformer 语言模型：接收文本提示，产出 token 嵌入。发布的 Stable Diffusion 模型使用 ClipText（一个 [基于 GPT 的模型](/illustrated-gpt2/)），而论文用的是 [BERT](/illustrated-bert/)。

语言模型的选型有多重要，Imagen 论文给出了证明：换上更大的语言模型，对生成图像质量的影响，比加大图像生成组件还大。

<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/text-language-models-clip-image-generation.png" />
  <br />

  更大/更好的语言模型对图像生成模型的质量影响显著。来源：<a href="https://arxiv.org/abs/2205.11487">Google Imagen 论文（Saharia et. al.）</a>，图 A.5。

</div>

早期的 Stable Diffusion 模型直接接上了 OpenAI 发布的预训练 ClipText 模型。未来的模型有可能改用新发布的、大得多的 [OpenCLIP](https://laion.ai/blog/large-openclip/) CLIP 变体（2022 年 11 月更新：果然如此，[Stable Diffusion V2 用上了 OpenClip](https://stability.ai/blog/stable-diffusion-v2-release)）。新一批模型的文本模型规模可达 3.54 亿参数，而 ClipText 只有 6300 万参数。

#### CLIP 是怎么训练的

CLIP 在图像及其配文组成的数据集上训练。想象一个长这样的数据集，只不过它有 4 亿张图及其配文：

<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/images-and-captions-dataset.png" />
  <br />
  一个由图像及其配文组成的数据集。
</div>

实际上，CLIP 是在从网络爬取的图像及它们的 "alt" 标签上训练的。

CLIP 是一个图像编码器加一个文本编码器的组合。它的训练过程可以简化理解为：取一张图和它的配文，分别用图像编码器和文本编码器编码。 

<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/clip-training-step-1.png" />
  <br />
  
</div>

然后用余弦相似度比较得到的两个嵌入。训练开始时，哪怕文字正确地描述了图像，相似度也会很低。 

<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/clip-training-step-2.png" />
  <br />
  
</div>

我们更新这两个模型，使下次再编码它们时，得到的嵌入彼此相似。 

<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/clip-training-step-3.png" />
  <br />
  
</div>


在整个数据集上、用大批量重复这个过程，最终两个编码器就能产出这样的嵌入：一张狗的照片和"a picture of a dog"这句话是相似的。和 [word2vec](/illustrated-word2vec/) 一样，训练过程还需要包含互不匹配的图像与配文的**负样本**，模型要给它们打很低的相似度分数。

## 把文本信息喂进图像生成过程

要让文本成为图像生成过程的一部分，我们必须调整噪声预测器，让它把文本作为输入。 



<div class="img-div" markdown="0">
<a href="/images/stable-diffusion/stable-diffusion-unet-inputs-v2.png">
  <img src="/images/stable-diffusion/stable-diffusion-unet-inputs-v2.png" />
  </a><br />

</div>


我们的数据集现在包含了编码后的文本。由于我们工作在潜空间，输入图像和预测的噪声都处于潜空间中。

<div class="img-div" markdown="0">
  <img src="/images/stable-diffusion/stable-diffusion-text-dataset-v2.png" />
  <br />

</div>

为了对 Unet 如何使用文本 token 有更具体的感受，我们钻进 Unet 内部看看。

### Unet 噪声预测器的层（不含文本）

先看一个不使用文本的扩散 Unet。它的输入输出长这样：

<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/unet-inputs-outputs-v2.png" />
  <br />

</div>

在里面我们可以看到：

* Unet 是一系列层，负责变换 latents 数组
* 每一层都在上一层的输出上工作
* 一些输出会通过残差连接（residual connections）馈入网络更后面的处理中
* 时间步（timestep）被变换成一个时间步嵌入向量，各层用的就是这个向量 


<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/unit-resnet-steps-v2.png" />
  <br />

</div>


### Unet 噪声预测器的层（含文本）


现在看看如何改造这套系统，让它能关注文本。

<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/unet-with-text-inputs-outputs-v2.png" />
  <br />

</div>

为支持文本输入（专业术语：文本条件化，text conditioning）所需的主要改动，是在 ResNet 块之间加一个注意力层。


<div class="img-div-any-width" markdown="0">
  <img src="/images/stable-diffusion/unet-with-text-steps-v2.png" />
  <br />

</div>

注意，ResNet 块本身并不直接"看"文本。是注意力层把那些文本表示融进了 latents。随后的 ResNet 就能在自己的处理中利用这份已融入的文本信息。

## 结语

希望本文给了你一个关于 Stable Diffusion 如何工作的良好直觉起点。它还牵涉许多其他概念，但我认为，一旦你熟悉了上面的这些积木，它们都会变得容易理解。下面列出的资源是我觉得很有用的下一步。如有任何勘误或反馈，请在 <a href="https://twitter.com/JayAlammar">Twitter</a> 上联系我。


## 资源

* 我拍了一个[一分钟的 YouTube 短片](https://youtube.com/shorts/qL6mKRyjK-0?feature=share)，演示用 [Dream Studio](https://beta.dreamstudio.ai/) 以 Stable Diffusion 生成图像。
* [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion)
* [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
* [How does Stable Diffusion work? – Latent Diffusion Models EXPLAINED](https://www.youtube.com/watch?v=J87hffSMB60) [视频]
* [Stable Diffusion - What, Why, How?](https://www.youtube.com/watch?v=ltLNYA3lWAQ) [视频]
* [High-Resolution Image Synthesis with Latent Diffusion Models](https://ommer-lab.com/research/latent-diffusion-models/) [Stable Diffusion 论文]
* 想要更深入地看算法与数学，请看 Lilian Weng 的 [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
* 观看 fast.ai 的 [精彩 Stable Diffusion 视频](https://www.youtube.com/watch?v=_7rMfsA24Ls&ab_channel=JeremyHoward)


## 致谢

感谢 Robin Rombach、Jeremy Howard、Hamel Husain、Dennis Soemers、Yan Sidyakin、Freddie Vargus、Anna Golubeva 以及 [Cohere For AI](https://cohere.for.ai/) 社区对本文早期版本提供的反馈。

## 参与贡献

请帮我把这篇文章做得更好。可以这样做：

* 通过 [Twitter](https://twitter.com/JayAlammar) 或 [Pull Request](https://github.com/jalammar/jalammar.github.io) 提出任何反馈或勘误
* 为图示建议更易懂的说明文字和 alt-text，帮助更多人看懂这篇文章（最好以 pull request 提交）
* 把它翻译成另一种语言并发布在你的博客上。把链接发给我，我会在这里加上链接。此前参与翻译的朋友们都说，把翻译做一遍之后，对这些概念的理解深刻得多。


## 讨论

如果你对图像生成模型与语言模型的交叉领域感兴趣，欢迎在 [Cohere Discord 社区](https://discord.gg/co-mmunity)的 #images-and-words 频道发帖。我们在其中讨论各种交叉话题，包括：

* 微调语言模型以生成好的图像生成提示词
* 用 LLM 把图像描述提示词拆分为主体和风格两部分
* 图生提示词（image-to-prompt，借助 [Clip Interrogator](https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb) 之类的工具）

## 引用

如果这项工作对你的研究有帮助，请按如下方式引用：

<div class="cite" markdown="1">


```code
@misc{alammar2022diffusion, 
  title={The Illustrated Stable Diffusion},
  author={Alammar, J},
  year={2022},
  url={https://jalammar.github.io/illustrated-stable-diffusion/}
}
```

</div>
