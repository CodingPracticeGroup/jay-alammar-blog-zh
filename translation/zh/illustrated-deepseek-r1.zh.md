![](https://substackcdn.com/image/fetch/$s_!fn53!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F623a9dbf-c76e-438c-ba69-43ae9613ebbe_2930x1496.png)

DeepSeek-R1 是 AI 进展这通稳定鼓声中最新的一记重响。对机器学习研发社区来说，它是一次重大发布，原因包括：

1. 它是一个开放权重（open weights）模型，并附有更小的蒸馏版本；

2. 它分享并复盘了一种训练方法——用以复现像 OpenAI o1 这样的推理模型。

在这篇文章里，我们将看到它是怎么造出来的。

目录：

- 回顾：LLM 是如何训练的

- DeepSeek-R1 训练配方

- 1- 长推理链 SFT 数据

- 2- 一个临时的、高质量的推理 LLM（但非推理任务更差）

- 3- 用大规模强化学习（RL）创造推理模型

    - 3.1- 大规模推理导向强化学习（R1-Zero）

    - 3.2- 用临时推理模型创造 SFT 推理数据

    - 3.3- 通用 RL 训练阶段

- 架构

理解这类模型所需的大部分基础知识，都在我们的书 [Hands-On Large Language Models](https://github.com/handsOnLLM/Hands-On-Large-Language-Models) 里。

![Hands-On Large Language Models](https://substackcdn.com/image/fetch/$s_!KUww!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faeca4378-59bd-4c16-8753-a91cbb3bb939_400x525.jpeg "Hands-On Large Language Models")

本书[官方网站](https://www.llm-book.com/)。可以在 [Amazon](https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961) 订购。全部代码已上传到 [GitHub](https://github.com/handsOnLLM/Hands-On-Large-Language-Models)。

## 回顾：LLM 是如何训练的

与大多数现有 LLM 一样，DeepSeek-R1 也是逐 token 生成的。只是它特别擅长解决数学与推理问题，因为它能通过生成解释思维链（chain of thought）的"思考 token"，在题目上花更多时间进行加工。

![](https://substackcdn.com/image/fetch/$s_!aHqo!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5280089e-8989-45d7-8194-93396b25557d_613x152.gif)

下面这张图来自我们书的第 12 章，展示了分三步打造高质量 LLM 的通用配方：

![](https://substackcdn.com/image/fetch/$s_!Eopd!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faa354473-6ae0-4ae7-a20c-e858c804d6c4_1600x477.png)

1）语言建模步骤：用海量网络数据训练模型预测下一个词。这一步得到一个基础模型（base model）。

2）监督微调（supervised fine-tuning）步骤：让模型在遵循指令和回答问题方面更有用。这一步得到一个指令微调模型，或称监督微调 / SFT 模型。

3）最后是偏好调优（preference tuning）步骤：进一步打磨其行为、对齐人类偏好，得到你在 playground 和应用里交互的那个最终偏好调优 LLM。

## DeepSeek-R1 训练配方

DeepSeek-R1 遵循这个通用配方。第一步的细节来自[此前 DeepSeek-V3 模型的论文](https://arxiv.org/pdf/2412.19437v1)。R1 使用的是那篇论文里的*基础*模型（不是最终的 DeepSeek-V3 模型），同样会经过 SFT 和偏好调优步骤，但差异恰恰出在它做这些步骤的具体方式上。

![](https://substackcdn.com/image/fetch/$s_!gcwD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc66dff5b-8332-4696-b484-b2ddb029b78c_854x234.png)

R1 的创造过程中有三件特别值得一提的事。

### 1- 长推理链 SFT 数据

![](https://substackcdn.com/image/fetch/$s_!aCYU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F26136780-897d-4f64-b1e5-45936b6078dd_854x434.png)

这是一大批长思维链（long chain-of-thought）推理样例（60 万条）。这类数据极度稀缺，在这个规模上靠人工标注成本极其高昂。这正是"它们是怎么造出来的"成为第二个亮点的理由。

### 2- 一个临时的、高质量的推理 LLM（但非推理任务更差）

这批数据由 R1 的一个前身——一个未命名的、专精推理的"兄弟"模型——创造。这个兄弟又受到第三个模型 *R1-Zero* 的启发（马上就会讲到）。它的意义不在于它是个好用的 LLM，而在于：创造它所需的标注数据少得惊人，配合大规模强化学习，就得到了一个极擅长解决推理问题的模型。

随后，这个未命名的推理专家模型的输出可以用来训练一个更通用的模型，让它同时也能胜任其他非推理任务，达到用户对 LLM 的期望水平。

![](https://substackcdn.com/image/fetch/$s_!J73r!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4caea6a5-52a1-4651-8c71-4586c0637f3e_924x427.png)

### 3- 用大规模强化学习（RL）创造推理模型

这分两步进行：

![](https://substackcdn.com/image/fetch/$s_!paM3!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45ca8c84-6eb6-4879-ab53-035174b17ce1_1620x700.png)

#### **3.1 大规模推理导向强化学习（R1-Zero）**

在这里，RL 被用来创造那个临时推理模型；该模型随后用来生成 SFT 推理样例。而让创造这个模型成为可能的，是更早的一次实验——创建了一个叫 *DeepSeek-R1-Zero* 的更早的模型。

![](https://substackcdn.com/image/fetch/$s_!PgMh!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F69b9f117-caa3-42fd-a949-dc6433990d26_1526x506.png)

R1-Zero 之所以特别，是因为它不需要带标注的 SFT 训练集就能在推理任务上表现出色。它的训练从预训练基础模型直接跳到 RL 训练（没有 SFT 步骤），而且做得如此之好，足以与 o1 竞争。

![](https://substackcdn.com/image/fetch/$s_!Bktd!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7b5c964f-b654-49b2-ab5a-5618b256ef99_1588x418.png)

这件事意义重大，因为数据一直就是 ML 模型能力的燃料。这个模型凭什么跳出这段历史？这指向两件事：

1- 现代基础模型已经越过了某个质量与能力的门槛（这个基础模型是在 14.8 万亿高质量 token 上训练的）。

2- 与一般聊天或写作请求不同，推理问题可以自动验证或自动标注。我们举个例子说明。

##### 示例：推理问题的自动验证

这可以是 RL 训练步骤中使用的某个提示/问题：

> 写一段 Python 代码，接收一个数字列表，按升序返回，并在开头加上 42。

这类问题天然适合多种自动验证方式。假设我们把它交给正在训练的模型，它生成了一个补全：

- 软件 linter 可以检查该补全是否是规范的 Python 代码

- 我们可以运行这段 Python 代码，看它能不能跑起来

- 其他现代编程 LLM 可以创建单元测试来验证期望行为（它们自己并不需要是推理专家）。

- 我们甚至可以更进一步，测量执行时间，让训练过程在同样能解决问题的正确 Python 程序之外，更偏好性能更好的解。

在某个训练步骤中，我们可以把这样的问题交给模型，生成多个可能的解。

![](https://substackcdn.com/image/fetch/$s_!lygC!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8edd9db2-a071-4bba-9d14-bbdb076d6355_798x444.png)

无需人工干预就能自动检查发现：第一个补全根本不是代码；第二个是代码但不是 Python；第三个可能是个解，但没通过单元测试；第四个才是正确的解。

![](https://substackcdn.com/image/fetch/$s_!sUyG!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1f9645a0-b1fb-4753-942c-583504297c25_972x517.png)

这些都是可以直接用来改进模型的信号。当然，这是在大量样本（以小批次）上、跨连续训练步骤完成的。

![](https://substackcdn.com/image/fetch/$s_!3-5E!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5b595e04-bd57-4f78-8c9b-ab37797e9b66_955x543.png)

这些奖励信号与模型更新，就是模型在 RL 训练过程中不断提升任务表现的方式——正如论文图 2 所示。

![](https://substackcdn.com/image/fetch/$s_!MIn4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe48af6fa-8956-44b0-84cf-915e607f3b5e_1546x884.png)

与这一能力提升同步发生的，是生成响应长度的增长：模型会生成更多思考 token 来加工问题。

![](https://substackcdn.com/image/fetch/$s_!ziuE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcd2b7d78-62ac-408c-8bd7-e14053bb8a46_1518x912.png)

这个过程很有用，但 R1-Zero 模型尽管在这些推理题上得分很高，却存在其他问题，使它的可用性不如人意。

> 尽管 DeepSeek-R1-Zero 展现出了很强的推理能力，并自主发展出意外而强大的推理行为，它仍面临若干问题。例如，DeepSeek-R1-Zero 存在可读性差、语言混杂等困难。

R1 的目标是一个可用性更强的模型。因此，正如本节早前所述，它没有完全依赖 RL 过程，而是把 RL 用在两处：

1- 创造那个生成 SFT 数据点的临时推理模型；

2- 训练 R1 模型本身，提升其在推理与非推理问题上的表现（使用其他类型的验证器）。

![](https://substackcdn.com/image/fetch/$s_!paM3!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45ca8c84-6eb6-4879-ab53-035174b17ce1_1620x700.png)

#### 3.2 用临时推理模型创造 SFT 推理数据

为了让这个临时推理模型更有用，它在几千条推理题样例（其中一部分由 R1-Zero 生成并经过筛选）上做了一步监督微调（SFT）。论文把这称为"冷启动数据"（cold start data）：

> **2.3.1. 冷启动**  
> 与 DeepSeek-R1-Zero 不同，为了避免从基础模型直接开始 RL 训练时早期不稳定的冷启动阶段，DeepSeek-R1 构造并收集了一小部分长思维链（CoT）数据，先对模型做微调，作为 RL 的初始 actor。为了收集这些数据，我们探索了若干方法：使用带长思维链示例的 few-shot 提示；直接提示模型生成带有反思与验证的详细回答；以可读格式收集 DeepSeek-R1-Zero 的输出；以及由人类标注员对结果做后处理精修。

![](https://substackcdn.com/image/fetch/$s_!E5Rw!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1a89a9a0-c08f-430d-b135-7f012c2810ba_1824x586.png)

但等等，既然我们有这批数据，为什么还要依赖 RL 过程？因为数据的规模。这个数据集可能有 5,000 条（这是可以搜集到的），但要训练 R1，需要 60 万条。这个临时模型恰好弥合了这个缺口，让我们能以合成方式生成那份极其宝贵的数据。

![](https://substackcdn.com/image/fetch/$s_!rbw2!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F857e61c8-03e7-4bc7-bcbe-ca182f60a70e_3300x1170.png)

如果你是"监督微调（SFT）"概念的新手：它的做法是以"提示 + 正确补全"的形式向模型提供训练样例。这张来自第 12 章的图展示了几个 SFT 训练样例：

![](https://substackcdn.com/image/fetch/$s_!xHUV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8b630dbc-aaa4-4c27-804b-542055b0f298_2264x1324.png)

#### 3.3 通用 RL 训练阶段

这一步使 R1 既能在推理上出色，也能胜任其他非推理任务。过程与我们前面看到的 RL 过程类似。但由于扩展到了非推理应用，它对这些应用所属的提示使用了有用性（helpfulness）与安全（safety）两个奖励模型（与 Llama 系列模型的做法相仿）。

![](https://substackcdn.com/image/fetch/$s_!HRLo!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0e5f9acf-b4ca-4ec4-9731-4845c8fc5515_902x394.png)

## 架构

与 [GPT2](https://jalammar.github.io/illustrated-gpt2/) 和 [GPT-3](https://jalammar.github.io/how-gpt3-works-visualizations-animations/) 开创时代以来的既往模型一样，DeepSeek-R1 是一个 [Transformer](https://jalammar.github.io/illustrated-transformer/) 解码器块的堆叠，共由 61 个块组成。前 3 个是稠密层，其余都是混合专家（mixture-of-experts）层（参见我合著者 Maarten 的精彩入门指南：[A Visual Guide to Mixture of Experts (MoE)](https://substack.com/home/post/p-148217245)）。

![](https://substackcdn.com/image/fetch/$s_!YRrM!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F199f326e-9a8d-4a95-8574-4778d5b7657b_538x413.png)

模型维度大小及其他超参数如下：

![](https://substackcdn.com/image/fetch/$s_!CUyJ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2ee664ae-a544-4e19-a145-0ae87acc43fa_916x481.png)

关于模型架构的更多细节在他们此前两篇论文中：

- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437v1)

- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)

### 结语

读到这里，理解 DeepSeek-R1 模型所需的主要直觉你就都具备了。

![](https://substackcdn.com/image/fetch/$s_!a0EN!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fed7fd8c3-7654-497c-a8e2-1f2e7930992e_3302x1438.png)

如果你觉得理解本文还需要一点基础知识，建议入一本 [Hands-On Large Language Models](https://www.llm-book.com/)，或在 [O'Reilly](https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/) 上在线阅读，也可以在 [Github](https://github.com/handsOnLLM/Hands-On-Large-Language-Models) 上查看。

![Book Cover](https://substackcdn.com/image/fetch/$s_!PPT9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbd7beb5f-e943-4d2d-8b4c-eb1e80231670_582x768.png "Book Cover")

其他推荐资源：

- [A Visual Guide to Reasoning LLMs](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms?)，作者 [Maarten Grootendorst](https://open.substack.com/users/14309499-maarten-grootendorst?utm_source=mentions)

- [DeepSeek R1's recipe to replicate o1 and the future of reasoning LMs](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1)，作者 [Nathan Lambert](https://open.substack.com/users/10472909-nathan-lambert?utm_source=mentions)

- [A Visual Guide to Mixture of Experts (MoE)](https://substack.com/home/post/p-148217245)，作者 [Maarten Grootendorst](https://open.substack.com/users/14309499-maarten-grootendorst?utm_source=mentions)

- Sasha Rush 的 YouTube 视频 [Speculations on Test-Time Scaling (o1)](https://www.youtube.com/watch?v=6PEJ96k1kiw)

- Yannis Kilcher 的 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Paper Explained)](https://www.youtube.com/watch?v=bAWV_yrqx4w)

- [Open R1](https://github.com/huggingface/open-r1) 是 HuggingFace 公开复现 DeepSeek-R1 的项目

- [Putting RL back in RLHF](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo)

- 读这篇论文时，我想起了 2022 年的 [Galactica 论文](https://arxiv.org/abs/2211.09085)。它有很多很棒的想法，其中包括一个专门的"思考 token"。
