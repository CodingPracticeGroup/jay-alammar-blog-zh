OpenAI 的 [GPT-OSS 发布](https://openai.com/index/gpt-oss-model-card/)，是继六年前 [GPT-2](https://jalammar.github.io/illustrated-gpt2/) 之后其主要的开源 LLM 发布。这几年间 LLM 能力有了突飞猛进的进步。虽然该模型本身相对于 DeepSeek、Qwen、Kimi 等现有开源模型未必带来能力的跃升，但它给了我们一个好机会，回顾 LLM 在这段时间里发生了哪些变化。

![](https://substackcdn.com/image/fetch/$s_!Mw5k!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fef7bcc2e-e402-4a79-ad3e-de80689b1617_1616x814.png)

## 与之前开源 GPT 模型的差别

GPT-OSS 与之前的模型有相似之处：它也是一个自回归 Transformer，逐 token 生成。

2025 年中的 LLM，主要的差异领域在于：它们生成的 token 能够解决困难得多的问题，靠的是：

- 使用工具

- 推理

- 在问题求解和编程上更强

下图展示了它的主要架构特征——与当前这批强大的开源模型相比并没有大的偏离。它与 GPT2 最大的架构区别在于：GPT-OSS 是一个[混合专家（mixture-of-experts）](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)模型。

![](https://substackcdn.com/image/fetch/$s_!SUm9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc64aa599-20ff-4ad5-a458-d862f3acfd34_2210x1648.png)

如果你想更深入理解这个架构，在我们的免费课程 [How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/?utm_campaign=handsonllm-launch&utm_medium=partner) 里，我们用大量图示（还有独家动画！）详细讲过它。

用我们在课程里引入的这套描述 attention 的视觉语言，GPT-OSS 的 Transformer Block 长下面这样：

![](https://substackcdn.com/image/fetch/$s_!w789!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F113b6ba8-237c-4ede-881b-a5202e56486a_2722x844.png)

需要说明：这些架构细节里没有什么特别新颖的东西。在最新的 SoTA 开源 MoE 模型中，它们大体都是相似的。

# 消息格式（Message Formatting）

对多得多的用户来说，模型的推理与工具调用的行为和格式细节，比架构更重要。

下图可以看到模型的输入与输出长什么样。

![](https://substackcdn.com/image/fetch/$s_!9UZU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8b85c00b-ddc5-4c2b-b015-6d53ce23294e_1980x1824.png)

## 消息与输出通道

我们从开源 LLM 的三类主要用户出发，把这件事拆开讲：

- **LLM 应用的最终用户**

  - 例子：ChatGPT 应用的用户

  - 这类用户主要打交道的是自己发出的用户消息和自己看到的最终回答。在有些应用里，他们可能会看到一部分中间的推理轨迹。

- **LLM 应用的构建者**

  - 例子：Cursor 或 Manus

  - **输入消息**：这些构建者可以设置自己的 system 和 developer 消息——定义模型应有的总体行为与指令、安全选择、推理等级，以及供模型使用的工具定义。他们还必须在用户消息里做大量提示词工程和上下文管理。

  - **输出消息**：构建者可以选择是否向用户展示推理轨迹；他们还要定义工具、设定推理的量。

- **LLM 的后训练者（post-trainers）**

  - 会微调模型的高级用户需要与所有消息类型打交道，并把数据格式化成正确的形状，包括推理、工具调用与响应的格式。

后两类人——LLM 应用构建者和 LLM 后训练者——会受益于理解 assistant 消息的"通道（channel）"概念。它在 [OpenAI Harmony](https://github.com/openai/harmony) 仓库中有实现。

*（如果你觉得这种讲解有用，一定来看看这本畅销书：300 多张图、在这个深度上解释 LLM，其 [GitHub 仓库](https://github.com/handsOnLLM/Hands-On-Large-Language-Models)目前已有 14K stars）*

![Hands-On Large Language Models](https://substackcdn.com/image/fetch/$s_!KUww!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faeca4378-59bd-4c16-8753-a91cbb3bb939_400x525.jpeg "Hands-On Large Language Models")

本书[官方网站](https://www.llm-book.com/)。可以在 [Amazon](https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961) 订购。全部代码已上传到 [GitHub](https://github.com/handsOnLLM/Hands-On-Large-Language-Models)。

## 消息通道

![](https://substackcdn.com/image/fetch/$s_!OUqw!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8918690c-bc23-431c-b060-40c6705a534e_1292x704.png)

模型的输出全都是 assistant 消息。模型会把它们归入一个"通道（channel）"类别，以表明消息类型：

- **Analysis（分析）**：用于推理（以及部分工具调用）

- **Commentary（解说）**：用于函数式调用（以及大部分工具调用）

- **Final（最终）**：包含最终回答的消息

假设我们给模型的提示需要它推理并使用几次工具调用，下图展示了一段三种消息类型都用上的对话：

![](https://substackcdn.com/image/fetch/$s_!VvDI!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F76d71004-9cd6-49c7-ac0a-0949cf578fb7_1348x1220.png)

它们被标为第 1、3、5 轮，因为第 2、4 轮是这些调用的工具响应。最终回答才是最终用户会看到的内容。

## 推理

推理存在权衡，高级用户必须为此做选择。一方面，更多的推理给了模型更多的时间与算力去琢磨问题，帮助它解决更难的问题；另一方面，其代价是延迟和算力。这个选择在现实中很直白：既有强推理 LLM，也有非推理 LLM，各自擅长不同类型的任务。

一个折中方案是提供一个能响应特定*推理预算（reasoning budget）*的推理模型。GPT-OSS 就属于这一类。它允许在 system 消息中设置推理模式（*low*、*medium* 或 *high*）。[模型卡](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)的图 3 展示了这如何影响各项基准的得分，以及推理轨迹（即思维链，CoT）中的 token 数量。

![](https://substackcdn.com/image/fetch/$s_!kYXS!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb08d0ade-6e6c-4a1c-ab29-1168708a55ff_1668x930.png)

可以对比一下 Qwen3 的推理模式：那是二元的 *thinking* / *non-thinking* 模式。对于 *thinking* 模式，它们确实给出了一种在超过某个 token 阈值后停止思考的方法，并[报告](https://qwenlm.github.io/blog/qwen3/)了这对各推理基准得分的影响。

![](https://substackcdn.com/image/fetch/$s_!vhZM!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2d67cd4d-ee73-4b94-87ef-edcbe91fd11f_3180x1970.png)

## 推理模式（low、medium 与 high）

展示不同推理模式差异的好办法，是问一道困难的推理题。所以我从 AIME25 数据集里挑了一道，分别在三种推理模式下问 120B 模型：

![](https://substackcdn.com/image/fetch/$s_!ZCth!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff8855923-20a0-42c6-8f65-31f0725f3b18_2478x1884.png)

这道题的正确答案是 104。于是 *medium* 和 *high* 两种推理模式都答对了，只是 *high* 模式到达这个答案所需的算力/生成时间是前者的两倍。

这正印证了我们前面提过的那点：为你的用例选择合适的推理模式：

- 在做 agentic 任务？如果你的轨迹可能横跨很多步骤，*high* 甚至 *medium* 推理可能都太慢了。

- 实时 vs. 离线——考虑哪些任务可以在用户并不主动等待结果的情况下离线完成。

  - 一个值得琢磨的例子是搜索引擎——查询时之所以能非常快出结果，是因为大量处理与设计工作早已为这种体验准备好了系统。

## 分词器

分词器与 GPT-4 的相当相似，但在我看来效率略高一点——尤其是在非英文 token 上。注意 emoji 和中文字符各自被分成两个 token 而不是三个；阿拉伯语文本也有更多片段被作为一个整体 token 分组，而不是按字母拆分。

不过，尽管分词器在这方面可能更好，模型的训练数据仍以英文为主。

代码（包括 Python 缩进用的制表符 tab）的表现看起来大体一致。数字的分词方式似乎也相同：最多三位的数字分到一个独立 token，更大的数字则被拆开。

![](https://substackcdn.com/image/fetch/$s_!Bpio!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F27b67620-e631-4b45-9f9f-ce3629a9e483_1878x1174.png)

## 延伸阅读

以下是我读到的几篇很值得一看的延伸阅读：

- [From GPT-2 to gpt-oss: Analyzing the Architectural Advances](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the)，作者 [Sebastian Raschka, PhD](https://open.substack.com/users/27393275-sebastian-raschka-phd?utm_source=mentions)

- [gpt-oss: OpenAI validates the open ecosystem (finally)](https://www.interconnects.ai/p/gpt-oss-openai-validates-the-open)，作者 [Nathan Lambert](https://open.substack.com/users/10472909-nathan-lambert?utm_source=mentions)

- [gpt-oss-120B (high): API Provider Benchmarking & Analysis](https://artificialanalysis.ai/models/gpt-oss-120b/providers#aime25x32-performance-gpt-oss-120b)

*（如果你觉得这种讲解有用，一定来看看这本畅销书：300 多张图、在这个深度上解释 LLM，其 [GitHub 仓库](https://github.com/handsOnLLM/Hands-On-Large-Language-Models)目前已有 14K stars）*

![Hands-On Large Language Models](https://substackcdn.com/image/fetch/$s_!KUww!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faeca4378-59bd-4c16-8753-a91cbb3bb939_400x525.jpeg "Hands-On Large Language Models")

本书[官方网站](https://www.llm-book.com/)。可以在 [Amazon](https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961) 订购。全部代码已上传到 [GitHub](https://github.com/handsOnLLM/Hands-On-Large-Language-Models)。
