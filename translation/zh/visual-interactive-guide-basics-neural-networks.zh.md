<div class="img-div" markdown="0">
    <img src="/images/NNs_2_variables.png" />
</div>

**更新**：第二部分已上线：[神经网络基础数学的可视化讲解](https://jalammar.github.io/feedforward-neural-networks-visual-interactive/)

## 引子
我并不是机器学习专家。我是软件工程师出身，此前几乎没接触过 AI。我一直想深入机器学习，却始终没找到"入口"。所以当 Google 在 2015 年 11 月开源 TensorFlow 时，我异常兴奋，觉得是时候跳进这个领域、开启学习之旅了。说得夸张一点，那一刻在我看来，就像普罗米修斯把火种从机器学习的奥林匹斯山带到了人间。我脑海里回响的一个念头是：当年 Google 的研究员发表 MapReduce 论文时，整个大数据领域和 Hadoop 这类技术因此被极大地加速了；而这一次不是一篇论文——是他们内部历经多年演进后真正在用的软件。



于是我开始尽我所能学习这个领域的基础，也意识到需要一份对零基础读者更友好的资料。这篇文章就是我的尝试。


## 从这里开始
举个简单的例子。你在帮一位想买房的朋友参谋：一套 2000 平方英尺（约 185 平方米）的房子报价 40 万美元。这价格算好吗？

没有参照系就很难判断。于是你去问在那同几个社区买过房的朋友，最终得到三个数据点：



 | 面积（平方英尺）(x) | 价格 (y) |
 | --- | --- |
 | 2,104 | 399,900 |
 | 1,600 | 329,900 |
 | 2,400 | 369,000 |

就我个人而言，第一反应是算出平均每平方英尺的单价。结果是每平方英尺 180 美元。

欢迎来到你的第一个神经网络！它虽然还没到 Siri 的水平，但你现在已经认识了最基本的构件。它长这样：


<div class="img-div" markdown="0">
    <img src="/images/simple_NN_1.png" />
</div>

这类图展示了网络的结构，以及它如何计算预测。计算从左侧的输入节点开始。输入值向右流动，乘上权重，结果就是我们的输出。

2,000 平方英尺乘 180，得到 36 万美元——在这一层，事情就这么简单。计算预测只是一次乘法。但在那之前，我们需要想好要乘的那个权重。这里我们从平均值出发；之后会介绍更好的算法，能随着输入变多、模型变复杂而扩展。寻找权重就是我们的"训练"阶段。所以当你听说有人在"训练"神经网络时，请记住：那正是他们在为输入相乘寻找权重的过程。

### 预测
我们继续用例子说明。把面积乘以 180 美元，就得到预测价格：

<div class="img-div" markdown="0">
    <img src="/images/NNs_formula_no_bias.png" />
</div>

### 引入偏置
再来看一个"小插曲"。假设你决定把上面的公式画在图上。X 轴是面积，Y 轴是价格：

<div class="notice">原站此处是交互图：拖动滑块调整斜率（每平方英尺价格），观察直线的变化。静态译文页无法呈现，请到<a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/#introducing-bias">原站交互页面</a>体验。</div>

把上面三个房子的数据点画到图上，我们马上发现问题：

<div class="notice">原站此处是交互图：三个数据点与可调整的直线叠在一起，可以点按钮用最小二乘法拟合最优直线。静态译文页无法呈现，请到<a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/#introducing-bias">原站交互页面</a>体验。</div>

这条"每平方英尺 180 美元"的直线没有穿过这些点！对其中两套，它给出的预测明显低于市场价。如果直线能平移一些，让它们都落在线上，或者至少更靠近，会好得多。

我们往公式里加一项来实现这一点。在直线公式里加的这个值 b，我们叫它"偏置"。加上它之后，神经网络长这样：

<div class="img-div" markdown="0">
    <img src="/images/NNs_bias.png" />
</div>

我们把 b 称为偏置（bias，有些资料也叫 intercept）。它的作用只是把整条线平移。

可以把结论推广：一个一输入、一输出（*剧透警告：*也没有隐藏层）的神经网络，长这样：

<div class="img-div" markdown="0">
    <img src="/images/NNs_bias_2.png" />
</div>
在这张图里，W 和 b 是训练过程中求出的值；X 是我们代入公式的输入（例子里是房屋面积，单位平方英尺）；Y 是预测的价格。

加上偏置后，预测公式变成：

<div class="img-div" markdown="0">
    <img src="/images/NNs_formula_ex.png" />
</div>

现在，"训练"就是要找到能最好地穿过数据点的权重 w 和偏置 b。

### 引入误差函数
怎样才算"最好地穿过数据点"？需要给出明确定义。先写一个更紧凑的公式：

<div class="img-div" markdown="0">
    <img src="/images/NNs_formula.png" />
</div>

我们用下标标注 w1 和 w0，是为了与下一节的惯例保持一致。

现在为三个数据点分别计算 y 值，与真实价格比较，就能量化"我们错了多少"。我们用误差函数来做这件事。

先定义一个衡量"预测值偏离真实值多少"的函数。一个常用选择是"平方和误差"：

<div class="img-div" markdown="0">
    <img src="/images/NNs_formula_two_variables.png" />
</div>

注意这里的下标是数据样本编号，不是特征编号。计算过程是这样：

<div class="notice">原站此处是交互表格：每个数据点的预测值、误差与平方误差逐项计算。静态译文页无法呈现，请到<a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/#introducing-the-error-function">原站交互页面</a>体验。</div>

这就是我们现在对"做得更好"的定义——误差更小的模型就是更好的模型。误差的度量方式是：对数据集中每个点，把"真实值与预测值之差的平方"取平均。这就是[均方误差（Mean Square Error）](http://mste.illinois.edu/patel/amar430/meansquare.html)。用它来指导模型训练，它就是我们的**损失函数（loss function）**（也叫**代价函数，cost function**）。

既然"什么样的模型算更好"已经有了一把尺子，我们不妨再多试几个权重取值，跟前面那个"平均值"选择比较一下：

![](/images/lines_and_errors_animated.gif)

<div class="img-div" markdown="0">
单靠调权重已经没什么可改进的了；但如果加上偏置，就能找到让模型变好的取值。
</div>

## 训练你的神经网络 <a name="train_your_dragon" href="#train_your_dragon">#</a>
要不要亲手试试训练我们这个小玩具神经网络？拨动权重和偏置的滑块，把损失函数最小化。你能把误差降到 799 以下吗？

<div class="notice">原站此处是交互练习：两个滑块（Weight、Bias）+ 实时误差读数 + 损失曲线与拟合图。静态译文页无法呈现，请到<a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/#train_your_dragon">原站交互页面</a>体验。</div>




## 自动化 <a name="automation" href="#automation">#</a>
恭喜你手工训练出了第一个神经网络！接下来看看如何把训练过程自动化。下面这个例子里多了类似自动驾驶的功能——"GD Step"按钮。它们用一种叫"梯度下降（Gradient Descent）"的算法，尝试一步步逼近使损失函数最小的权重和偏置取值。

<div class="notice">原站此处是交互实验区：GD Step / 10 GD Steps / 100 GD Steps 按钮 + 滑块 + 误差曲线 + (权重,偏置) 误差热力图。静态译文页无法呈现，请到<a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/#automation">原站交互页面</a>体验。</div>

新增的两张图是为了帮你在拨动参数（权重与偏置）时跟踪误差的变化。跟踪误差很重要，因为训练过程的全部目标就是尽量把误差降下来。

梯度下降怎么知道下一步该往哪走？靠微积分。你看，在我们知道要最小化的函数（损失函数：所有数据点上 (y_ - y)² 的平均值）以及它当前的输入（当前的权重与偏置）的情况下，损失函数的导数会告诉我们该把 W 和 b 往哪个方向推，才能让误差更小。

关于梯度下降以及如何用它计算新的权重和偏置，可以在 Coursera [机器学习](https://www.coursera.org/learn/machine-learning)课程的前几讲深入了解。

## 从一到二：加入第二个特征 <a name="two" href="#two">#</a>

房子的价格只取决于面积吗？显然影响因素很多。我们再加一个变量，看看神经网络如何随之调整。

假设你的朋友做了更多功课，找到了更多数据点，还弄清了每套房有几个卫生间：

 | 面积（平方英尺）(x1) | 卫生间数 (x2) | 价格 (y)
 | --- | --- | --- |
 | 2,104 |  3 | 399,900 |
 | 1,600 |  3 | 329,900 |
 | 2,400 |  3 | 369,000 |
 | 1,416 |	2 | 232,000 |
 | 3,000 |	4 | 539,900 |
 | 1,985 |	4 | 299,900 |
 | 1,534 |	3 | 314,900 |
 | 1,427 |	3 | 198,999 |
 | 1,380 |	3 | 212,000 |
 | 1,494 |	3 | 242,500 |


我们有两个输入特征的神经网络长这样：


<div class="img-div" markdown="0">
    <img src="/images/NNs_2_variables.png" />
</div>



现在要找两个权重（每个输入一个）和一个偏置，才能构成新模型。

计算 Y 的公式变成：



<div class="img-div" markdown="0">
    <img src="/images/NNs_formula_two_variables.png" />
</div>



可是怎么求 w1 和 w2？这比只找一个权重要棘手。多一个卫生间，对房价预测的影响到底该有多大？

来试试找到正确的权重和偏置吧。你会从这个起点感受到：随着输入增多，问题开始变复杂——我们不再能用一眼看尽的二维图形来可视化模型了，主要只能靠观察误差数值随参数调整的变化。

<div class="notice">原站此处是双特征训练实验区：两个权重滑块 + 偏置滑块 + GD 按钮 + 误差曲线（特征变多后不再画拟合图，只跟踪误差）。静态译文页无法呈现，请到<a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/#two">原站交互页面</a>体验。</div>

我们可靠的梯度下降再次登场帮忙。它在帮我们找到正确的权重与偏置这件事上依然可靠。

## 特征
现在你见过一特征、二特征的神经网络，大致能推想出如何继续添加特征并用它们算预测。权重数量会持续增长，而我们每加一个特征，梯度下降的实现也要相应改造，才能更新新特征对应的新权重。


这里要强调：我们并不会把关于样本的一切信息都一股脑喂给网络，喂给模型哪些特征必须有所取舍。特征选择/特征处理是一门有自己一整套最佳实践与考量的学科。想看"检查数据集、挑选喂给预测模型的特征"的完整过程，可以读 [A Journey Through Titanic](https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic)。那是 [Omar EL Gabry](https://twitter.com/Omar_ElGabry) 讲解他解决 Kaggle 泰坦尼克号挑战全过程的 notebook。Kaggle 提供了泰坦尼克号的乘客名单，包括姓名、性别、年龄、舱位、是否生还等数据；挑战是根据这些信息建一个预测某人生还与否的模型。

## 分类

继续改造我们的例子。假设朋友这次给你一张房子清单，并且标注了哪些房子的面积和卫生间数量合她心意：

 | 面积（平方英尺）(x1) | 卫生间数 (x2) | 标签 (y) |
 | --- | --- | --- |
 | 2,104 |  3 | Good |
 | 1,600 |  3 | Good |
 | 2,400 |  3 | Good |
 | 1,416 |	2 | Bad |
 | 3,000 |	4 | Bad |
 | 1,985 |	4 | Good |
 | 1,534 |	3 | Bad |
 | 1,427 |	3 | Good |
 | 1,380 |	3 | Good |
 | 1,494 |	3 | Good |

她要你用这份清单建一个模型，输入面积和卫生间数，预测她会不会喜欢某套房子；模型建好后，她会用来评估很多其他房子。流程里还多了一个变化：她手里另有一份标注过的 10 套房子清单，但不给你看——那份是用来在你训练完模型后评估它的，从而检验模型真正掌握的，是不是让她喜欢这套房子的条件。

到目前为止我们玩的神经网络都在做"回归"——输出一个"连续"值（可以是 4，或 100.6，或 2143.342343）。但实践中，神经网络更常用于"分类"问题：输出必须是离散值集合（"类别"）中的一个，比如 "Good" 或 "Bad"。具体做起来，模型不会直接吐出 "good" 或 "bad"，而是说它有 75% 的把握认为这栋房子是 "Good"。



<div class="img-div" markdown="0">
    <img src="/images/android_tensorflow_classifier_results.jpg" />
    我在[另一篇旧文](https://jalammar.github.io/Supercharging-android-apps-using-tensorflow/)里谈过的 TensorFlow Android 应用，就是一个实际分类模型的好例子。

</div>



把已见的网络改造成分类网络的一种办法，是让它输出两个值——每个类别一个（现在类别是 "good" 和 "bad"），然后把这两个值经过一个叫"[softmax](https://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/)"的运算。softmax 的输出是每个类别的概率。比如网络某层输出 "good" 为 2、"bad" 为 4，把 [2, 4] 喂给 softmax 会返回 [0.11, 0.88]——也就是网络 88% 确定输入的房子是 "bad"，朋友不会喜欢。


Softmax 接收一个数组，返回等长数组。注意输出全为正且总和为 1——输出概率值时这非常有用。再注意 4 虽然只是 2 的两倍，其概率却不是两倍、而是 2 的八倍。这种放大输出差距的性质对训练过程很有帮助。


 |  | 输出 |
 | --- | --- |
 | softmax([ 1 ]) | [ 1 ] |
 | softmax([ 1, 1 ]) | [ 0.5, 0.5 ] |
 | softmax([ 0, 1 ]) | [ 0.26,  0.73 ] |
 | softmax([ 2, 4 ]) | [ 0.11,  0.88 ] |
 | softmax([ 5, 10 ]) | [ 0.007,  0.993 ] |
 | softmax([ -1, 0, 1 ]) | [ 0.09,  0.24,  0.66 ] |
 | softmax([ 1, 2, 4 ]) | [ 0.04,  0.11,  0.84 ] |

如最后两行所示，softmax 可扩展到任意数量的输入。所以如果朋友再添第三个标签（比如"不错，但我得把一间房租出去"），softmax 照样能适配这种变化。

试着改变特征数量（x1、x2、x3……可以是面积、卫生间数、价格、离学校或公司的距离……）和类别数量（y1、y2、y3……可以是"太贵"、"划算"、"出租一间就行"、"太小"），观察网络形状的变化：

<div class="notice">原站此处是可调节特征数/类别数、实时绘制网络形状的交互组件。静态译文页无法呈现，请到<a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/#classification">原站交互页面</a>体验。</div>

用 TensorFlow 创建并训练这种网络的示例，见我随本文写的这个 [notebook](https://github.com/jalammar/simpleTensorFlowClassificationExample/blob/master/Basic%20Classification%20Example%20with%20TensorFlow.ipynb)。


## 真正的动机
如果你读到这里，我得再交代一个写这篇文章的动机。本文其实是 TensorFlow 入门教程的"更缓坡版"。如果你现在直接去读 [MNIST For ML Beginners](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html)，遇到下面这张图：


<div class="img-div" markdown="0">
    <img src="/images/softmax-regression-scalargraph.png" />
    我写这篇文章，就是为了让没有机器学习经验的读者能看懂 TensorFlow 入门教程里的这张图——所以我刻意模仿了它的视觉风格。
</div>


我希望你现在已做好了准备，理解这套系统及其工作原理。想动手写代码，可以接着看入门[教程](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html)，教一个神经网络识别手写数字。

接下来你应当继续学习本文这些概念背后的理论与数学。现在值得追问的好问题包括：

 * 还有哪些代价函数？分别更适合哪些应用？
 * 用梯度下降实际计算新权重的算法是什么？
 * 在你已经熟悉的领域里，机器学习能做什么？把这个新法术加入你的法术书，你能施展什么新的魔法？


很好的学习资源：

 * Coursera 上 [Andrew Ng](https://twitter.com/AndrewYNg) 的[机器学习](https://www.coursera.org/learn/machine-learning)课程——我自己的起点，从回归讲到分类再到神经网络。
 * Coursera 上 [Geoffrey Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) 的 [Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks)——更聚焦神经网络及其视觉应用。
 * 斯坦福 [CS231n: Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/watch?v=g-PvXUjD6qg&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA)（[Andrej Karpathy](https://twitter.com/karpathy)）——可以看到一些进阶概念，以及深度神经网络在视觉识别上的前沿。
 * [The Neural Network Zoo](http://www.asimovinstitute.org/neural-network-zoo/)——了解各类神经网络架构的绝佳资料。

## 致谢

感谢 [Yasmine Alfouzan](https://www.linkedin.com/in/yasmine-alfouzan-b05ba317)、[Ammar Alammar](https://twitter.com/a3ammar)、[Khalid Alnuaim](https://www.linkedin.com/in/khalidalnuaim)、[Fahad Alhazmi](https://twitter.com/fahd09)、[Mazen Melibari](https://www.linkedin.com/in/mkhdev) 和 [Hadeel Al-Negheimish](https://www.linkedin.com/in/hadeel-al-negheimish-4a73abb3) 审阅本文早期版本提供的帮助。

如有任何勘误或反馈，请在 [Twitter](https://twitter.com/JayAlammar) 联系我。
