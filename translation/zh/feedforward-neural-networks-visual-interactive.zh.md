在[上一篇文章中，我们探讨了神经网络的基本概念](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)。现在再举一个例子，借它探索神经网络预测中涉及的一些基本数学思想。

<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/titanic_nn_calculation.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>




如果你当时就在泰坦尼克号上，船沉的时候你能活下来吗？我们来建一个模型，预测一个人的生还概率。

这将是一个在[上一篇文章](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)基础上讨论的神经网络模型，但因为加入了隐藏层和激活函数，它的预测精度会更高。

这次使用的数据集是 Kaggle 的[泰坦尼克号乘客名单](https://www.kaggle.com/c/titanic/data)。它列出了乘客的姓名和其他信息，并标明每位乘客是否在这场沉船事故中生还。

原始数据长这样：

<div class="titanic-dataset">

| PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked|
|--- | ---| --- | ---| ---| ---| ---| ---| ---| ---| ---| ---|
|1	|0|	3|	Braund, Mr. Owen Harris|	male	|22.0	|1	|0	|A/5 21171	|7.2500	| NaN	|S|
|2	|1|	1|	Cumings, Mrs. John Bradley (Florence Briggs Th...	| female	|38.0	|1	|0	|PC 17599	|71.2833	|C85	|C|
|3	|1|	3|	Heikkinen, Miss. Laina	|female	|26.0	|0	|0	|STON/O2. 3101282	|7.9250	|NaN	|S|

</div>

大部分列我们暂时用不上，只用 sex（性别）和 age（年龄）两列做特征，用是否生还作为要预测的标签。



<div class="two_variables">
  <table>
    <thead>
      <tr>
        <th>年龄</th>
        <th>性别</th>
        <th>是否生还？</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>22</td>
        <td>0</td>
        <td>0</td>
      </tr>
      <tr>
        <td>38</td>
        <td>1</td>
        <td>1</td>
      </tr>
      <tr>
        <td>26</td>
        <td>1</td>
        <td>1</td>
      </tr>
      <tr>
        <td colspan="3">…… 共 891 行</td>
      </tr>
    </tbody>
  </table>
</div>

我们要建一个网络，预测某位乘客是否生还。

神经网络的输入必须是数值，所以 sex 列要做转换——男性记 0，女性记 1。你会注意到数据集在生还列上已经用了类似编码——生还记 1，遇难记 0。

能用来训练这个预测的最简单神经网络长这样：


<div class="img-div" markdown="0">
    <img src="/images/two-input-one-output-sigmoid-network.png" alt="两个输入、一个输出、输出层用 sigmoid 激活的神经网络"/>
    计算一次预测，就是把"age"和"sex"的取值填进去，计算便从左向右流动。
    不过在拿它做预测之前，得先跑一遍"训练"过程，求出权重（<span class="weight-node-text">w</span>）和偏置（<span class="bias-node-text">b</span>）的值。

<br />
注：相比上一篇文章，我们稍微调整了网络的画法——偏置节点更常见的表示方式就是图中这样。
</div>



下面回顾这个网络的组成要素及其工作方式：

<div class="row neuron-expo vertical-align">
<div class="col-sm-4 small-column" >
<img src="/images/input-neuron.png" alt="输入神经元"/>
</div>
<div class="col-sm-8 side-column">

输入神经元是填入输入值的地方（比如一个人的年龄），计算从这里开始。向外的连接以及整张图其余部分，告诉我们要为算出预测还需要做哪些计算。

</div>
</div>






<div class="row neuron-expo vertical-align">
<div class="col-sm-4 small-column" >
<img src="/images/weight.png" alt="带权重的连接"/>
</div>
<div class="col-sm-8 side-column">

如果一条连接带权重，数值经过它时就要乘以这个权重。

    connection_output = weight * connection_input
</div>
</div>




<div class="row neuron-expo vertical-align">
<div class="col-sm-4 small-column" markdown="0">
    <img src="/images/neuron.png" alt="神经元"/>
</div>
<div class="col-sm-8 side-column">
如果神经元有输入，它把所有输入的值加起来，沿输出连接发送这个和。

    node_output = input_1 + input_2
</div>
</div>





### Sigmoid <a name="sigmoid" href="#sigmoid">#</a>
<div class="row neuron-expo vertical-align">
<div class="col-sm-4 small-column">

<img src="/images/sigmoid-neuron.png" alt="sigmoid 神经元"/>


</div>
<div class="col-sm-8 side-column">

要把网络的计算结果变成一个 0 到 1 之间的概率值，必须把输出层的数值通过一个 "sigmoid" 公式。Sigmoid 按一条特定的曲线把神经元的输出值压缩到 0 和 1 之间。

f(x) = 1 / (1 + e^(-x))


其中 e 是数学常数，约等于 2.71828。



    def sigmoid(x):
        return 1/(1 + math.exp(-x))

    output = sigmoid(value)


</div>
</div>




### Sigmoid 可视化 <a name="sigmoid-visualization" href="#sigmoid-visualization">#</a>
<div class="row neuron-expo vertical-align">
<div class="col-sm-4 small-column" >

<img src="/images/sigmoid-neuron.png" />

</div>
<div class="col-sm-8 side-column">

<div class="notice">原站此处是一个可交互的滑块与图像（拖动滑块观察 sigmoid 如何变换各种数值，输入范围 -20 到 20）。静态译文页没有交互组件，请到<a href="https://jalammar.github.io/feedforward-neural-networks-visual-interactive/#sigmoid-visualization">原站交互页面</a>体验。</div>

f(x) = 1 / (1 + e^(-x))

</div>
</div>




<div class="row neuron-expo vertical-align">
<div class="col-sm-4 small-column" >
<img src="/images/two-input-one-output-sigmoid-network.png" alt="带权重的网络"/>
</div>
<div class="col-sm-8 side-column">

把这一切合起来，用这个浅层网络算一次预测就是：

    def sigmoid(x):
        return 1/(1 + math.exp(-x))

    def calculate_prediction(age, sex, weight_1, weight_2, bias):

        # Multiply the inputs by their weights, sum the results up
        layer_2_node = age * weight_1 + sex * weight_2 + 1 * bias

        prediction = sigmoid(layer_2_node)
        return prediction


</div>
</div>

既然知道了网络结构，就可以用 891 行数据集的前 600 行，通过[梯度下降]（gradient descent）来训练它。训练过程不在本文展开，那是另一个话题。现在只希望你先对"训练好的网络如何算预测"建立足够的直觉。这个直觉建立起来后，我们以后再用一篇文章讲训练。


训练过程给出了下面这些值（准确率 73.20%）：

    weight_1 =   -0.016852 # Associated with "Age"
    weight_2 =   0.704039  # Associated with "Sex" (where male is 0, female is 1)
    bias =       -0.116309

直观地说，权重表示对应特征对预测的贡献有多大——年龄越小，生还机会越好（年龄越大乘以负的权重，得到的负数越大）；如果乘客是女性，生还机会提升得更多。


### 预测计算 <a name="prediction-calculation" href="#prediction-calculation">#</a>
训练好的网络长这样：
（原站可把鼠标悬停/点击表格中的数值查看各自的预测）


<div class="notice">原站此处是由 JavaScript 绘制的交互表格与网络图（悬停某一行数据会在网络图上看到数值流动）。静态译文页无法呈现，请到<a href="https://jalammar.github.io/feedforward-neural-networks-visual-interactive/#prediction-calculation">原站交互页面</a>体验。</div>


73.20% 的准确率并不出色。这类问题正适合加一个隐藏层：隐藏层能提升模型的表达能力，从而表示更复杂的预测函数，取得更好的效果（[《Deep Learning》第 5 章第 113 页](http://www.deeplearningbook.org/contents/ml.html)）。


<div class="row neuron-expo vertical-align">
<div class="col-sm-4 small-column" >

<img src="/images/neuron_with_activation.png" alt="带激活函数的加权神经元"/>

</div>
<div class="col-sm-8 side-column">

对加权输出施加某些数学函数往往很有用。这类函数叫"激活函数"，因为历史上它们把神经元的输出转换为 1（激活）或 0（关闭）。


    def activation_function(x):
        # Do something to the value
        ...

    weighted_sum = weight * (input_1 + input_2)
    output = activation_function(weighted_sum)

激活函数对隐藏层至关重要。没有它们，深层网络并不比浅层线性网络强。各类激活函数的综述请看 [Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/) 中"常用激活函数"一节。
</div>
</div>


### ReLU <a name="relu" href="#relu">#</a>
<div class="row neuron-expo vertical-align">
<div class="col-sm-4 small-column" >

<img src="/images/relu.png" alt="带激活的加权神经元"/>

</div>
<div class="col-sm-8 side-column">

激活函数的主流选择叫 ReLU：输入为负就返回 0，否则返回数字本身。就这么简单！

f(x) = max(0, x)

    # Naive scalar relu implementation. In the real world, most calculations are done on vectors
    def relu(x):
        if x < 0:
            return 0
        else:
            return x


    output = relu(value)
</div>
</div>




### ReLU 可视化 <a name="relu-visualization" href="#relu-visualization">#</a>


<div class="row neuron-expo vertical-align">
<div class="col-sm-4 small-column" >

<img src="/images/relu.png" />

</div>
<div class="col-sm-8 side-column">

<div class="notice">原站此处同样是可交互滑块与图像，用于观察 ReLU 如何变换各种数值。静态译文页无法呈现，请到<a href="https://jalammar.github.io/feedforward-neural-networks-visual-interactive/#relu-visualization">原站交互页面</a>体验。</div>

f(x) = max(0, x)

</div>
</div>



## 结语
这篇文章在草稿箱里躺了一年多。我曾尝试在这里之后可视化一个更深的网络，但终究没有完成。希望你喜欢这篇内容。有任何反馈，请到 [Twitter @JayAlammar](https://twitter.com/JayAlammar) 告诉我。
