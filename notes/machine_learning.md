# 周志华《机器学习》读书笔记

## 于士尧同学的课程学习文档（样例）
写这个文档是为了加深印象，理清这本书的框架，以后方便查阅。



## 第一章 绪论

本章主要在解释机器学习的相关术语。

**研究的主要内容**：研究从数据中产生“模型”的算法，即 “学习算法” 。有了学习算法，我们提供经验数据，它可以基于数据产生模型；在面对新情况时，模型会给我们提供相应的判断。

**样本**：数据集的每条记录是关于一件事件或对象的描述，也称为一个 “示例” 。

**属性**：反映事件或对象在某方面的表现或性质的事项，例如“性别”，“种族”，称为 “属性” 或 *“特征”* 。属性张成的空间称为 “属性空间” 、 “样本空间” ，或 *“输入空间”* 。每个样本都可以在这个空间中找到自己的坐标位置，并且因为空间中的每个点都对应一个坐标向量，因此也可以把一个样本称作一个 *“特征向量”* 。

**训练集**：从数据中学得模型的过程称为 *“学习”* 或 *“训练”* ，训练过程中使用的数据叫 “训练数据” ，其中每个样本叫做一个 “训练样本”，训练样本组成的集合叫做 *”训练集”* 。

**假设**：关于数据的某种潜在规律，而这种潜在的规律本身叫 *"ground-truth"* 。

**泛化**：学得模型适用于新样本的能力，称为 “泛化” 能力。



## 第二章 模型评估与选择

怎样刻画模型实际预测输出与样本的真实输出之间的之间的差异呢？我们把这种差异叫做  “误差”，学习器在训练集上的误差称为 “训练误差” 或 “经验误差”。由于事先并不知道新样本长什么样，实际能做的只有努力使经验误差最小化。

### 2.1 过拟合

事实证明一个在训练集上表现很好的分类器泛化能力不一定好，原因可能是模型的学习能力过于强大。我们需要从训练样本中尽可能地学出适用于所有潜在样本的 “普遍规律” ，这样才能在遇到新样本时做出正确的判断。当学习器把训练样本学的 “太好” 的时候，很可能把训练样本本身的一些特点当作所有潜在样本都会有的一般性质，导致泛化性能下降，这种现象叫做 *“过拟合”* 。

过拟合不可以彻底避免的。如果过拟合可以被完全消除，也就是说模型不存在泛化能力下降的问题，那么我们在当前训练集下将模型训练到极致（反正不会产生泛化性能下降的问题），只需要经验误差最小化就能获得最优模型。机器学习面临的问题通常都是NP-hard问题，上述的训练过程相当于在有限的时间里得出了一个最优解，这是不可能的。

由于过拟合的存在，不能直接使用训练误差作为标准，我们也无法直接获得泛化误差。需要一些方法进行模型的评估和选择。

### 2.2 评估方法

我们需要一些与训练集互斥的样本作验证集，用学习器在验证集上的误差近似作为泛化误差，有三种主要的划分验证集的方法。

#### 2.2.1 留出法 (hold-out)

“留出法”直接将数据集 $D$ 划分成两个**互斥**的集合，一个作为训练集 $S$ ，另一个作为验证集 $T$ ，即 $D=S∪T$，$S∩T=\varnothing$。在 $S$ 上训练出模型后，用 $T$ ​来评估其测试误差，作为对泛化误差的估计。

注意：

**(a)** 验证集合和训练集合要尽可能保持数据分布的一致性，例如要保证正反样本的比例不变（这也是一种导致过拟合的原因）

**(b)** 给定了训练/验证集合的样本比例后，仍存在多种对 $D$ 的划分方式。例如，可以把前m个正例放到训练集中，也可以把后m个正例放入训练集中。不同的划分会产生不同的测试/验证集，相应的结果也会有差别。一般要进行若干次随机划分，重复进行实验评估后取平均值作为留出法的评估结果

**(c)** 训练/验证集合的大小比例问题：验证集合过小，会导致测评结果的方差变大，训练集合过小，会导致偏差过大。一般使用的都是$\frac{2}{3}$~$\frac{4}{5}$的样本用于训练

#### 2.2.2 交叉验证法 (k-fold)

“交叉验证法”先将数据集 $D$ 划分成 $k$ 个大小相似的互斥子集，每个自己都尽可能保证数据分布的一致性。然后每次用 $k-1$ 个子集的并集作为训练集，余下的子集作为验证集。这样就可以得到 $k$ 组训练/验证集，从而可以进行 $k$ 次训练和测试，最终返回的是这 $k$ 个测试结果的均值。

和留出法相似，为了减小因为样本划分不同而引入的差别，k-fold 通常要使用不同的划分重复 $p$ 次，最终的结果是这 $p$ 次 $k$ 折交叉验证结果的均值。10 次 10-$fold$ 的训练/测试次数和100次留出法一致。

**留一法 (Leave-One-Out)**

k-fold的特例（ k=m，m 是数据集大小），我们每次使用 m-1 个样例进行训练，1 个样例进行测试。这样用训练出的模型和直接使用全部数据集 $D$ 训练出的模型非常接近，但当数据量比较大时，例如数据集包含 100 万个样例，使用留一法意味着我们要训练 100 万个模型，这显然是不现实的。 

#### 2.2.3 自助法 (bootstrapping)

“自助法” 每次从数据集 $D$ 中抽取一个样本加入数据集 $D'$ 中，然后再将该样本放回到原数据集 $D$ 中，即 $D$ 中的样本可以被重复抽取。这样，$D$ 中的一部分样本会被多次抽到，而另一部分样本从未被抽到，没被抽到的比例约为36.8%，所以我们可以将 $D'$ 作为训练集，将 $D$  \ $D'$（$D$ 与 $D'$ 的差集）作为验证集。

自助法适用于数据集较小，难以划分训练\验证集的情况。

#### 2.2.4 调参

机器学习涉及两种参数，一类是算法的参数，也叫做 “超参数”，另一种是模型的参数，数目可能很多，大模型可能涉及上百亿个参数。前者通常是由人工设定多个参数候选值后产生模型，为每个参数选择一个范围和步长，例如在[0, 0.2]范围内以0.05为步长；后者则是通过学习来产生多个候选模型（例如神经网络在不同轮数停止训练）。

在学习算法和参数选定之后，要重新使用完整数据集对模型进行训练，因为每次只用了一部分数据训练模型。

### 2.3 性能度量

这一章主要讲模型的测试指标。给定样例集 $D=\{(\boldsymbol{x}_1,y_1),(\boldsymbol{x}_2,y_2),…,(\boldsymbol{x}_m,y_m)\}$，其中 $y_i$ 是示例 $\boldsymbol{x}_i$ 的真实标记。要评估学习器 $f$的性能，需要把 $f(\boldsymbol{x})$ 与真实标记 $y$ 进行比较。

对于回归任务，最常用的性能度量是 “均方误差” (mean square error)，
$$
E(f;D)=\frac{1}{m}\sum_{i=1}^{m}{(f(\boldsymbol{x}_i)-y_i)^2}
$$
对于分类任务，性能度量的选择更多。

#### 2.3.1 错误率和精度

这是最常用的两种性能度量，适用于二分类任务和多分类任务。错误率是分类错误的样本数占样本总数的比例，精度是分类正确的样本数占样本总数的比例。

错误率定义为：
$$
E(f ; D)=\frac{1}{m} \sum_{i=1}^m \mathbb{I}\left(f\left(\boldsymbol{x}_i\right) \neq y_i\right).
$$
精度定义为：
$$
\begin{aligned} \operatorname{acc}(f ; D) & =\frac{1}{m} \sum_{i=1}^m \mathbb{I}\left(f\left(\boldsymbol{x}_i\right)=y_i\right) \\ & =1-E(f ; D) .\end{aligned}
$$

#### 2.3.2 查准率、查全率和F1度量

|              |   预测结果   |   预测结果   |
| :----------: | :----------: | :----------: |
| **真实情况** |     正例     |     反例     |
|     正例     | TP（真正例） | FN（假反例） |
|     反例     | FP（假正例） | TN（真反例） |

**查准率P（precision）**$P=\frac{T P}{T P+F P}$ 所有预测的正例中，真实的正例有多少比例，例如所有预测的好瓜有真的好瓜的比例

**查全率R（recall）**$R=\frac{T P}{T P+F N}$ 所有的正例有多少被查出来，例如好瓜有多少被找出来了

但是分别看这两个还是不方便，F1度量把他们统一起来，取他们的调和平均。
$$
\frac{1}{F 1}=\frac{1}{2} \cdot\left(\frac{1}{P}+\frac{1}{R}\right)
$$

$$
\begin{aligned} \mathrm{F} 1 & =\frac{2 \times \mathrm{P} \times \mathrm{R}}{\mathrm{P}+\mathrm{R}}=\frac{2 \times \mathrm{TP}}{\text { 样例总数 }+\mathrm{TP}-\mathrm{TN}}\end{aligned}
$$

#### 2.3.4 ROC和AUC

很多时候，学习器为测试样本产生一个实值或概率预测，然后与分类阈值比较，大于阈值则为正类，否则为反类。需要将样本排序，以截断点位置分类。这个排序本身的质量应该怎样考察呢？

**（1）**真正例率TPR：正例里比预测为正例的比率
$$
\begin{aligned} & \mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}\end{aligned}
$$
**（2）**假正例率FPR：反例里被预测为正例的比率
$$
\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{TN}+\mathrm{FP}}
$$
ROC曲线的横坐标是FPR，纵坐标是TPR

**绘图过程**：给定 $m^+$ 个正例和 $m^-$ 个反例，根据学习器的预测结果对样例进行排序。起初把阈值设为最大，此时所有样例均预测为反例，$TPR$ 和 $FPR$ 均为0，在坐标 $(0,0)$ 处标记一个点。然后依次把每个样例设置为阈值，也就是依次将每个样本划分为正例。设之前的点坐标为$(x,y)$，若当前样本为真正例，则对应标记点的坐标为$(x,y+\frac{1}{m^+})$；当前样本为假正例的话，对应的坐标记为$(x+\frac{1}{m^-},y)$，然后用线段连接起来。

比较两个$ROC$曲线时，若一个曲线能把另一个曲线“包住”的话，可以断言前者的性能强于后者。若两曲线存在交叉点，难以比较，则比较**曲线下的面积$AUC$** (Area Under ROC Curve)，面积大的性能好。
$$
\mathrm{AUC}=\frac{1}{2} \sum_{\mathrm{i}=1}^{\mathrm{m}-1}\left(\mathrm{x}_{\mathrm{i}+1}-\mathrm{x}_{\mathrm{i}}\right)\left(\mathrm{y}_{\mathrm{i}}+\mathrm{y}_{\mathrm{i}+1}\right)
$$
$AUC$就是各个小梯形面积的和。

当代价不一致时不能使用$ROC$比较性能，比如把正例分成反例和把反例分成正例的代价不同，要绘制代价曲线。

**代价曲线**：用来反映学习器的期望总体代价的曲线，以取值为[0,1]正例概率代价为横轴，纵轴以取值[0,1]的归一化代价，通过$ROC$曲线上的每一个点来计算出$FPR$和$FNR$，在代价平面上化出$(0,FPR)$到$(1,FNR)$的线段，**所有线段下的面积交集就是总体代价期望**。

### 2.4 比较检验

**统计假设检验（hypothesis test）**若在测试集上观察到学习器A比B好，则A的泛化性能是否在统计意义上优于B，以及这个结论的把握有多大。

这牵扯到概率统计里的内容，需要时再查书。



## 第三章 线性模型

### 3.1 定义

$\boldsymbol{x}=(x_1;x_2;...;x_d)$，其中 $x_i$ 是 $\boldsymbol{x}$ 在第 $i$ 个属性上的取值。线性模型(linear model)试图学得一个通过属性的线性组合来进行预测的函数，即
$$
f(\boldsymbol{x})=w_1x_1+w_2x_2+...+w_dx_d+b,
$$
向量形式为
$$
f(\boldsymbol{x})=\boldsymbol{w}^T\boldsymbol{x}+b
$$
其中$\boldsymbol{w}=(w_1;w_2;...;w_d).$ 许多功能更强大的非线性模型可以在线性模型的基础上通过引入层级结构或高维映射而得到。

### 3.2 线性回归

给定一个只有一个属性大小为 $m$ 的数据集 $ D=\left\{\left(x_i, y_i\right)\right\}_{i=1}^m$。线性回归使用均方误差作为损失函数，基于均方误差最小化来进行模型求解的方法称为“最小二乘法”，$E_{(w,b)}=\sum_{i=1}^{m}(y_i-wx_i-b)^2$，我们需要找到一条直线使得所有样本到直线上的欧氏距离最小，即
$$
\begin{aligned}\left(w^*, b^*\right) & =\underset{(w, b)}{\arg \min } \sum_{i=1}^m\left(f\left(x_i\right)-y_i\right)^2 \\ & =\underset{(w, b)}{\arg \min } \sum_{i=1}^m\left(y_i-w x_i-b\right)^2 .\end{aligned}
$$
我们可将 $E_{(w,b)}$ 分别对 $w$ 和 $b$ 求导，得到
$$
\begin{aligned} & \frac{\partial E_{(w, b)}}{\partial w}=2\left(w \sum_{i=1}^m x_i^2-\sum_{i=1}^m\left(y_i-b\right) x_i\right), \\ & \frac{\partial E_{(w, b)}}{\partial b}=2\left(m b-\sum_{i=1}^m\left(y_i-w x_i\right)\right),\end{aligned}
$$
另其为零可以得到 $w$ 和 $b$ 的最优解。

对于有 $d$ 个属性的 $x$ ​的数据集同理，
$$
\mathbf{X}=\left(\begin{array}{ccccc}x_{11} & x_{12} & \ldots & x_{1 d} & 1 \\ x_{21} & x_{22} & \ldots & x_{2 d} & 1 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ x_{m 1} & x_{m 2} & \ldots & x_{m d} & 1\end{array}\right)=\left(\begin{array}{cc}\boldsymbol{x}_1^{\mathrm{T}} & 1 \\ \boldsymbol{x}_2^{\mathrm{T}} & 1 \\ \vdots & \vdots \\ \boldsymbol{x}_m^{\mathrm{T}} & 1\end{array}\right),
$$
把 $b$ 吸收进 $\boldsymbol{w}$，记为 $\hat{\boldsymbol{w}}=(\boldsymbol{w} ; b)$；把标记写成向量形式 $\boldsymbol{y}=\left(y_1 ; y_2 ; \ldots ; y_m\right)$

令 $E_{\hat{\boldsymbol{w}}}=(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})^{\mathrm{T}}(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})$，对 $\hat{\boldsymbol{w}}$ 求导可得
$$
\frac{\partial E_{\hat{\boldsymbol{w}}}}{\partial \hat{\boldsymbol{w}}}=2 \mathbf{X}^{\mathrm{T}}(\mathbf{X} \hat{\boldsymbol{w}}-\boldsymbol{y})
$$
当 $\mathbf{X}^{\mathrm{T}} \mathbf{X}$ 满秩或正定，令上式等于零可得
$$
\hat{\boldsymbol{w}}^*=\left(\mathbf{X}^{\mathrm{T}} \mathbf{X}\right)^{-1} \mathbf{X}^{\mathrm{T}} \boldsymbol{y}
$$
不满秩则需要引入正则项。

**广义线性模型**：考虑单调可微函数 $g(·)$，令
$$
y=g^{-1}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b\right)
$$
这样得到的模型称为 “广义线性模型” (generalized linear model), 其中函数 $g(\cdot)$ 称为 “联系函数” (link function)。

**对数几率回归**：
$$
ln{\frac{y}{1-y}}=\boldsymbol{w}^T\boldsymbol{x}+b
$$
**线性判别分析$LDA$**：

给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离；在对新样本进行分类时，将其投影到同样的这条直线上，再根据投影点的位置来确定新样本的类别。

给定数据集 $D=\left\{\left(\boldsymbol{x}_i, y_i\right)\right\}_{i=1}^m$，$y_i \in\{0,1\}$，令 $X_i 、 \boldsymbol{\mu}_i 、 \boldsymbol{\Sigma}_i$ 分别表示第 $i \in\{0,1\}$ 类示例的集合、均值向量、协方差矩阵。若将数据投影到直线 $\boldsymbol{w}$ 上，则两类样本的中心在直线上的投影分别为 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\mu}_0$ 和 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\mu}_1$；若将所有样本点都投影到直线上，则两类样本的协方差分别为 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\Sigma}_0 \boldsymbol{w}$ 和 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\Sigma}_1 \boldsymbol{w}$。由于直线是一维空间，因此 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\mu}_0 、 \boldsymbol{w}^{\mathrm{T}} \boldsymbol{\mu}_1 、 \boldsymbol{w}^{\mathrm{T}} \boldsymbol{\Sigma}_0 \boldsymbol{w}$ 和 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\Sigma}_1 \boldsymbol{w}$ 均为实数。

欲使同类样例的投影点尽可能接近，可以让同类样例投影点的协方差尽可能小，即 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\Sigma}_0 \boldsymbol{w}+\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\Sigma}_1 \boldsymbol{w}$ 尽可能小；而欲使异类样例的投影点尽可能远离，可以让类中心之间的距离尽可能大，即 $\left\|\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\mu}_0-\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\mu}_1\right\|_2^2$ 尽可能大。同时考虑二者，则可得到欲最大化的目标
$$
\begin{aligned}
J & =\frac{\left\|\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\mu}_0-\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\mu}_1\right\|_2^2}{\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\Sigma}_0 \boldsymbol{w}+\boldsymbol{w}^{\mathrm{T}} \boldsymbol{\Sigma}_1 \boldsymbol{w}} \\
& =\frac{\boldsymbol{w}^{\mathrm{T}}\left(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1\right)\left(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1\right)^{\mathrm{T}} \boldsymbol{w}}{\boldsymbol{w}^{\mathrm{T}}\left(\boldsymbol{\Sigma}_0+\boldsymbol{\Sigma}_1\right) \boldsymbol{w}} .
\end{aligned}
$$
定义 “类内散度矩阵”，表示第 $i$ 类样本距离样本中心的距离
$$
\begin{aligned}
\mathbf{S}_w & =\boldsymbol{\Sigma}_0+\boldsymbol{\Sigma}_1 \\
& =\sum_{\boldsymbol{x} \in X_0}\left(\boldsymbol{x}-\boldsymbol{\mu}_0\right)\left(\boldsymbol{x}-\boldsymbol{\mu}_0\right)^{\mathrm{T}}+\sum_{\boldsymbol{x} \in X_1}\left(\boldsymbol{x}-\boldsymbol{\mu}_1\right)\left(\boldsymbol{x}-\boldsymbol{\mu}_1\right)^{\mathrm{T}}
\end{aligned}
$$

以及 “类间散度矩阵”，两个样本中心间的距离
$$
\mathbf{S}_b=\left(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1\right)\left(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1\right)^{\mathrm{T}},
$$

则 $J$ 可重写为
$$
J=\frac{\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_b \boldsymbol{w}}{\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_w \boldsymbol{w}} .
$$

这就是 $LDA$ 欲最大化的目标, 即 $\mathbf{S}_b$ 与 $\mathbf{S}_w$ 的 “广义瑞利商” (generalized Rayleigh quotient)。

如何确定 $\boldsymbol{w}$ 呢？注意到上式的分子和分母都是关于 $\boldsymbol{w}$ 的二次项，因此 $J$ 的解与 $\boldsymbol{w}$ 的长度无关，只与其方向有关。不失一般性，令 $\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_w \boldsymbol{w}=1$（反正 $\boldsymbol{w}$ 的大小无所谓，为了方便计算，我们调整它的值使得 $\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_w \boldsymbol{w}=1$ ），则 $J$ 等价于
$$
\begin{array}{ll}
\min _{\boldsymbol{w}} & -\boldsymbol{w}^{\mathrm{T}} \mathbf{S}_b \boldsymbol{w} \\
\text { s.t. } & \boldsymbol{w}^{\mathrm{T}} \mathbf{S}_w \boldsymbol{w}=1 .
\end{array}
$$

由拉格朗日乘子法，上式等价于
$$
L=-\boldsymbol{w}^T\mathbf{S}_b\boldsymbol{w}+\lambda(\boldsymbol{w}^T\mathbf{S}_w\boldsymbol{w}-1)
$$
对 $\boldsymbol{w}$ 求导，得
$$
=-(\mathbf{S}_b+\mathbf{S}_b^T)\boldsymbol{w}+\lambda(\mathbf{S}_w+\mathbf{S}_w^T)\boldsymbol{w}
$$


$\mathbf{S}_b$ 和 $\mathbf{S}_w$ 都是对称矩阵，所以
$$
\\=-2\mathbf{S}_b\boldsymbol{w}+\lambda·2\mathbf{S}_w\boldsymbol{w}
$$
令其等于零，有
$$
\mathbf{S}_b \boldsymbol{w}=\lambda \mathbf{S}_w \boldsymbol{w},
$$
已知 $\mathbf{S}_b=\left(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1\right)\left(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1\right)^{\mathrm{T}}$，这里$(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1)^T\boldsymbol{w}$是标量，且$\boldsymbol{w}$的大小无所谓，那放缩$\boldsymbol{w}$使得
$$
\mathbf{S}_b \boldsymbol{w}=\lambda\left(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1\right),
$$

代入$\mathbf{S}_b \boldsymbol{w}=\lambda \mathbf{S}_w \boldsymbol{w}$即得
$$
\boldsymbol{w}=\mathbf{S}_w^{-1}\left(\boldsymbol{\mu}_0-\boldsymbol{\mu}_1\right) .
$$

### 3.3 多分类学习

多分类问题的本质是基于一些基本策略，利用二分类学习器来解决多分类问题。

**一对一 ($OvO$)**：给定数据集 $D=\left\{\left(\boldsymbol{x}_1, y_1\right),\left(\boldsymbol{x}_2, y_2\right), \ldots,\left(\boldsymbol{x}_m, y_m\right)\right\}$，$y_i \in\left\{C_1, C_2, \ldots, C_N\right\}$。$\mathrm{OvO}$ 将这 $N$ 个类别两两配对，从而产生 $N(N-1) / 2$ 个二分类任务，例如为区分类别 $C_i$ 和 $C_j$ 训练一个分类器, 该分类器把 $D$ 中的 $C_i$ 类样例作为正例，$C_j$ 类样例作为反例。在测试阶段，新样本将同时提交给所有分类器，于是我们将得到 $N(N-1) / 2$ 个分类结果, 最终结果可通过投票产生: 即把被预测得最多的类别作为最终分类结果。

**一对多 ($OvR$)**：$\mathrm{OvR}$则是每次将一个类的样例作为正例、所有其他类的样例作为反例来训练 $N$ 个分类器。在测试时若仅有一个分类器预测为正类，则对应的类别标记作为最终分类结果。若有多个分类器预测为正类，则通常考虑各分类器的预测置信度，选择置信度最大的类别标记作为分类结果。

**多对多($MvM$)**：$ECOC$编码

### 3.4 类别不平衡问题

类别不平衡问题指的不同类别的训练样例数目差别很大的情况。一般来说，数据集中正反例的比例为1：1。

**再放缩**：当训练集中正、反例的数目不同时，令 $m^{+}$表示正例数目，$m^{-}$表示反例数目，则观测几率是 $\frac{m^{+}}{m^{-}}$​。由于我们通常假设训练集是真实样本总体的无偏采样，因此观测几率就代表了真实几率。我们这里认为，这个正反例的比例就是真实数据集里面二者的比值。于是，只要分类器的预测几率高于观测几率就应判定为正例，即
$$
若\frac{y}{1-y}>\frac{m^+}{m^-}
$$
则预测为正例。但是分类器实际决策时大于号右侧是1，所以需要移项
$$
\frac{y^{\prime}}{1-y^{\prime}}=\frac{y}{1-y} \times \frac{m^{-}}{m^{+}}.
$$
**过采样**：增加一些正例使得两类数量接近，例如 $SMOTE$ 算法，通过对数据集里的正例进行额外插值来做到。

**欠采样**：欠采样原理是丢掉一些反例使得两类数据数目大致一样，但是随机丢弃可能会丢掉一些边界样本导致分类结果边界差距很大。 $EasyEnsemble$ 是每次选取几个和正例相同数目的反例训练学习器，最后多个学习器的结果进行投票。



## 第四章 决策树

**属性**：基于树的结构来进行预测；决策过程的最终结论对应了我们所希望的判定结果；决策过程中提出的每个判定问题都是对某个属性的 “测试”；每个测试的结果或是导出最终结论，或是导出进一步的判定问题。

![6bc3597a71b44e669a47cb163bade921](C:\Users\YSY\Desktop\论文阅读\6bc3597a71b44e669a47cb163bade921.png)

决策树的构建过程需要时查书。

### 4.1 划分选择

从上述的伪代码可以看到，第八行如何划分最优属性是关键。一般而言，随着划分过程不断进行，我们希望决策树的分支结点所包含的样本尽可能属于同一类别，即结点的 “纯度” (purity)越来越高。

假定当前样本集合 $D$ 中第 $k$ 类样本所占的比例为 $p_k (k=1,2, \ldots,|\mathcal{Y}|)$，则 $D$的信息熵定义为
$$
\operatorname{Ent}(D)=-\sum_{k=1}^{|\mathcal{Y}|} p_k \log _2 p_k .
$$
$Ent(D)$的值越小，则$D$的纯度越高。这里的$Ent(D)$是针对样本最终的标签来进行计算，比如西瓜的好坏、或者一些三个或多个的label。

计算信息增益需要再对每个离散属性 $a$ 进行计算
$$
\operatorname{Gain}(D, a)=\operatorname{Ent}(D)-\sum_{v=1}^V \frac{\left|D^v\right|}{|D|} \operatorname{Ent}\left(D^v\right)
$$
假定离散属性 $a$ 有 $V$ 个可能的取值 $\left\{a^1, a^2, \ldots, a^V\right\}$，若使用 $a$ 来对样本集 $D$ 进行划分，则会产生 $V$ 个分支结点, 其中第 $v$ 个分支结点包含了 $D$ 中所有在属性 $a$ 上取值为 $a^v$ 的样本，记为 $D^v$。我们可根据式 $Ent(D)$ 公式计算出 $D^v$ 的信息熵，再考虑到不同的分支结点所包含的样本数不同，给分支结点赋予权重 $\left|D^v\right| /|D|$ ，即样本数越多的分支结点的影响越大，于是可计算出用属性 $a$ 对样本集 $D$ 进行划分所获得的“信息增益” (information gain)。计算 $Ent(D^v)$ 的时候，把每个 $D^v$​ 作为一个整体，分别对各个label进行计算。

特别地，对于连续值，需要将其离散化处理。在 $D$ 上的属性 $a$，假定它有 $n$ 个不同的取值，将其从小到大排序，记为{$a^1,a^2,...,a^n$}。依次把每两个相邻属性的中点作为划分，将数据集划分为两部分，小于阈值的为 $D_t^-$，大于阈值的为 $D^+_t$，计算信息增益。
$$
\begin{aligned} \operatorname{Gain}(D, a) & =\max _{t \in T_a} \operatorname{Gain}(D, a, t) \\ & =\max _{t \in T_a} \operatorname{Ent}(D)-\sum_{\lambda \in\{-,+\}} \frac{\left|D_t^\lambda\right|}{|D|} \operatorname{Ent}\left(D_t^\lambda\right),\end{aligned}
$$

### 4.2 几种决策树

上述的方法构造出的决策树叫做 ID3 决策树，下面列举 C4.5 和 CART 决策树

#### 4.2.1 C4.5决策树

实际上，信息增益准则对可取值数目较多的属性有所偏好，C4.5 不直接使用信息增益，而是使用“增益率”来划分最优属性。
$$
\operatorname{Gain\_ ratio}(D, a)=\frac{\operatorname{Gain}(D, a)}{\operatorname{IV}(a)},
$$

其中
$$
\operatorname{IV}(a)=-\sum_{v=1}^V \frac{\left|D^v\right|}{|D|} \log _2 \frac{\left|D^v\right|}{|D|}(类似Ent公式)
$$
 C4.5 在实际使用时：先从候选划分属性中找出信息增益高于平均水平的属性，再从中选出增益率最高的。

#### 4.2.2 CART

CART 决策树使用基尼系数选择划分属性
$$
\begin{aligned} \operatorname{Gini}(D) & =\sum_{k=1}^{|\mathcal{Y}|} \sum_{k^{\prime} \neq k} p_k p_{k^{\prime}} \\ & =1-\sum_{k=1}^{|\mathcal{Y}|} p_k^2 .
\end{aligned}
$$

$$
\operatorname{Gini\_ index}(D, a)=\sum_{v=1}^V \frac{\left|D^v\right|}{|D|} \operatorname{Gini}\left(D^v\right).
$$

选择划分后使得基尼系数最小的属性

#### 4.2.3 多变量决策树

以实现斜划分的多变量决策树为例，在此类决策树中，非叶结点不再是仅对某个属性，而是对属性的线性组合进行测试。

### 4.3 剪枝处理

是解决过拟合的主要手段

**预剪枝**：是指在决策树生成过程中，在每个结点划分前先进行估计，若当前节点划分不能在测试集上得到性能提升，则停止划分并把这个结点划分成叶结点。
**后剪枝**：后剪枝则是先从训练集生成一棵完整的决策树，然后自底向上地对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升，则将该子树替换为叶结点。

### 4.4 缺失值处理

两个主要问题：**（1）**如何在属性值缺失的情况下进行划分属性选择？即怎样算信息增益；（2）给定划分属性，若样本在该属性上的值缺失，如何对样本进行划分？

**基本思路：样本赋权，权重划分**

给定训练集 $D$ 和属性 $a$，令 $\tilde{D}$ 表示 $D$ 中在属性 $a$ 上没有缺失值的样本子集。对问题(1)，显然我们仅可根据 $\tilde{D}$ 来判断属性 $a$ 的优劣。假定属性 $a$ 有 $V$ 个可取值 $\left\{a^1, a^2, \ldots, a^V\right\}$，令 $\tilde{D}^v$ 表示 $\tilde{D}$ 中在属性 $a$ 上取值为 $a^v$ 的样本子集，$\tilde{D}_k$ 表示 $\tilde{D}$ 中属于第 $k$ 类$(k=1,2, \ldots,|\mathcal{Y}|)$ 的样本子集（这里的类指的是label，“好瓜”与“坏瓜”）, 则显然有 $\tilde{D}=\bigcup_{k=1}^{|\mathcal{Y}|} \tilde{D}_k$，$\tilde{D}=\bigcup_{v=1}^V \tilde{D}^v$。假定我们为每个样本 $\boldsymbol{x}$ 赋予一个权重 $w_{\boldsymbol{x}}$​, 并定义
$$
\begin{aligned} \rho & =\frac{\sum_{\boldsymbol{x} \in \tilde{D}} w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in D} w_{\boldsymbol{x}}}, \\ \tilde{p}_k & =\frac{\sum_{\boldsymbol{x} \in \tilde{D}_k} w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in \tilde{D}} w_{\boldsymbol{x}}} \quad(1 \leqslant k \leqslant|\mathcal{Y}|), \\ \tilde{r}_v & =\frac{\sum_{\boldsymbol{x} \in \tilde{D}^v} w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in \tilde{D}} w_{\boldsymbol{x}}} \quad(1 \leqslant v \leqslant V) .\end{aligned}
$$
直观地看，对属性 $a$，$\rho$ 表示无缺失值样本所占的比例，$\tilde{p}_k$ 表示无缺失值样本中第 $k$ 类所占的比例，$\tilde{r}_v$ 则表示无缺失值样本中在属性 $a$ 上取值 $a^v$ 的样本所占的比例。显然，$\sum_{k=1}^{|\mathcal{Y}|} \tilde{p}_k=1$，$\sum_{v=1}^V \tilde{r}_v=1$
基于上述定义，我们可将信息增益的计算式推广为
$$
\begin{aligned}
\operatorname{Gain}(D, a) & =\rho \times \operatorname{Gain}(\tilde{D}, a) \\
& =\rho \times\left(\operatorname{Ent}(\tilde{D})-\sum_{v=1}^V \tilde{r}_v \operatorname{Ent}\left(\tilde{D}^v\right)\right),
\end{aligned}
$$

且，
$$
\operatorname{Ent}(\tilde{D})=-\sum_{k=1}^{|\mathcal{Y}|} \tilde{p}_k \log _2 \tilde{p}_k .
$$

对问题(2)，若样本 $\boldsymbol{x}$ 在划分属性 $a$ 上的取值已知，则将 $x$ 划入与其取值对应的子结点，且样本权值在子结点中保持为 $w_{\boldsymbol{x}}$。若样本 $\boldsymbol{x}$ 在划分属性 $a$ 上的取值未知，则将 $x$ 同时划入所有子结点，且样本权值在与属性值 $a^v$ 对应的子结点中调整为 $\tilde{r}_v \cdot w_{\boldsymbol{x}}$​；直观地看，这就是让同一个样本以不同的概率划入到不同的子结点中去（**注意**开始阶段所有样本权重为1）。



## 第五章 神经网络

### 5.1 一些概念

**神经元**：接受来自$n$个其他神经元传递过来的输入信号，这些输入信号通过带权重的连接进行传递，神经元接收到的总输入值将与神经元的阈值进行比较，之后通过激活函数处理以产生神经元的输出

**感知机**：只包含输入层和输出层，只有输出层神经元进行激活函数处理

**隐含层**：输入层和输出层中间的神经元，和输出层一样有激活函数

**多层前馈神经网络**：每层神经元与下一层神经元全互连，神经元之间不存在同层连接，也不存在跨层连接。只需一个包含足够多神经元的隐层，多层前馈网络就能以任意精度逼近任意复杂度的连续函数

**局部最小和全局最小**：是局部最小未必是全局最小。以下是几种方法跳出局部最小接近全局最小：(1)以多组不同的参数初始化多个神经网络，按标准方法训练后，取其中误差最小的解作为最终参数；(2)使用 “模拟退火” 技术；(3)使用随机梯度下降

### 5.2 BP算法

![c7c874e0cca64d43b8a519bd7fddbf93(1)](C:\Users\YSY\Desktop\论文阅读\c7c874e0cca64d43b8a519bd7fddbf93(1).png)

训练集 $\mathrm{D}=\left\{\left(x_1, y_1\right),\left(x_2, y_2\right), \ldots\left(x_m, y_m\right)\right\}$，且输入样本 $x$ 是 $\mathrm{d}$ 维，输出值 $y$ 为 $l$ 维
$q$：隐层神经元个数

$\theta_{j}$ ：输出层第 $\tilde{j}$ 个神经元的阈值

$\gamma_h$ : 隐层第 $h$ 个神经元的阈值
$\nu_{i h}$ : 输入层第 $i$ 个神经元与隐层第 $h$ 个神经元之间的连接权
$\omega_{h j}$ : 隐层第 $h$ 个神经元与输出层第 $j$ 个神经元之间的连接权
$\alpha_h=\sum_{i=1}^d \nu_{i h} x_i$ : 隐层第 $h$ 个单元收到的输入数据
$\beta_j=\sum_{h=1}^q \omega_{h j} b_h$ : 输出层第 $j$ 个神经元收到的输入数据
$b_h:$ 隐层第 $h$ 个神经元的输出
$\hat{y}_j^k$ : 神经网络的最终输出数据，且 $\hat{y}_j^k=f\left(\beta_j-\theta_j\right)$
$E_k=\frac{1}{2} \sum_{j=1}^l\left(y_j^k \hat{-} y_j^k\right)^2:$ 神经网络在单个训练样本的均方误差

以 $w$ 为例子，任意参数 $v$ 的更新公式为
$$
v←v+Δv
$$
那么
$$
\Delta w_{h j}=-\eta \frac{\partial E_k}{\partial w_{h j}}
$$
**链式法则**：注意到 $w_{h j}$ 先影响到第 $j$ 个输出层神经元的输入 $\beta_{\mathrm{j}}$ ，再影响到其输出值 $\hat{y}_{\mathrm{j}}^{\mathrm{k}}$ ，然后影响到 $\mathrm{E}_{\mathrm{k}}$ ，有
$$
\frac{\partial \mathrm{E}_{\mathrm{k}}}{\partial \mathrm{w}_{\mathrm{hj}}}=\frac{\partial \mathrm{E}_{\mathrm{k}}}{\partial \hat{\mathrm{y}}_{\mathrm{j}}^{\mathrm{k}}} \cdot \frac{\partial \hat{\mathrm{y}}_{\mathrm{j}}^{\mathrm{k}}}{\partial \beta_{\mathrm{j}}} \cdot \frac{\partial \beta_{\mathrm{j}}}{\partial \mathrm{w}_{\mathrm{hj}}}
$$
根据定义
$$
\frac{\partial \beta_j}{\partial w_{h j}}=b_h
$$
Sigmoid 函数有很好的性质：
$$
f'(x)=f(x)(1-f(x)),
$$
由 $\hat{y}_j^k$ 和 $E_k$ 的定义
$$
\begin{aligned} \mathrm{g}_{\mathrm{j}} & =-\frac{\partial \mathrm{E}_{\mathrm{k}}}{\partial \hat{\mathrm{y}}_{\mathrm{j}}^{\mathrm{k}}} \cdot \frac{\partial \hat{\mathrm{y}}_{\mathrm{j}}^{\mathrm{k}}}{\partial \beta_{\mathrm{j}}} \\ & =-\left(\hat{\mathrm{y}}_{\mathrm{j}}^{\mathrm{k}}-\mathrm{y}_{\mathrm{j}}^{\mathrm{k}}\right) \mathrm{f}^{\prime}\left(\beta_{\mathrm{j}}-\theta_{\mathrm{j}}\right) \\ & =\hat{\mathrm{y}}_{\mathrm{j}}^{\mathrm{k}}\left(1-\hat{\mathrm{y}}_{\mathrm{j}}^{\mathrm{k}}\right)\left(\mathrm{y}_{\mathrm{j}}^{\mathrm{k}}-\hat{\mathrm{y}}_{\mathrm{j}}^{\mathrm{k}}\right)\end{aligned}
$$
那么 $w$ 的更新公式
$$
\Delta w_{h j}=\eta g_j b_h
$$



## 第六章 支持向量机

### 6.1 定义及求解过程

假设这里有一个包含两类的样本数据集 $D$（该数据集是线性可分的），那么这种简单的二分类问题，我们最基本的的思想就是找一个超平面将数据集划分好；但是会有多个可行的超平面，我们应该选哪一个呢？这就是支持向量机所考虑的问题：使得学习出的模型既不偏向正样本，也不偏向负样本，泛化能力更好。要选最中间的，因为最中间的与各个实例的间隔最大。

在样本空间中，划分超平面可以通过如下线性方程来描述：
$$
\boldsymbol{w}^T+b=0
$$
其中 $\boldsymbol{w}=(w_1;w_2;...;w_d)$ 是法向量，决定超平面的方向，$b$ 是位移项，记超平面为 $(\boldsymbol{w},b)$，样本空间中任意点 $\boldsymbol{x}$ 到超平面 $(\boldsymbol{w},b)$ 的距离可写为
$$
r=\frac{|\boldsymbol{w}^T\boldsymbol{x}+b|}{\|\boldsymbol{w}\|}.
$$
假设超平面 $(\boldsymbol{w}, b)$ 能将训练样本正确分类，即对于 $\left(\boldsymbol{x}_i, y_i\right) \in D$，若 $y_i=$ +1，则有 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_i+b>0$；若 ，则有 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_i+b<0$. 令
$$
\begin{cases}\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_i+b \geqslant+1, & y_i=+1 \\ \boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_i+b \leqslant-1, & y_i=-1 .\end{cases}
$$
是最靠近超平面的几个样本使得上式的等号成立，它们被称为“支持向量”，两个异类支持向量到超平面的距离为
$$
\gamma=\frac{2}{\|\boldsymbol{w}\|},
$$
这个叫做“间隔”，想找到最大“间隔”，即找到满足约束的参数 $\boldsymbol{w}$ 和 $b$，使得 $\gamma$ 最大，即
$$
\begin{aligned} \max _{\boldsymbol{w}, b} & \frac{2}{\|\boldsymbol{w}\|} \\ \text { s.t. } & y_i\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_i+b\right) \geqslant 1, \quad i=1,2, \ldots, m .\end{aligned}
$$
即最小化$\|\boldsymbol{w}\|^2$
$$
\begin{aligned} \min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^2 \\ \text { s.t. } & y_i\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_i+b\right) \geqslant 1, \quad i=1,2, \ldots, m .\end{aligned}
$$
使用拉格朗日乘子法得到其“对偶问题”，函数可写为
$$
L(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2}\|\boldsymbol{w}\|^2+\sum_{i=1}^m \alpha_i\left(1-y_i\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_i+b\right)\right)
$$
对 $\boldsymbol{w}$ 和 $b$ 求偏导
$$
\frac{\partial{L}}{\partial{\boldsymbol{w}}}=\frac{1}{2}(\boldsymbol{w}+\boldsymbol{w})+\sum_{i=1}^{m}(-\alpha_iy_i\boldsymbol{x_i})=0
\\\frac{\partial{L}}{\partial{b}}=\sum_{i=1}^{m}\alpha_iy_i=0
$$
那么有
$$
\begin{aligned} \boldsymbol{w} & =\sum_{i=1}^m \alpha_i y_i \boldsymbol{x}_i, \\ 0 & =\sum_{i=1}^m \alpha_i y_i\end{aligned}
$$
因为“对偶问题”得到的最小值是原始问题最优解的下界，所以要求一个极大 $\boldsymbol{\alpha}$（使用 $SMO$ 算法）；这里将上面两个条件代入，得
$$
\max _{\boldsymbol{\alpha}} \sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \boldsymbol{x}_i^{\mathrm{T}} \boldsymbol{x}_j
\\
\begin{array}{ll}
\text { s.t. } & \sum_{i=1}^m \alpha_i y_i=0 \\
& \alpha_i \geqslant 0, \quad i=1,2, \ldots, m .
\end{array}
$$
解出 $\boldsymbol{\alpha}$ 后，再求出 $\boldsymbol{w}$ 和 $b$ 即可得到模型 
$$
\begin{aligned} f(\boldsymbol{x}) & =\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b \\ & =\sum_{i=1}^m \alpha_i y_i \boldsymbol{x}_i^{\mathrm{T}} \boldsymbol{x}+b .\end{aligned}
$$
**性质**：

拉格朗日乘子法有不等式约束，所以上述过程要满足 $KKT$ 条件，即
$$
\left\{\begin{array}{l}\alpha_i \geqslant 0 ; \\ y_i f\left(\boldsymbol{x}_i\right)-1 \geqslant 0 ; \\ \alpha_i\left(y_i f\left(\boldsymbol{x}_i\right)-1\right)=0 .\end{array}\right.
$$
对任意训练样本 $\left(\boldsymbol{x}_i, y_i\right)$：若 $\alpha_i=0$，则该样本将不会在式 $\begin{aligned} f(\boldsymbol{x}) & =\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b=\sum_{i=1}^m \alpha_i y_i \boldsymbol{x}_i^{\mathrm{T}} \boldsymbol{x}+b\end{aligned}$ 的求和中出现，也就不会对 $f(\boldsymbol{x})$ 有任何影响；若 $\alpha_i>0$，则必有 $y_i f\left(\boldsymbol{x}_i\right)=1$，所对应的样本点位于最大间隔边界上，也就是说它是一个支持向量。这显示出支持向量机的一个重要性质：训练完成后，大部分的训练样本都不需保留，最终模型仅与支持向量有关。

### 6.2 核函数

理想的样本空间可以让被线性分开，也就是说有一个超平面能把训练样本正确分类，但是原始的样本空间不一定存在一个正确划分两类样本的超平面。

解决这个问题，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分。幸运的是，如果原始空间是有限维，即属性数有限，那么一定存在一个高维特征空间使样本可分。

令 $\phi(\boldsymbol{x})$ 表示将 $\boldsymbol{x}$ 映射后的特征向量，于是，在特征空间中划分超平面所对应的模型可表示为
$$
f(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T}} \phi(\boldsymbol{x})+b,
$$

其中 $\boldsymbol{w}$ 和 $b$ 是模型参数，和原来方法一样
$$
\begin{aligned}
& \min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^2 \\
& \text { s.t. } y_i\left(\boldsymbol{w}^{\mathrm{T}} \phi\left(\boldsymbol{x}_i\right)+b\right) \geqslant 1, \quad i=1,2, \ldots, m .
\end{aligned}
$$

其对偶问题如下
$$
\max _{\boldsymbol{\alpha}} \sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \phi\left(\boldsymbol{x}_i\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_j\right)
\\\begin{array}{ll}
\text { s.t. } & \sum_{i=1}^m \alpha_i y_i=0, 
\\& \alpha_i \geqslant 0, \quad i=1,2, \ldots, m .
\end{array}
$$

这里涉及到计算 $\phi\left(\boldsymbol{x}_i\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_j\right)$，这是样本 $\boldsymbol{x}_i$ 与 $\boldsymbol{x}_j$ 映射到特征空间之后的内积。由于特征空间维数可能很高，甚至可能是无穷维，因此直接计算 $\phi\left(\boldsymbol{x}_i\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_j\right)$ 通常是困难的。为了避开这个障碍，可以设想这样一个函数:
$$
\kappa\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=\left\langle\phi\left(\boldsymbol{x}_i\right), \phi\left(\boldsymbol{x}_j\right)\right\rangle=\phi\left(\boldsymbol{x}_i\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_j\right),
$$

即 $\boldsymbol{x}_i$ 与 $\boldsymbol{x}_j$ 在特征空间的内积等于它们在原始样本空间中通过函数 $\kappa(\cdot, \cdot)$ 计算的结果。有了这样的函数，我们就不必直接去计算高维甚至无穷维特征空间中的内积，于是上式可重写为
$$
\begin{aligned}
\max _{\boldsymbol{\alpha}} & \sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \kappa\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right) \\
\text { s.t. } & \sum_{i=1}^m \alpha_i y_i=0 \\
& \alpha_i \geqslant 0, \quad i=1,2, \ldots, m .
\end{aligned}
$$

求解后即可得到
$$
\begin{aligned}
f(\boldsymbol{x}) & =\boldsymbol{w}^{\mathrm{T}} \phi(\boldsymbol{x})+b \\
& =\sum_{i=1}^m \alpha_i y_i \phi\left(\boldsymbol{x}_i\right)^{\mathrm{T}} \phi(\boldsymbol{x})+b \\
& =\sum_{i=1}^m \alpha_i y_i \kappa\left(\boldsymbol{x}, \boldsymbol{x}_i\right)+b
\end{aligned}
$$
这里的函数 $\kappa(\cdot, \cdot)$ 就是 “核函数”。

什么样的函数可以做核函数，借用原书定义

**核函数定理：**令 $\mathcal{X}$ 为输入空间，$\kappa(\cdot, \cdot)$ 是定义在 $\mathcal{X} \times \mathcal{X}$ 上的对称函数，则 $\kappa$ 是核函数当且仅当对于任意数据 $D=\left\{x_1, x_2, \ldots, x_m\right\}$，“核矩阵” (kernel matrix) $\boldsymbol{K}$ 总是半正定的:
$$
\mathbf{K}=\left[\begin{array}{ccccc}
\kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_1\right) & \cdots & \kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_j\right) & \cdots & \kappa\left(\boldsymbol{x}_1, \boldsymbol{x}_m\right) \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\kappa\left(\boldsymbol{x}_i, \boldsymbol{x}_1\right) & \cdots & \kappa\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right) & \cdots & \kappa\left(\boldsymbol{x}_i, \boldsymbol{x}_m\right) \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\kappa\left(\boldsymbol{x}_m, \boldsymbol{x}_1\right) & \cdots & \kappa\left(\boldsymbol{x}_m, \boldsymbol{x}_j\right) & \cdots & \kappa\left(\boldsymbol{x}_m, \boldsymbol{x}_m\right)
\end{array}\right] .
$$

该定理表明，只要一个对称函数所对应的核矩阵半正定，它就能作为核函数使用。

我们希望样本在特征空间内线性可分，因此特征空间的好坏对支持向量机的性能至关重要。需注意的是，在不知道特征映射的形式时，我们并不知道什么样的核函数是合适的，而核函数也仅是隐式地定义了这个特征空间。于是，“核函数选择” 成为了支持向量机的最大变数。

**几个性质**：

（1）若 $\kappa_1$ 和 $\kappa_2$ 为核函数，则对于任意正数 $\gamma_1 、 \gamma_2$，其线性组合 $\gamma_1 \kappa_1+\gamma_2 \kappa_2$ 也是核函数；

（2）若 $\kappa_1$ 和 $\kappa_2$ 为核函数，则核函数的直积 $\kappa_1 \otimes \kappa_2(\boldsymbol{x}, \boldsymbol{z})=\kappa_1(\boldsymbol{x}, \boldsymbol{z}) \kappa_2(\boldsymbol{x}, \boldsymbol{z})$ 也是核函数;

（3）若 $\kappa_1$ 为核函数,则对于任意函数 $g(x)$，$\kappa(\boldsymbol{x}, \boldsymbol{z})=g(\boldsymbol{x}) \kappa_1(\boldsymbol{x}, \boldsymbol{z}) g(\boldsymbol{z})$ 也是核函数



## 第七章 贝叶斯分类器

### 7.1 贝叶斯决策论

**先验概率**：反映了我们在实际观察之前对某种状态的预期

**后验概率**：给定观测向量 $\boldsymbol{x}$，某个特定类别的概率 $P(y \mid \boldsymbol{x})$ 

**贝叶斯定理**：
$$
P(y,\boldsymbol{x})=P(y\mid\boldsymbol{x})P(\boldsymbol{x})=P(\boldsymbol{x}\mid y)P(y)
\\P(y\mid \boldsymbol{x})=\frac{P(\boldsymbol{x} \mid y)P(y)}{P(\boldsymbol{x})}=\frac{P(\boldsymbol{x}\mid y)P(y)}{\sum_{i=1}^{m}P(\boldsymbol{x}\mid y_i)P(y_i)}
$$


假设有 $N$ 种可能的类别标记，即 $\mathcal{Y}=\left\{c_1, c_2, \ldots, c_N\right\}$，$\lambda_{i j}$ 是将一个真实标记为 $c_j$ 的样本误分类为 $c_i$ 所产生的损失。基于后验概率 $P\left(c_i \mid \boldsymbol{x}\right)$ 可获得将样本 $\boldsymbol{x}$ 分类为 $c_i$ 所产生的期望损失，即在样本 $\boldsymbol{x}$ 上的 “条件风险”
$$
R\left(c_i \mid \boldsymbol{x}\right)=\sum_{j=1}^N \lambda_{i j} P\left(c_j \mid \boldsymbol{x}\right).
$$
这里的 $c_i$ 是分类的label，$P(c_j \mid \boldsymbol{x})$ 为 $\boldsymbol{x}$ 属于 $c_j$ 的概率。

我们的任务是找到一个判定准则 $h: \mathcal{X} \mapsto \mathcal{Y}$ 以最小化总体风险
$$
R(h)=\mathbb{E}_{\boldsymbol{x}}[R(h(\boldsymbol{x}) \mid \boldsymbol{x})].
$$
对于每个样本 $\boldsymbol{x}$，若 $h$ 能最小化条件风险 $R(h(\boldsymbol{x}) \mid \boldsymbol{x})$，则总体风险 $R(h)$ 也被最小化。这就产生了贝叶斯判定准则：为最小化总体风险，只需在每个样本上选择那个能使条件风险 $R(c \mid \boldsymbol{x})$ 最小的类别标记，即
$$
h^*(\boldsymbol{x})=\underset{c \in \mathcal{Y}}{\arg \min } R(c \mid \boldsymbol{x}),
$$
 具体来说，若目标是最小化分类错误率，则误判损失 $\lambda_{i j}$ 可写为
$$
\lambda_{i j}= \begin{cases}0, & \text { if } i=j \\ 1, & \text { otherwise, }\end{cases}
$$

**（这里为什么要这样取$\lambda$的值？逻辑在哪？）**

此时条件风险为（这里的 $c$ 是分类正确的label），
$$
R(c \mid \boldsymbol{x})=1-P(c \mid \boldsymbol{x}),
$$

目标是找到一个 $h(x)$ 让 $R(c\mid \pmb{x})=1-P(c \mid \boldsymbol{x})$ 最小。于是，最小化分类错误率的贝叶斯最优分类器为

$$
h^*(\boldsymbol{x})=\underset{c \in \mathcal{Y}}{\arg \max } P(c \mid \boldsymbol{x}),
$$

即这个分类器对每个样本 $\boldsymbol{x}$，选择能使后验概率 $P(c \mid \boldsymbol{x})$ 最大的类别标记

### 7.2 极大似然估计

不难发现，$P(c \mid \boldsymbol{x})$ 很难得到，我们需要基于有限的训练样本尽可能准确地估计出后验概率 $P(c \mid \boldsymbol{x})$。这里有两种策略：给定$\boldsymbol{x}$，可以通过直接建模 $P(c \mid \boldsymbol{x})$ 来预测 $c$，这种叫做**“判别式模型”**；贝叶斯判别器的方法是使用 **“生成式模型”**，即先对联合概率分布 $P(\boldsymbol{x}, c)$ 建模，然后再由此获得 $P(c \mid \boldsymbol{x})$ 。

基于贝叶斯定理, $P(c \mid \boldsymbol{x})$ 可写为
$$
P(c \mid \boldsymbol{x})=\frac{P(\boldsymbol{x}, c)}{P(\boldsymbol{x})}=\frac{P(c) P(\boldsymbol{x} \mid c)}{P(\boldsymbol{x})}
$$
在实际问题中 $P(c)$ 和 $P(\boldsymbol{x})$ 可以得到，那么要想办法得到$P(\boldsymbol{x} \mid c)$​。

**这个 $P(\boldsymbol{x})$ 是什么？**

方法是先假定分布类型再基于训练样本估计参数，假设 $P(\boldsymbol{x} \mid c)$ 具有确定的形式并且被参数向量 $\boldsymbol{\theta}_c$ 唯一确定，则我们的任务就是利用训练集 $D$ 估计参数 $\boldsymbol{\theta}_c$。为明确起见，我们将 $P(\boldsymbol{x} \mid c)$ 记为 $P\left(\boldsymbol{x} \mid \boldsymbol{\theta}_c\right)$​。

令 $D_c$ 表示训练集 $D$ 中第 $c$ 类样本组成的集合，假设这些样本是独立同分布的，则参数 $\boldsymbol{\theta}_c$ 对于数据集 $D_c$ 的似然是
$$
P\left(D_c \mid \boldsymbol{\theta}_c\right)=\prod_{\boldsymbol{x} \in D_c} P\left(\boldsymbol{x} \mid \boldsymbol{\theta}_c\right) .
$$
通常使用对数似然（防止积太小而下溢）
$$
\begin{aligned} L L\left(\boldsymbol{\theta}_c\right) & =\log P\left(D_c \mid \boldsymbol{\theta}_c\right) \\ & =\sum_{\boldsymbol{x} \in D_c} \log P\left(\boldsymbol{x} \mid \boldsymbol{\theta}_c\right)\end{aligned}
$$
极大似然估计是试图在 $\boldsymbol{\theta}_c$ 所有可能的取值中，找到一个能使数据出现的 “可能性” 最大的值。参数 $\boldsymbol{\theta}_c$ 的极大似然估计 $\hat{\boldsymbol{\theta}}_c$ 为
$$
\hat{\boldsymbol{\theta}}_c=\underset{\boldsymbol{\theta}_c}{\arg \max } L L\left(\boldsymbol{\theta}_c\right)
$$

### 7.3 朴素贝叶斯分类器

朴素贝叶斯认为每个属性独立地对分类结果发生影响，
$$
P(c \mid \boldsymbol{x})=\frac{P(c) P(\boldsymbol{x} \mid c)}{P(\boldsymbol{x})}=\frac{P(c)}{P(\boldsymbol{x})} \prod_{i=1}^d P\left(x_i \mid c\right),
$$
其中 $d$ 为属性数目, $x_i$ 为 $\boldsymbol{x}$ 在第 $i$ 个属性上的取值。

由于对所有类别来说 $P(\boldsymbol{x})$ 相同，因此贝叶斯判定准则有
$$
h_{n b}(\boldsymbol{x})=\underset{c \in \mathcal{Y}}{\arg \max } P(c) \prod_{i=1}^d P\left(x_i \mid c\right)
$$
**（1）**令 $D_c$ 表示训练集 $D$ 中第 $c$ 类样本组成的集合，若有充足的独立同分布样本，则可容易地估计出类先验概率
$$
P(c)=\frac{\left|D_c\right|}{|D|}
$$

**（2）**对离散属性而言，令 $D_{c, x_i}$ 表示 $D_c$ 中在第 $i$ 个属性上取值为 $x_i$ 的样本组成的集合，则条件概率 $P\left(x_i \mid c\right)$ 可估计为
$$
P\left(x_i \mid c\right)=\frac{\left|D_{c, x_i}\right|}{\left|D_c\right|}
$$

为了避免其他属性携带的信息被训练集中未出现的属性值 “抹去”（连乘的概率会是零），常用“拉普拉斯修正”。具体来说，令 $N$ 表示训练集 $D$ 中可能的类别数，$N_i$ 表示第 $i$ 个属性可能的取值数，上述两式分别修正为
$$
\begin{aligned}
\hat{P}(c) & =\frac{\left|D_c\right|+1}{|D|+N}, \\
\hat{P}\left(x_i \mid c\right) & =\frac{\left|D_{c, x_i}\right|+1}{\left|D_c\right|+N_i}
\end{aligned}
$$
**（3）**对连续属性可考虑概率密度函数，假定 $p\left(x_i \mid c\right) \sim \mathcal{N}\left(\mu_{c, i}, \sigma_{c, i}^2\right)$，其中 $\mu_{c, i}$ 和 $\sigma_{c, i}^2$ 分别是第 $c$ 类样本在第 $i$ 个属性上取值的均值和方差，则有
$$
p\left(x_i \mid c\right)=\frac{1}{\sqrt{2 \pi} \sigma_{c, i}} \exp \left(-\frac{\left(x_i-\mu_{c, i}\right)^2}{2 \sigma_{c, i}^2}\right)
$$


## 第八章 集成学习

### 8.1 简介

**集成学习的一般结构**：先产生一组 “个体学习器”，在使用某种策略将它们结合起来。

**“同质”集成**：集成中只包含同种类型的个体学习器，比如 “决策树集成” 中全是决策树，“神经网络集成” 中全是神经网络。个体学习器也叫做 “基学习器”，相应的学习算法叫做 “基学习算法”。

**“异质”集成**：包含不同种类个体学习器，比如同时含有神经网络和决策树。个体学习器在这里称作 “组件学习器”，或者直接叫做个体学习器。

**弱学习器**：泛化性能略优于随即猜测的学习器；例如在二分类问题上精度略高于50%的分类器。

集成学习通过将多个学习器进行结合，常可以获得比单一学习器显著优越的泛化性能，这对 “弱学习器” 尤为明显，因此集成学习的很多理论研究都是针对弱学习器进行的，基学习器有时也被直接成为弱学习器。

**原则**：要获得好的集成，个体学习器应 “好而不同”，即个体学习器要有一定的 “准确性”，不能效果太差，并且要有 “多样性”，即学习器间具有差距。

> 个体学习器的 “准确性” 和 “多样性” 本身就存在冲突。一般的，准确性很高之后，要增加多样性就需牺牲准确性。事实上，如何产生并结合 “好而不同” 的个体学习器，恰是集成学习研究的核心。
>
> 根据个体学习器的生成方式，目前的集成学习方法大致可分为两大类，即个体学习器间存在强依赖关系、必须串行生成的序列化方法，以及个体学习器间不存在强依赖关系、可同时生成的并行化方法；前者的代表是 Boosting，后者的代表是 Bagging 和 “随机森林” (Random Forest)。

### 8.2 Boosting

#### 8.2.1 简介

**工作机制**：先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基学习器；如此重复进行，直至基学习器数目达到事先指定的值 $T$，最终将这 $T$ 个基学习器进行加权结合。

#### 8.2.2 AdaBoost简介

定义**“加性模型”**：即基学习器的线性组合
$$
H(\boldsymbol{x})=\sum_{t=1}^{T}\alpha_t h_t(\boldsymbol{x})
$$
其中 $\alpha_t$ 表示基学习器的权重，$AdaBoost$ 是 “加权表决”，其结果为
$$
F(\boldsymbol{x})=\operatorname{sign}\left(H(\boldsymbol{x})\right)=\operatorname{sign}\left(\sum_{t=1}^T \alpha_t h_t(\boldsymbol{x})\right)
$$
**两个工具公式**：
$$
\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[f(\boldsymbol{x})\right]=\sum_{\boldsymbol{x}\in D}D(\boldsymbol{x})f(\boldsymbol{x})\\
\sum_{i=1}^{|D|}D(\boldsymbol{x}_i)\mathbb{I}(f(\boldsymbol{x}_i)=1)=P(f(\boldsymbol{x}_i)=1|\boldsymbol{x}_i)
$$
**指数损失函数**：
$$
\ell_{\exp }(H \mid \mathcal{D})=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}\right]
$$
其中 $\boldsymbol{x} \sim \mathcal{D}$ 表示服从概率分布 $D(\boldsymbol{x})$

这个函数有一些很好的性质：$f$ 是真实函数，$f(\boldsymbol{x})\in \lbrace-1,+1\rbrace$，$H(\boldsymbol{x})$ 是一个实数，代表集成学习器的预测结果；对于样本 $\boldsymbol{x}$， 当 $H(\boldsymbol{x})$ 和 $f(\boldsymbol{x})$ 符号一致时，$f(\boldsymbol{x})H(\boldsymbol{x})>0$，因此 $e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}=e^{|H(\boldsymbol{x})|}<1$，并且 $|H(\boldsymbol{x})|$ 越大，损失函数的值就越小；当 $H(\boldsymbol{x})$ 和 $f(\boldsymbol{x})$ 符号不一致时，$f(\boldsymbol{x})H(\boldsymbol{x})<0$，因此 $e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}=e^{|H(\boldsymbol{x})|}>1$，并且 $|H(\boldsymbol{x})|$ 越大，损失函数的值就越大。

**以下证明指数损失函数的合理性：**

将工具公式（1）代入损失函数，得
$$
\ell_{\exp }(H \mid \mathcal{D})=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H(\boldsymbol{x})}\right]=\sum_{\boldsymbol{x} \in D}D(\boldsymbol{x})e^{-f(\boldsymbol{x})H(\boldsymbol{x})}
$$
对 $\ell$ 求导，分类讨论
$$
\frac{\partial \ell_{\exp }(H \mid \mathcal{D})}{\partial H(\boldsymbol{x})}=\sum_{\boldsymbol{x} \in D}D(\boldsymbol{x})\left(-e^{-H(\boldsymbol{x})}\mathbb{I}(f(\boldsymbol{x})=1)+e^{H(\boldsymbol{x})}\mathbb{I}(f(\boldsymbol{x})=-1)\right)
$$
由工具公式（2）可得
$$
=-e^{-H(\boldsymbol{x})} P(f(\boldsymbol{x})=1 \mid \boldsymbol{x})+e^{H(\boldsymbol{x})} P(f(\boldsymbol{x})=-1 \mid \boldsymbol{x})
$$
令上式为零，移项并取对数，可解得
$$
H(\boldsymbol{x})=\frac{1}{2} \ln \frac{P(f(x)=1 \mid \boldsymbol{x})}{P(f(x)=-1 \mid \boldsymbol{x})},
$$
因此，有
$$
\begin{aligned}
\operatorname{sign}(H(\boldsymbol{x})) & =\operatorname{sign}\left(\frac{1}{2} \ln \frac{P(f(x)=1 \mid \boldsymbol{x})}{P(f(x)=-1 \mid \boldsymbol{x})}\right) \\
& = \begin{cases}1, & P(f(x)=1 \mid \boldsymbol{x})>P(f(x)=-1 \mid \boldsymbol{x}) \\
-1, & P(f(x)=1 \mid \boldsymbol{x})<P(f(x)=-1 \mid \boldsymbol{x})\end{cases} \\
& =\underset{y \in\{-1,1\}}{\arg \max } P(f(x)=y \mid \boldsymbol{x}),
\end{aligned}
$$

> 这意味着 $\operatorname{sign}(H(\boldsymbol{x}))$ 达到了贝叶斯最优错误率。换言之，若指数损失函数最小化，则分类错误率也将最小化；这说明指数损失函数是分类任务原本 $0 / 1$ 损失函数的一致的替代损失函数。

#### 8.2.3 AdaBoost参数更新

**分类器权重公式**

> 在 AdaBoost 算法中，第一个基分类器 $h_1$ 是通过直接将基学习算法用于初始数据分布而得；此后迭代地生成 $h_t$ 和 $\alpha_t$，当基分类器 $h_t$ 基于分布 $\mathcal{D}_t$ 产生后，该基分类器的权重 $\alpha_t$ 应使得 $\alpha_t h_t$ 最小化指数损失函数

$$
\begin{aligned} \ell_{\exp }\left(\alpha_t h_t \mid \mathcal{D}_t\right) & =\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_t}\left[e^{-f(\boldsymbol{x}) \alpha_t h_t(\boldsymbol{x})}\right] &f(\boldsymbol{x})和h(\boldsymbol{x})取值都是\lbrace-1,+1\rbrace\\ & =\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_t}\left[e^{-\alpha_t} \mathbb{I}\left(f(\boldsymbol{x})=h_t(\boldsymbol{x})\right)+e^{\alpha_t} \mathbb{I}\left(f(\boldsymbol{x}) \neq h_t(\boldsymbol{x})\right)\right] &根据工具公式1和2\\ & =e^{-\alpha_t} P_{\boldsymbol{x} \sim \mathcal{D}_t}\left(f(\boldsymbol{x})=h_t(\boldsymbol{x})\right)+e^{\alpha_t} P_{\boldsymbol{x} \sim \mathcal{D}_t}\left(f(\boldsymbol{x}) \neq h_t(\boldsymbol{x})\right) \\ & =e^{-\alpha_t}\left(1-\epsilon_t\right)+e^{\alpha_t} \epsilon_t\end{aligned}
$$

其中，错误率 $\epsilon_t=P_{\boldsymbol{x} \sim \mathcal{D}_t}\left(f(\boldsymbol{x}) \neq h_t(\boldsymbol{x})\right)$

对指数损失函数求导：
$$
\frac{\partial \ell_{\exp }\left(\alpha_t h_t \mid \mathcal{D}_t\right)}{\partial \alpha_t}=-e^{-\alpha_t}\left(1-\epsilon_t\right)+e^{\alpha_t} \epsilon_t
$$
令其为零，得
$$
\alpha_t=\frac{1}{2} \ln \left(\frac{1-\epsilon_t}{\epsilon_t}\right)
$$
至此得到了AdaBoost的分类器权重公式

**样本分布更新公式**

AdaBoost 算法在获得 $H_{t-1}$ 之后样本分布将进行调整，使下一轮的基学习器 $h_t$ 能纠正 $H_{t-1}$ 的一些错误。理想的 $h_t$ 能纠正 $H_{t-1}$ 的全部错误，$\ell_{\exp }\left(H_{t-1}+\alpha_t h_t \mid \mathcal{D}\right)$，因为 $\alpha_t$ 在每一轮中是常数，去掉不影响结果，可简化为
$$
\begin{aligned}
\ell_{\exp }\left(H_{t-1}+h_t \mid \mathcal{D}\right) & =\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x})\left(H_{t-1}(\boldsymbol{x})+h_t(\boldsymbol{x})\right)}\right] \\
& =\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})} e^{-f(\boldsymbol{x}) h_t(\boldsymbol{x})}\right]
\end{aligned}
$$
注意到 $f^2(\boldsymbol{x})=h_t^2(\boldsymbol{x})=1$，对 $e^{-f(\boldsymbol{x})h_t(\boldsymbol{x})}$ 泰勒展开，可得
$$
\begin{aligned}
\ell_{\exp }\left(H_{t-1}+h_t \mid \mathcal{D}\right) & \simeq \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\left(1-f(\boldsymbol{x}) h_t(\boldsymbol{x})+\frac{f^2(\boldsymbol{x}) h_t^2(\boldsymbol{x})}{2}\right)\right] \\
& =\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\left(1-f(\boldsymbol{x}) h_t(\boldsymbol{x})+\frac{1}{2}\right)\right]
\end{aligned}
$$
因为理想的基学习器 $h_t(\boldsymbol{x})$ 要使得指数损失函数最小化，所以
$$
\begin{aligned}
h_t(\boldsymbol{x}) & =\underset{h}{\arg \min } \ell_{\exp }\left(H_{t-1}+h \mid \mathcal{D}\right) \\
& =\underset{h}{\arg \min } \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\left(1-f(\boldsymbol{x}) h(\boldsymbol{x})+\frac{1}{2}\right)\right]  &1+\frac{1}{2}是常数不影响结果 \\
& =\underset{h}{\arg \max } \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})} f(\boldsymbol{x}) h(\boldsymbol{x})\right] &去掉负号\arg \min变\arg \max\\
& =\underset{h}{\arg \max } \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[\frac{e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]} f(\boldsymbol{x}) h(\boldsymbol{x})\right],
\end{aligned}
$$
注意到 $\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]$ 是一个常数，是构造 $\mathcal{D}_t$​ 分布
$$
\mathcal{D}_t(\boldsymbol{x})=\frac{\mathcal{D}(\boldsymbol{x}) e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]}
$$
继续对 $h_t(\boldsymbol{x})$ 变形
$$
\begin{aligned}
h_t(x) & =\underset{h}{\operatorname{argmax}} \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[\frac{e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]} f(\boldsymbol{x}) h(\boldsymbol{x})\right] \\
& =\underset{h}{\operatorname{argmax}} \sum_{i=1}^{|\mathcal{D}|} D\left(\boldsymbol{x}_i\right)\left[\frac{e^{-f\left(\boldsymbol{x}_i\right) H_{t-1}\left(\boldsymbol{x}_i\right)} f\left(\boldsymbol{x}_i\right) h\left(\boldsymbol{x}_i\right)}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f\left(\boldsymbol{x}\right) H_{t-1}\left(\boldsymbol{x}\right)}\right]}\right] &由数学期望定义得到\\
& =\underset{h}{\operatorname{argmax}} \sum_{i=1}^{|\mathcal{D}|} D_t\left(\boldsymbol{x}_i\right) f\left(\boldsymbol{x}_i\right) h\left(\boldsymbol{x}_i\right) \\
& =\underset{h}{\operatorname{argmax}} \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_t}[f(\boldsymbol{x}) h(\boldsymbol{x})]
\end{aligned}
$$

由于 $f(\boldsymbol{x}), h(\boldsymbol{x}) \in\{-1,+1\}$ ，有:
$$
f(\boldsymbol{x}) h(\boldsymbol{x})=1-2 \mathbb{I}(f(\boldsymbol{x}) \neq h(\boldsymbol{x}))
$$

则理想的基学习器 $h_t(\boldsymbol{x})$ :
$$
\begin{array}{rlr}h_t(\boldsymbol{x}) & =\underset{h}{\operatorname{argmax}} \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_t}[f(\boldsymbol{x}) h(\boldsymbol{x})] \\ & =\underset{h}{\operatorname{argmax}} \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_t}[1-2 \mathbb{I}(f(\boldsymbol{x}) \neq h(\boldsymbol{x}))]  \\ & =\underset{h}{\operatorname{argmin}} \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_t}[\mathbb{I}(f(\boldsymbol{x}) \neq h(\boldsymbol{x}))] & \text { 去掉常数和负号 }\end{array}
$$

由此可见，理想的 $h_t(\boldsymbol{x})$ 将在分布 $\mathcal{D}_t$​ 下最小化分类误差。

这里 $\mathcal{D}_{t+1}$ 的通项公式
$$
\begin{aligned} \mathcal{D}_{t+1}(\boldsymbol{x}) & =\frac{\mathcal{D}(\boldsymbol{x}) e^{-f(\boldsymbol{x}) H_t(\boldsymbol{x})}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_t(\boldsymbol{x})}\right]} \\ & =\frac{\mathcal{D}(\boldsymbol{x}) e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})} e^{-f(\boldsymbol{x}) \alpha_t h_t(\boldsymbol{x})}}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_t(\boldsymbol{x})}\right]} &展开H_t(\boldsymbol{x})可得\\ & =\mathcal{D}_t(\boldsymbol{x}) \cdot e^{-f(\boldsymbol{x}) \alpha_t h_t(\boldsymbol{x})} \frac{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]}{\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[e^{-f(\boldsymbol{x}) H_t(\boldsymbol{x})}\right]} &后面的复杂分数是个常数\end{aligned}
$$
至此，我们得到了 AdaBoost 的样本分布更新公式。

#### 8.2.4 AdaBoost的算法描述

输入：    训练集 $D=\left\{\left(\boldsymbol{x}_1, y_1\right),\left(\boldsymbol{x}_2, y_2\right), \ldots,\left(\boldsymbol{x}_m, y_m\right)\right\}$；
		基学习算法 $\mathfrak{L}$；
		训练轮数 $T$​

过程：
$$
\begin{aligned}
& 1: D_1(x)=\frac{1}{m} \text {. } \\
& 2 \text { : for } t=1,2, \ldots, T \text { do } \\
& 3: \quad h_t=\mathfrak{L}\left(D, D_t\right) \text {; } \\
& 4: \quad \epsilon_t=P_{\boldsymbol{x} \sim D_t}\left(h_t(\boldsymbol{x}) \neq f(\boldsymbol{x})\right) \text {; } \\
& 5: \quad \text { if } \epsilon>0.5 \text { then break } \\
& 6: \quad \alpha_t=\frac{1}{2} \ln \left(\frac{1-\epsilon_t}{\epsilon_t}\right) ; \\
& 7: \quad D_{t+1}(x)=\frac{D_t(x)}{Z_t} \times\left\{\begin{array}{ll}
e^{-\alpha_t}, & \text { if } h_t(\boldsymbol{x})=f(\boldsymbol{x}) \\
e^{\alpha_t}, & \text { if } h_t(\boldsymbol{x}) \neq f(\boldsymbol{x})
\end{array}=\frac{D_t(x) e^{-\alpha_t f(\boldsymbol{x}) h_t(\boldsymbol{x})}}{Z_t}\right. \\
& 8 \text { : end for } \\
\end{aligned}
$$
输出：
$$
F(x)=\operatorname{sign}\left(\sum_{t=1}^T \alpha_t h_t(\boldsymbol{x})\right)
$$


> 其中， $Z_t$ 是规范化因子，它使得 $\mathcal{D}_{t+1}$ 是一个分布：
> $$
> Z_t=\sum_{i=1}^m w_{t i} e^{-\alpha_t f\left(\boldsymbol{x}_i\right) h_t\left(\boldsymbol{x}_i\right)}
> $$
>
> 我们知道，$\mathcal{D}_{t+1}$ 表示的是样本权值的分布；因此，要想使得 $\mathcal{D}_{t+1}$ 是一个分布，应当使得 $\mathcal{D}_{t+1}$ 中的各样本概率 $w_{t i}$ 之和为1。为了做到这一点，令 $\mathcal{D}_{t+1}$ 中的每一个规范化前的样本权值 $w_{t i} e^{-\alpha_t f\left(\boldsymbol{x}_i\right) h_t\left(\boldsymbol{x}_i\right)}$ 都除以样本权值的总和 $\sum_{i=1}^m w_{t i} e^{-\alpha_t f\left(\boldsymbol{x}_i\right) h_t\left(\boldsymbol{x}_i\right)}$（这个总和也就是规范化因子 $Z_t$），从而让我们得到的各样本权值之和为1。(直观来说，规范化后的样本权值，即为规范化前的样本权值占总样本权值的比例，这个比例的总和当然为1）。由于项 $\frac{\mathbb{E}_{\boldsymbol{x} \sim D}\left[e^{-f(\boldsymbol{x}) H_{t-1}(\boldsymbol{x})}\right]}{\mathbb{E}_{\boldsymbol{x} \sim D}\left[e^{-f(\boldsymbol{x}) H_t(\boldsymbol{x})}\right]}$ 是一个常数，所有样本权值同时乘以常数，不会改变规范化前的样本权值占总样本权值的比例；因此无论是否有它，规范化之后的结果都是一样的，所以我们不去计算这个常数的值。

### 8.3 Bagging和随机森林

#### 8.3.1 Bagging

Bagging 这个名字是由 Bootstrap AGGregatING 缩写而来。他采用 bootstrap获得 $T$ 个大小为 $m$ 的数据集，然后基于每个采样集训练出一个基学习器，再将这些基学习器结合。在对预测输出进行结合的时候，Bagging 通常使用对分类任务使用简单投票法，对回归任务使用简单平均法。若票数相同，最简单的做法是随机选择一个，也可以进一步考察学习器的投票的置信度来确定最后胜者。

> 输入：	训练集 $D=\left\{\left(\boldsymbol{x}_1, y_1\right),\left(\boldsymbol{x}_2, y_2\right), \ldots,\left(\boldsymbol{x}_m, y_m\right)\right\}$；
> 		   基学习算法 $\mathfrak{L}$；
> 		   训练轮数 $T$
>
> 过程:
> $$
> \begin{aligned}
> 1:& \text { for } t=1,2, \ldots, T \text { do } \\
> 2:& \quad h_t=\mathfrak{L}\left(D, \mathcal{D}_{b s}\right) \\
> 3:& \text { end for }
> \end{aligned}
> $$
>
> 输出：$H(\boldsymbol{x})=\underset{y \in \mathcal{Y}}{\arg \max } \sum_{t=1}^T \mathbb{I}\left(h_t(\boldsymbol{x})=y\right)$

由于 bootstrap 只采样了 $63.2\%$ 的样本，剩下的 $36.8\%$ 的样本可以对每个基学习器的泛化性能做 “包外估计”。

#### 8.3.2 随机森林

Random Forest 是 Bagging 的一个变体，具体来说，传统决策树在选择划分属性时是在当前结点的属性集合（假定有 $d$ 个属性）中选择一个最优属性；而在 $\mathrm{RF}$ 中，对基决策树的每个结点，先从该结点的属性集合中随机选择一个包含 $k$ 个属性的子集，然后再从这个子集中选择一个最优属性用于划分。若令 $k=d$，则基决策树的构建与传统决策树相同；若令 $k=1$，则是随机选择一个属性用于划分；一般情况下，推荐值 $k=\log _2 d$。$\mathrm{RF}$ 也使用 bootstrap 获得 $T$ 个数据集，训练 $T$ 个决策树，最后通过投票得到结果。

与 Bagging 中基学习器的 “多样性” 仅通过样本扰动（通过对初始训练集采样）而来不同，随机森林中基学习器的多样性不仅来自样本扰动，还来自属性扰动，这就使得最终集成的泛化性能可通过个体学习器之间差异度的增加而进一步提升。

### 8.4 结合策略

学习器的结合有以下三大优点：

**（1）**从统计的方面来看，由于学习任务的假设空间往往很大，可能有多个假设在训练集上达到同等性能，此时若使用单学习器可能因误选而导致泛化性能不佳，结合多个学习器则会减小这一风险；

**（2）**从计算的方面来看，学习算法往往会陷入局部极小，有的局部极小点所对应的泛化性能可能很糟糕，而通过多次运行之后进行结合，可降低陷入糟糕局部极小点的风险；

**（3）**从表示的方面来看，某些学习任务的真实假设可能不在当前学习算法所考虑的假设空间中，此时若使用单学习器则肯定无效，而通过结合多个学习器，由于相应的假设空间有所扩大,有可能学得更好的近似。

#### 8.4.1 平均法

对于数值型输出 $h_i(\boldsymbol{x}) \in \mathbb{R}$

**简单平均法**：
$$
H(\boldsymbol{x})=\frac{1}{T} \sum_{i=1}^T h_i(\boldsymbol{x})
$$
**加权平均法**：
$$
H(\boldsymbol{x})=\sum_{i=1}^T w_i h_i(\boldsymbol{x})
$$

其中 $w_i$ 是个体学习器 $h_i$ 的权重，通常要求 $w_i \geqslant 0$，$\sum_{i=1}^T w_i=1$

#### 8.4.2 投票法

> 对分类任务来说，学习器 $h_i$ 将从类别标记集合 $\left\{c_1, c_2, \ldots, c_N\right\}$ 中预测出一个标记，最常见的结合策略是使用投票法。为便于讨论，我们将 $h_i$ 在样本 $\boldsymbol{x}$ 上的预测输出表示为一个 $N$ 维向量 $\left(h_i^1(\boldsymbol{x}) ; h_i^2(\boldsymbol{x}) ; \ldots ; h_i^N(\boldsymbol{x})\right)$，其中 $h_i^j(\boldsymbol{x})$是 $h_i$ 在类别标记 $c_j$ 上的输出。

**绝对多数投票法**：
$$
H(\boldsymbol{x})= \begin{cases}c_j, & \text { if } \sum_{i=1}^T h_i^j(\boldsymbol{x})>0.5 \sum_{k=1}^N \sum_{i=1}^T h_i^k(\boldsymbol{x}) \\ \text { reject, } & \text { otherwise. }\end{cases}
$$

即若某标记得票过半数，则预测为该标记；否则拒绝预测

**相对多数投票法**：
$$
H(\boldsymbol{x})=c_j^{\arg \max } \sum_{i=1}^T h_i^j(\boldsymbol{x})
$$
即预测为得票最多的标记，若同时有多个标记获最高票，则从中随机选取一个。

**加权投票法**：
$$
H(\boldsymbol{x})=c_j^{\arg \max } \sum_{i=1}^T w_i h_i^j(\boldsymbol{x})
$$

与加权平均法类似，$w_i$ 是 $h_i$ 的权重，通常 $w_i \geqslant 0, \sum_{i=1}^T w_i=1$。

上面三种方法没有限制个体学习器输出值的类型。在现实任务中，不同类型个体学习器可能产生不同类型的 $h_i^j(\boldsymbol{x})$ 值，常见的有：
- 类标记：$h_i^j(\boldsymbol{x}) \in\{0,1\}$，若 $h_i$ 将样本 $\boldsymbol{x}$ 预测为类别 $c_j$ 则取值为 1，否则为 0。使用类标记的投票亦称 “硬投票”。
- 类概率：$h_i^j(\boldsymbol{x}) \in[0,1]$，相当于对后验概率 $P\left(c_j \mid \boldsymbol{x}\right)$ 的一个估计。使用类概率的投票亦称 “软投票”。

#### 8.4.3 学习法

**Stacking算法**：

输入：	训练集 $D=\left\{\left(\boldsymbol{x}_1, y_1\right),\left(\boldsymbol{x}_2, y_2\right), \ldots,\left(\boldsymbol{x}_m, y_m\right)\right\}$;
		    初级学习算法 $\mathfrak{L}_1, \mathfrak{L}_2, \ldots, \mathfrak{L}_T$;
		    次级学习算法 $\mathfrak{L}$.

过程:
$$
\begin{aligned}
& \text {for } t=1,2, \ldots, \text {T } do\\
& \quad h_t=\mathfrak{L}_t(D)\\
& \text {end for}\\
& \text {D}^{\prime}=\varnothing;\\
& \text {for } i=1,2, \ldots, \text {m } do\\
& \quad \text {for } t=1,2, \ldots, \text {T } do\\
& \quad \quad z_{i t}=h_t\left(\boldsymbol{x}_i\right)\\
& \quad \text {end for}\\
& \quad D^{\prime}=D^{\prime} \cup\left(\left(z_{i 1}, z_{i 2}, \ldots, z_{i T}\right), y_i\right);\\
& \text {end for}\\
&h^{\prime}=\mathfrak{L}\left(D^{\prime}\right);\\
\end{aligned}
$$
输出：$H(\boldsymbol{x})=h^{\prime}\left(h_1(\boldsymbol{x}), h_2(\boldsymbol{x}), \ldots, h_T(\boldsymbol{x})\right)$

> 在训练阶段，次级训练集是利用初级学习器产生的，若直接用初级学习器的训练集来产生次级训练集，则过拟合风险会比较大；因此，一般是通过使用交叉验证或留一法这样的方式，用训练初级学习器未使用的样本来产生次级学习器的训练样本。以 $k$ 折交叉验证为例，初始训练集 $D$ 被随机划分为 $k$ 个大小相似的集合 $D_1, D_2, \ldots, D_k$。令 $D_j$ 和 $\bar{D}_j=D \backslash D_j$ 分别表示第 $j$ 折的测试集和训练集。给定 $T$ 个初级学习算法，初级学习器 $h_t^{(j)}$ 通过在 $\bar{D}_j$ 上使用第 $t$ 个学习算法而得。对 $D_j$ 中每个样本 $\boldsymbol{x}_i$，令 $z_{i t}=h_t^{(j)}\left(\boldsymbol{x}_i\right)$，则由 $\boldsymbol{x}_i$ 所产生的次级训练样例的示例部分为 $\boldsymbol{z}_i=\left(z_{i 1} ; z_{i 2} ; \ldots ; z_{i T}\right)$，标记部分为 $y_i$。于是，在整个交叉验证过程结束后，从这 $T$ 个初级学习器产生的次级训练集是 $D^{\prime}=\left\{\left(\boldsymbol{z}_i, y_i\right)\right\}_{i=1}^m$，然后 $D^{\prime}$ 将用于训练次级学习器。

### 8.5 多样性

个体学习器的准确性越高、多样性越大，集成度越好。

#### 8.5.1 多样性度量

多样性度量是用于度量集成中个体分类器的多样性，即估算个体学习器的多样化程度。

给定数据集 $D=\left\{\left(\boldsymbol{x}_1, y_1\right),\left(\boldsymbol{x}_2, y_2\right), \ldots,\left(\boldsymbol{x}_m, y_m\right)\right\}$，对二分类任务，$y_i \in$ $\{-1,+1\}$，分类器 $h_i$ 与 $h_j$ 的预测结果列联表为

|          | $h_i= +1$ | $h_i= -1$ |
| :------: | :-------: | :-------: |
| $h_j=+1$ |    $a$    |    $c$    |
| $h_j=-1$ |    $b$    |    $d$    |

其中，$a$ 表示 $h_i$ 与 $h_j$ 均预测为正类的样本数目；$b 、 c 、 d$ 含义由此类推；$a+b+c+d=m$。基于这个列联表，下面给出一些常见的多样性度量。

**不合度量**：
$$
d i s_{i j}=\frac{b+c}{m}
$$
$d i s_{i j}$ 的值域为 $[0,1]$，值越大则多样性越大。

**相关系数**：
$$
\rho_{i j}=\frac{a d-b c}{\sqrt{(a+b)(a+c)(c+d)(b+d)}}
$$
$\rho_{i j}$ 的值域为 $[-1,1]$，若 $h_i$ 与 $h_j$ 无关, 则值为 0；若 $h_i$ 与 $h_j$ 正相关则值为正，否则为负。

**$Q$-统计量**：
$$
Q_{i j}=\frac{a d-b c}{a d+b c} .
$$
$Q_{i j}$ 与相关系数 $\rho_{i j}$ 的符号相同，且 $\left|Q_{i j}\right| \leqslant\left|\rho_{i j}\right|$。

**$\kappa$-统计量**：
$$
\kappa=\frac{p_1-p_2}{1-p_2} .
$$

其中，$p_1$ 是两个分类器取得一致的概率；$p_2$ 是两个分类器偶然达成一致的概率，它们可由数据集 $D$ 估算：
$$
\begin{aligned}
& p_1=\frac{a+d}{m}, \\
& p_2=\frac{(a+b)(a+c)+(c+d)(b+d)}{m^2}
\end{aligned}
$$

若分类器 $h_i$ 与 $h_j$ 在 $D$ 上完全一致，则 $\kappa=1$；若它们仅是偶然达成一致，则 $\kappa=0$，$\kappa$ 通常为非负值，仅在 $h_i$ 与 $h_j$ 达成一致的概率甚至低于偶然性的情况下取负值。

#### 8.5.2 多样性增强

**样本数据扰动**：

> 给定初始数据集，可从中产生出不同的数据子集，再利用不同的数据子集训练出不同的个体学习器。数据样本扰动通常是基于采样法，例如在 Bagging 中使用自助采样，在 AdaBoost 中使用序列采样。此类做法简单高效，使用最广。对很多常见的基学习器，例如决策树、神经网络等，训练样本稍加变化就会导致学习器有显著变动，数据样本扰动法对这样的 “不稳定基学习器” 很有效；然而，有一些基学习器对数据样本的扰动不敏感，例如线性学习器、支持向量机、朴素贝叶斯、 $k$ 近邻学习器等，这样的基学习器称为稳定基学习器，对此类基学习器进行集成往往需使用输入属性扰动等其他机制。

**输入数据扰动**：

随机子空间算法：

输入：     训练集 $D=\left\{\left(\boldsymbol{x}_1, y_1\right),\left(\boldsymbol{x}_2, y_2\right), \cdots,\left(\boldsymbol{x}_m, y_m\right)\right\}$；
		基学习算法 $\mathfrak{L}$；
		基学习器数 $T$；
		子空间属性数 $d^{\prime}$

过程:
$$
\begin{array}{ll}
\text { 1: } & \text {for } t=1,2, \ldots, T \text { do } \\
\text { 2: } & \quad \mathcal{F}_t=\operatorname{RS}\left(D, d^{\prime}\right) \\
\text { 3: } & \quad D_t=\operatorname{Map}_{\mathcal{F}_t}(D) \\
\text { 4: } & \quad h_t=\mathfrak{L}\left(D_t\right)\\
\text { 5: } & \text{end  for}
\end{array}
$$
输出：$H(\boldsymbol{x})=\underset{y \in \mathcal{Y}}{\arg \max } \sum_{t=1}^T \mathbb{I}\left(h_t\left(\operatorname{Map}_{\mathcal{F}_t}(\boldsymbol{x})\right)=y\right)$

**输出表示扰动**：

> 此类做法的基本思路是对输出表示进行操纵以增强多样性。可对训练样本的类标记稍作变动，如 “翻转法” 随机改变一些训练样本的标记； 也可对输出表示进行转化，如 “输出调制法” 将分类输出转化为回归输出后构建个体学习器；还可将原任务拆解为多个可同时求解的子任务，如 ECOC 利用纠错输出码将多分类任务拆解为一系列二分类任务来训练基学习器。

**算法参数扰动**：

> 通过设置不同的基学习器的参数，往往可以产生差别较大的个体学习器。例如 “负相关法” 显式地通过正则化项来强制个体神经网络使用不同的参数。对参数较少的算法，可通过将其学习过程中某些环节用其他类似方式代替，从而达到扰动的目的，例如可将决策树使用的属性选择机制替换成其他的属性选择机制。值得指出的是，使用单一学习器时通常需使用交叉验证等方法来确定参数值，这事实上已使用了不同参数训练出多个学习器，只不过最终仅选择其中一个学习器进行使用，而集成学习则相当于把这些学习器都利用起来；由此也可看出，集成学习技术的实际计算开销并不比使用单一学习器大很多。

## 第九章 聚类

### 9.1 概述

在 “无监督学习” 中，训练样本的标记信息是未知的，目标是通过对无标记训练样本的学习来揭示数据的内在性质及规律，为进一步的数据分析提供基础。此类学习任务中研究最多、应用最广的是 “聚类”。

聚类试图将数据集中的样本划分为若干个通常是不相交的子集，每个子集称为一个 “簇”。 通过这样的划分，每个簇可能对应于一些潜在的概念（类别），如 “浅色瓜”、“深色瓜”、“有籽瓜”、“无籽瓜”，甚至 “本地瓜”、“外地瓜” 等；需说明的是，这些概念对聚类算法而言事先是未知的，聚类过程仅能自动形成簇结构，簇所对应的概念语义需由使用者来把握和命名。

形式化地说，假定样本集 $D=\left\{\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_m\right\}$ 包含 $m$ 个无标记样本,每个样本 $\boldsymbol{x_i}=\left(x_{i 1} ; x_{i 2} ; \ldots ; x_{i n}\right)$ 是一个 $n$ 维特征向量，则聚类算法将样本集 $D$ 划分为 $k$ 个不相交的簇 $\left\{C_l \mid l=1,2, \ldots, k\right\}$，其中 $C_{l^{\prime}} \bigcap_{l^{\prime} \neq l} C_l=\varnothing$且 $D=\bigcup_{l=1}^k C_l$。相应地，我们用 $\lambda_j \in\{1,2, \ldots, k\}$ 表示样本 $\boldsymbol{x}_j$ 的 “簇标记”，即 $\boldsymbol{x}_j \in C_{\lambda_j}$。于是，聚类的结果可用包含 $m$ 个元素的簇标记向量 $\boldsymbol{\lambda}=\left(\lambda_1 ; \lambda_2 ; \ldots ; \lambda_m\right)$​ 表示。

#### 9.1.1 性能度量

**为什么要引入性能度量？**

**（1）**对聚类结果，我们需通过某种性能度量来评估其好坏**（2）**另一方面，若明确了最终将要使用的性能度量，则可直接将其作为聚类过程的优化目标，从而更好地得到符合要求的聚类结果。

聚类性能度量大致有两类：一类是将聚类结果与某个 “参考模型” 进行比较，称为 “外部指标” ；另一类是直接考察聚类结果而不利用任何参考模型，称为 “内部指标”。

**外部指标**：

对数据集 $D=\left\{\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_m\right\}$，假定通过聚类给出的簇划分为 $\mathcal{C}=\left\{C_1\right.$, $\left.C_2, \ldots, C_k\right\}$，参考模型给出的簇划分为 $\mathcal{C}^*=\left\{C_1^*, C_2^*, \ldots, C_s^*\right\}$。相应地，令 $\boldsymbol{\lambda}$ 与 $\lambda^*$ 分别表示与 $\mathcal{C}$ 和 $\mathcal{C}^*$ 对应的簇标记向量。我们将样本两两配对考虑，定义
$$
\begin{aligned}
& \left.a=|S S|, \quad S S=\left\{\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right) \mid \lambda_i=\lambda_j, \lambda_i^*=\lambda_j^*, i<j\right)\right\}, \\
& \left.b=|S D|, \quad S D=\left\{\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right) \mid \lambda_i=\lambda_j, \lambda_i^* \neq \lambda_j^*, i<j\right)\right\}, \\
& \left.c=|D S|, \quad D S=\left\{\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right) \mid \lambda_i \neq \lambda_j, \lambda_i^*=\lambda_j^*, i<j\right)\right\}, \\
& \left.d=|D D|, \quad D D=\left\{\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right) \mid \lambda_i \neq \lambda_j, \lambda_i^* \neq \lambda_j^*, i<j\right)\right\},
\end{aligned}
$$

其中集合 $S S$ 包含了在 $\mathcal{C}$ 中隶属于相同簇且在 $\mathcal{C}^*$ 中也隶属于相同簇的样本对，集合 $S D$ 包含了在 $\mathcal{C}$ 中隶属于相同簇但在 $\mathcal{C}^*$ 中隶属于不同簇的样本对，$\cdots \cdots$ 由于每个样本对 $\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)(i<j)$ 仅能出现在一个集合中，因此有 $a+b+c+d=m(m-1) / 2$ 成立。

- **Jaccard 系数**
$$
\mathrm{JC}=\frac{a}{a+b+c}
$$
- **FM 指数**
$$
\mathrm{FMI}=\sqrt{\frac{a}{a+b} \cdot \frac{a}{a+c}}
$$
- **Rand 指数**

$$
\mathrm{RI}=\frac{2(a+d)}{m(m-1)}
$$

上述性能度量结果值均在 [0，1] 区间，值越大越好。

**内部指标**：

考虑聚类结果的簇划分 $\mathcal{C}=\left\{C_1, C_2, \ldots, C_k\right\}$, 定义
$$
\begin{aligned}
\operatorname{avg}(C) & =\frac{2}{|C|(|C|-1)} \sum_{1 \leqslant i<j \leqslant|C|} \operatorname{dist}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)\\
\operatorname{diam}(C) & =\max _{1 \leqslant i<j \leqslant|C|} \operatorname{dist}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)\\
d_{\text {min }}\left(C_i, C_j\right) & =\min _{\boldsymbol{x}_i \in C_i, \boldsymbol{x}_j \in C_j} \operatorname{dist}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)\\
d_{\text {cen }}\left(C_i, C_j\right) & =\operatorname{dist}\left(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j\right)
\end{aligned}
$$

其中，$\operatorname{dist}(\cdot, \cdot)$ 用于计算两个样本之间的距离；$\boldsymbol{\mu}$ 代表簇 $C$ 的中心点 $\boldsymbol{\mu}=$ $\frac{1}{|C|} \sum_{1 \leqslant i \leqslant|C|} x_i$。显然, $\operatorname{avg}(C)$ 对应于簇 $C$ 内样本间的平均距离，$\operatorname{diam}(C)$ 对应于簇 $C$ 内样本间的最远距离，$d_{\min }\left(C_i, C_j\right)$ 对应于簇 $C_i$ 与簇 $C_j$ 最近样本间的距离，$d_{\mathrm{cen}}\left(C_i, C_j\right)$ 对应于簇 $C_i$ 与簇 $C_j$​ 中心点间的距离。

- **DB 指数**（Davies - Bouldin Index，简称 $DBI$）
$$
\mathrm{DBI}=\frac{1}{k} \sum_{i=1}^k \max _{j \neq i}\left(\frac{\operatorname{avg}\left(C_i\right)+\operatorname{avg}\left(C_j\right)}{d_{\operatorname{cen}}\left(C_i, {C}_j\right)}\right)
$$
- **Dunn 指数**（Dunn Index，简称 $DI$）
$$
\mathrm{DI}=\min _{1 \leqslant i \leqslant k}\left\{\min _{j \neq i}\left(\frac{d_{\min }\left(C_i, C_j\right)}{\max _{1 \leqslant l \leqslant k} \operatorname{diam}\left(C_l\right)}\right)\right\}
$$

显然，$DBI$ 的值越小越好，而 $DI$ 则相反，值越大越好。

#### 9.1.2 距离计算

对函数 $\operatorname{dist}(\cdot, \cdot)$，若它是一个 “距离度量”，则需满足一些基本性质：
非负性：$\operatorname{dist}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right) \geqslant 0$；
同一性：$\operatorname{dist}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=0$ 当且仅当 $\boldsymbol{x}_i=\boldsymbol{x}_j$；
对称性：$\operatorname{dist}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=\operatorname{dist}\left(\boldsymbol{x}_j, \boldsymbol{x}_i\right)$；
直递性：$\operatorname{dist}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right) \leqslant \operatorname{dist}\left(\boldsymbol{x}_i, \boldsymbol{x}_k\right)+\operatorname{dist}\left(\boldsymbol{x}_k, \boldsymbol{x}_j\right)$

**有序属性**：例如定义域为 $\{1,2,3\}$ 的离散属性与连续属性的性质更接近一些，能直接在属性值上计算距离：“ 1 ” 与 “ 2 ” 比较接近、与 “ 3 ” 比较远，这样的属性称为 “有序属性”；而定义域为 $\{$ 飞机, 火车, 轮船 $\}$ 这样的离散属性则不能直接在属性值上计算距离，称为 “无序属性” 。

适用于有序属性的**闵可夫斯基距离**：
$$
\operatorname{dist}_{\mathrm{mk}}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=\left(\sum_{u=1}^n\left|x_{i u}-x_{j u}\right|^p\right)^{\frac{1}{p}} .
$$

$p=2$ 时，闵可夫斯基距离即欧氏距离(Euclidean distance)
$$
\operatorname{dist}_{\text {ed }}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=\left\|\boldsymbol{x}_i-\boldsymbol{x}_j\right\|_2=\sqrt{\sum_{u=1}^n\left|x_{i u}-x_{j u}\right|^2} .
$$
$p=1$ 时, 闵可夫斯基距离即曼哈顿距离(Manhattan distance)
$$
\operatorname{dist}_{\text {man }}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=\left\|\boldsymbol{x}_i-\boldsymbol{x}_j\right\|_1=\sum_{u=1}^n\left|x_{i u}-x_{j u}\right| .
$$
对于无序属性可使用 $VDM$：
$$
\operatorname{VDM}_p(a, b)=\sum_{i=1}^k\left|\frac{m_{u, a, i}}{m_{u, a}}-\frac{m_{u, b, i}}{m_{u, b}}\right|^p
$$
对于混合属性，假定有 $n_c$ 个有序属性、 $n-n_c$ 个无序属性，不失一般性，令有序属性排列在无序属性之前，则
$$
\operatorname{MinkovDM}_p\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=\left(\sum_{u=1}^{n_c}\left|x_{i u}-x_{j u}\right|^p+\sum_{u=n_c+1}^n \operatorname{VDM}_p\left(x_{i u}, x_{j u}\right)\right)^{\frac{1}{p}} .
$$

当样本空间中不同属性的重要性不同时，可使用 “加权距离”。以加权闵可夫斯基距离为例：
$$
\operatorname{dist}_{\mathrm{wmk}}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=\left(w_1 \cdot\left|x_{i 1}-x_{j 1}\right|^p+\ldots+w_n \cdot\left|x_{i n}-x_{j n}\right|^p\right)^{\frac{1}{p}},
$$

其中权重 $w_i \geqslant 0(i=1,2, \ldots, n)$ 表征不同属性的重要性，通常 $\sum_{i=1}^n w_i=1$。

**注意**：通常我们是基于某种形式的距离来定义 “相似度度量”，距离越大，相似度越小。然而，用于相似度度量的距离未必一定要满足距离度量的所有基本性质，尤其是直递性。

### 9.2 原形聚类

