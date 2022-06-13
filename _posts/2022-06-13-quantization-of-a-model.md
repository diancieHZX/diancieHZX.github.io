# 深度学习模型的部署——模型量化

*模型量化*

### 模型量化的概念

模型量化是一种能减少模型的大小（通过将float32数据类型转换为int8/int16数据类型等），并加速深度模型在推理时的速度及占用内存的一项优化技术。模型量化是在保持一个较低的精度损失上进行的，量化后的模型仍需要底层硬件的支持。一般量化可分为模型训练感知量化（Quantization Aware Training，QAT）和模型训练后量化（PTQ），而后量化又可以分为动态量化（Post Training Dynamic Quantization）和静态量化（Post Training Static Quantization）

### 模型量化的作用

1. 减少计算时内存的使用以及整个模型所需要的储存空间。

   深度学习模型主要记录每个层的参数如权重（weight）和偏差（bias），在Float32模型中每个参数为Float32的数据类型，即需占用32-bit的存储空间，量化后通常为int8的数据类型，即只需要8-bit的存储空间，因此模型大小将降低至大约为原来的1/4。在推理过程中，对于一些依赖于内存（memory bound）的算子来说这将明显降低其对内存的使用需求。

2. 提高系统吞吐量（throughput），降低系统延时（latency）

   对于一个专用寄存器宽度为512位的SIMD指令，但传入数据类型为Float32时一条指令理论能处理16个数，而采用int8，一条指令则能最大处理64个数，因此芯片的理论计算峰值将数倍增加。



### 量化算法

##### 非对称算法（asymmetric）

非对称算法那的基本思想是通过 收缩因子（scale） 和 零点（zero point） 将 FP32 张量 的 min/max 映射分别映射到 8-bit 数据的 min/max。

我们用 x_f 表示 原始浮点数张量, 用 x_q 表示量化张量, 用 q_x 表示 scale，用 zp_x 表示 zero_point, n 表示量化数值的 bit数，这里 n=8， 那么非对称算法的量化公式如下：
$$
x_q = round((x_f - min_{x_f})\frac{2^n - 1}{max_{x_f} - min_{x_f}}) \\ 
x_q = round(q_xx_f - zp_x)
$$


##### 对称算法（symmetric）

对称算法的基本思路是通过一个收缩因子（scale）将 FP32 tensor 中的最大绝对值映射到 8-bit数据的最大值，将最大绝对值的负值映射到 8-bit 数据的最小值。以 int8 为例，max(|x_f|)被映射到 127，-max(|x_f|)被映射到-128。

![人工智能干货｜一线工程师带你学习深度学习模型量化理论+实践](https://www.freesion.com/images/27/abf482bdd41bd42475f6f47cf61e65a3.JPEG)

量化公式为：
$$
x_q = round(x_f\frac{2^n - 1}{max_|x_f|}) \\ 
x_q = round(q_xx_f)
$$

##### 浮点数动态范围选取

为了计算 scale 和 zero_point 我们需要知道 FP32 weight/activiation 的实际动态范围。对于模型的推理过程来说， weights 是一个常量张量，不需要额外数据集进行采样即可确定实际的动态范围。 但 activation 的实际动态范围则必须经过采样获取，一般把这个过程称为数据校准(calibration) 。目前各个深度学习框架中，使用最多的有最大最小值(MinMax)， 滑动平均最大最小值(MovingAverageMinMax) 和 KL 距离(Kullback–Leibler divergence) 三种。如果量化过程中的每一个 FP32 数值都在这个实际动态范围内，一般称这种为不饱和状态；反之如果出现某些 FP32 数值不在这个实际动态范围之内称之为饱和状态。

- 最大最小值(MinMax)：这种算法的优点是简单直接，但是对于 activation 而言，如果采样数据中出现离群点，则可能明显扩大实际的动态范围。
- 滑动平均最大最小值(MovingAverageMinMax)：MovingAverageMinMax 会采用一个超参数 c (Pytorch 默认值为0.01)逐步更新动态范围。这种方法获得的动态范围一般要小于实际的动态范围。对于 weights 而言，由于不存在采样的迭代，因此 MovingAverageMinMax 与 MinMax 的效果是一样的。
- KL 距离采样方法(Kullback–Leibler divergence)：KL 距离一般被用来度量两个分布之间的相似性。 KL 距离采样方法从理论上似乎很合理，但是也有几个缺点：1）动态范围的选取相对耗时。2）上述算法只是假设左侧动态范围不变的情况下对右边的边界进行选取，对于 RELU 这种数据分布的情况可能很合理，但是如果对于右侧数据明显存在长尾分布的情况可能并不友好。除了具有像RELU等这种具有明显数据分布特征的情况，其他情况我们并不清楚从左边还是从右边来确定动态范围的边界。3）quantize/expand 方法也只是一定程度上模拟了量化的过程。

##### 量化粒度

量化粒度一般分为 张量级量化（tensor-wise）和 通道级量化 (channel-wise)。Tensor-wise 量化为一个张量指定一个 scale，是一种粗粒度的量化方式。Channel-wise 量化为每一个通道指定一个 scale 属于一种细粒度的量化方式。

- 张量级量化（tensor-wise/per_tensor/per_layer）：Activation 和 weights 都可以看做是一个张量，因此在这种量化方式，两者并没有区别。

- 通道级量化（channel-wise/per_channel）：在深度学习中，张量的每一个通道通常代表一类特征，因此可能会出现不同的通道之间数据分布较大的情况。对于通道之间差异较大的情况仍然使用张量级的量化方式可能对精度产生一定的影响，因此通道级量化就显得格外重要。对于 activation 而言，在卷积网络中其格式一般为 NCHW。其中 N 为 batch_size，C 为通道数，H 和W分别为高和宽。这时量化将会有C个 scale，即以通道为单位进行量化。对于 weights 而言，在卷积网络中其格式一般为 OIHW，其中 O 为输出通道数, I 为输入通道数，H 和 W分别为卷积核高和宽。这时量化将会有 O 个scale，即以输出通道为单位进行量化。

### Pytorch post-training 量化

##### 1）模型准备：

插桩：在需要 quantize 和 dequantize 操作的 module 中插入 QuantStub 和DeQuantStub。

去重： 保证每一个 module 对象不能被重复使用，重复使用的需要定义多个对象，比如 一个 nn.relu 对象不能在 forward 中使用两次，否则在 calibration 阶段无法观测正确的浮点数动态范围。。

转换：非 module 对象表示的算子不能转换成 quantized module。比如 "+" 算术运算符无法直接转成 quantize module。

##### 2）fuse modules:

为了提高精度和性能，一般将 conv + relu, conv + batchnorm + relu, linear + relu 等类似的操作 fuse 成一个操作。

##### 3）量化算法：

为 activations/weights 指定量化算法 比如 symmtric/asymmtric/minmax 等等。Pytorch 采用 qconfig 来封装量化算法，一般通过将 qconfig 作为 module 的属性来指定量化算法。常用的 qconfig 有default_per_channel_qconfig，default_qconfig等。

##### 4）插入observer和传入qconfig

torch.quantization.prepare() 向子 module 传播 qconfig,并为子 module 插入 observer。 Observer 可以在获取 FP32 activations/weights 的动态范围。

##### 6）module转化：

torch.quantization.convert 函数可以将 FP32 module 转化成 int8 module. 这个过程会量化 weights, 计算并存储 activation 的 scale 和 zero_point。