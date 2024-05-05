普通的机器学习通过手工设定的特征提取器识别和分类图片特征
更有意思的模式是特征提取器通过输入的图像自动完成特征提取和分类
但是这种模式有问题：1、参数太多
2、以线性形式输入，空间信息被销毁
针对这些问题，卷积神经网络有两个特性解决了问题
1、卷积神经网络具有平移不变性：可以跨空间复制权值
2、卷积神经网络由有限大小的卷积核遍历图像，能够有效提取局部空间特征（局部感受野）
三个架构思想
1、局部感受野
没有使用全连接，转而使用局部连接。神经网络可以提取初级的视觉特征
2、权重共享
卷积核遍历整张图像，可以强制整张图片的局部感受野拥有相同权重向量，这就是权值共享。输出的结果成为特征图
3、下采样层（池化层！！）
比如在识别手写字体时，关注的是特征和特征相对其他特征的位置。（如上方区域包括一个端点，下方包括另一个端点，那么就判断是数字1）然而，如果提取得过分精确（如端点垂直正下方是另一个端点才是1），就会造成过拟合，在测试集中效果不佳
（！池化层中的连续单元通常具有不重叠的感受野）
因此要添加池化层，降低空间分辨率，从而降低对平移、形变的敏感度（如识别斜的1）
总结：原始图像中的局部通过卷积核和偏置计算后得到一个特征值，卷积核遍历原始图像后，得到一张特征图，特征图中的每一位代表一个神经元，它展示了在它对应的原始图像上卷积核大小的局部区域（感受野）的某种特征（可能是边缘、转角等），原始图像平移时，特征值同时平移，这就是平移不变性；生成一张特征图只需要一组权重，这就是权值共享。池化层降低神经网络对形变、平移等的敏感度，避免过拟合

激活函数
经过下采样的特征图经过sigmoid或者relu激活，将输出化为是非状态