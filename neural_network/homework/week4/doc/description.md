@[TOC](目录)

# task
构建一个多层的神经网络，每一层分别使用 ReLU 或者 Sigmod 作为激活函数。  
要求构建的模型具有以下功能：  
1. 能够以可以接收的错误率对未知图片进行识别
2. 在进行模型训练时，能够绘制 cost-iteration curve, testingerror-iteration curve, step-cost-iteration curve

# the structure of model
1. 导入训练数据和测试数据
2. 网络层数以及节点数设置
3. 网络参数的初始化
4. 前向传播
 4.1 计算一层中线性求和的部分
 4.2 计算激活函数的输出（ReLU或者Sigmod）
 4.3 结合线性求和与激活函数
5. 计算误差
6. 反向传播