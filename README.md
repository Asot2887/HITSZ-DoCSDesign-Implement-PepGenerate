# HITSZ-DoCSDesign-Implement-PepGenerate
HITSZ 2022秋 面向领域的计算机设计与实践 复现并尝试优化用于生成肽链的FBGAN模型

### 介绍
  本仓库用于保存课程代码。主要包括训练在mnist上的WGAN、对论文源码稍作修改后的FBGAN、对FBGAN生成数据的分析等

### FBGAN 
+ 论文：https://doi.org/10.1038/s42256-019-0017-4
+ 源码：https://github.com/av1659/fbgan

### 课设内容
#### 训练在mnist上的WGAN
+ 用于熟悉GAN与WGAN

#### 稍作修改的FBGAN
+ 稍微修改，使得FBGAN能够在torch 1.10版本跑通。
+ 直接指定FBGAN中核苷酸映射表Charmap，避免因为读入顺序使得Charmap错误
+ 引入FBGAN中WGAN训练集留存率，每轮迭代保留一定比例的训练集，减小一次超过阈值的肽链过多对最终生成方向造成的影响

#### FBGAN生成肽链分析
+ 主要对FBGAN生成序列的数字特征和理化特征进行分析

