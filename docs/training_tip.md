### 变量

#### 定量

* n_epoch

#### 变量

1. min_error
2. loss_record

每个batch都记录

记录train和valid的loss

3. early_stop_cnt

#### 过程

每次epoch开始

每个batch的loss都要被记录

每个epoch结束

进行valid：

如果valid的error小于现在的min_error

就将模型存入文件，并设early_stop_cnt=0

如果valid的error大于现在的min_error

就将early_stop_cnt+1，当early_stop_cnt达到某个值提前停止训练

#### 从训练样本中随机抽10%作为valid



我准备线用直接的单层算算，但是loss直接无穷，

但当我调成两层，就可以训练了





### 数据预处理

1. 离散的量编成one-hot vector
2. 连续的量做normalization
