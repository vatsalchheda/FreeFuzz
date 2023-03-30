import paddle
import paddle.nn as nn

data = paddle.rand([30, 10, 2], dtype='float32')
linear = nn.Linear(2,
                   4,
                   weight_attr=nn.initializer.Constant(value=2.0))
res = linear(data)
print(linear.weight.numpy())
#result is [[2. 2. 2. 2.],[2. 2. 2. 2.]]