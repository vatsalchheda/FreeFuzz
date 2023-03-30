import paddle
import numpy as np

# numpy array
data_1 = paddle.ones(shape=[1, 2], dtype='float32')
weight_attr_1 = paddle.framework.ParamAttr(
    name="linear_weight_1",
    initializer=paddle.nn.initializer.Assign(np.array([2, 2])))
bias_attr_1 = paddle.framework.ParamAttr(
    name="linear_bias_1",
    initializer=paddle.nn.initializer.Assign(np.array([2])))
linear_1 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
# linear_1.weight:  [2. 2.]
# linear_1.bias:  [2.]

res_1 = linear_1(data_1)
# res_1:  [6.]

# python list
data_2 = paddle.ones(shape=[1, 2], dtype='float32')
weight_attr_2 = paddle.framework.ParamAttr(
    name="linear_weight_2",
    initializer=paddle.nn.initializer.Assign([2, 2]))
bias_attr_2 = paddle.framework.ParamAttr(
    name="linear_bias_2",
    initializer=paddle.nn.initializer.Assign([2]))
linear_2 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_2, bias_attr=bias_attr_2)
# linear_2.weight:  [2. 2.]
# linear_2.bias:  [2.]

res_2 = linear_2(data_2)
# res_2:  [6.]

# tensor
data_3 = paddle.ones(shape=[1, 2], dtype='float32')
weight_attr_3 = paddle.framework.ParamAttr(
    name="linear_weight_3",
    initializer=paddle.nn.initializer.Assign(paddle.full([2], 2)))
bias_attr_3 = paddle.framework.ParamAttr(
    name="linear_bias_3",
    initializer=paddle.nn.initializer.Assign(paddle.full([1], 2)))
linear_3 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_3, bias_attr=bias_attr_3)
# linear_3.weight:  [2. 2.]
# linear_3.bias:  [2.]

res_3 = linear_3(data_3)
# res_3:  [6.]