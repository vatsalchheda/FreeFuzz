import paddle
paddle.enable_static()
import paddle.fluid as fluid
x = fluid.data(name='x', shape=[3, 7, 3, 7], dtype='float32')
hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
hidden2 = fluid.layers.inplace_abn(input=hidden1)
hidden3 = fluid.layers.inplace_abn(input=hidden2, act='leaky_relu', act_alpha=0.2)