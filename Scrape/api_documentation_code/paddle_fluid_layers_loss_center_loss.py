import paddle.fluid as fluid
import paddle
paddle.enable_static()

input = fluid.data(name='x',shape=[20,30],dtype='float32')
label = fluid.data(name='y',shape=[20,1],dtype='int64')
num_classes = 1000
alpha = 0.01
param_attr = fluid.initializer.Xavier(uniform=False)
center_loss=fluid.layers.center_loss(input=input,
       label=label,
       num_classes=1000,
       alpha=alpha,
       param_attr=fluid.initializer.Xavier(uniform=False),
       update_center=True)