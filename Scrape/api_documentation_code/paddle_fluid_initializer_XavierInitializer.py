import paddle
paddle.enable_static()
import paddle.fluid as fluid
queries = fluid.data(name='x', shape=[None,1], dtype='float32')
fc = fluid.layers.fc(
    input=queries, size=10,
    param_attr=fluid.initializer.Xavier(uniform=False))