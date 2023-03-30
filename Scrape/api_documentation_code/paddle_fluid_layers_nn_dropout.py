import paddle
import paddle.fluid as fluid

paddle.enable_static()
x = fluid.data(name="data", shape=[None, 32, 32], dtype="float32")
dropped = fluid.layers.dropout(x, dropout_prob=0.5)