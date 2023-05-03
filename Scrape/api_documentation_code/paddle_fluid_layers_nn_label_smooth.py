import paddle
paddle.enable_static()
import paddle.fluid as fluid
import paddle.fluid.layers as layers

label = layers.data(name="label", shape=[1], dtype="int32")
one_hot_label = layers.one_hot(input=label, depth=10)
smooth_label = layers.label_smooth(
    label=one_hot_label, epsilon=0.1, dtype="float32")