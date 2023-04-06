import paddle
import paddle.fluid as fluid
paddle.enable_static()

# Correspond to the first example above, where label.shape is 4 and one_hot_label.shape is [4, 4].
label = fluid.data(name="label", shape=[4], dtype="int64")
one_hot_label = fluid.one_hot(input=label, depth=4)