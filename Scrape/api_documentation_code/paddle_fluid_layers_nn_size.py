import paddle
import paddle.fluid.layers as layers
paddle.enable_static()

input = layers.data(
    name="input", shape=[3, 100], dtype="float32", append_batch_size=False)
rank = layers.size(input) # 300