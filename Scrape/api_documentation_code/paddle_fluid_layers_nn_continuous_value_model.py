import paddle
paddle.enable_static()
import paddle.fluid as fluid
input = fluid.data(name="input", shape=[64, 1], dtype="int64")
label = fluid.data(name="label", shape=[64, 1], dtype="int64")
embed = fluid.layers.embedding(
                  input=input,
                  size=[100, 11],
                  dtype='float32')
ones = fluid.layers.fill_constant_batch_size_like(input=label, shape=[-1, 1], dtype="int64", value=1)
show_clk = fluid.layers.cast(fluid.layers.concat([ones, label], axis=1), dtype='float32')
show_clk.stop_gradient = True
input_with_cvm = fluid.layers.continuous_value_model(embed, show_clk, True)