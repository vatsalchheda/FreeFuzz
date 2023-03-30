import paddle
import paddle.fluid as fluid
paddle.enable_static()

# example 1:
input = fluid.data(name="input", shape=[1, 3], dtype='float32')
out_1 = fluid.layers.uniform_random_batch_size_like(input, [2, 4]) # out_1.shape=[1, 4]

# example 2:
out_2 = fluid.layers.uniform_random_batch_size_like(input, [2, 4], input_dim_idx=1, output_dim_idx=1) # out_2.shape=[2, 3]