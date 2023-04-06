import paddle.fluid as fluid
import paddle
paddle.enable_static()
dataX = fluid.layers.data(name="dataX", append_batch_size = False, shape=[2, 5], dtype="float32")
dataY = fluid.layers.data(name="dataY", append_batch_size = False, shape=[5, 3], dtype="float32")
output = fluid.layers.mul(dataX, dataY,
                          x_num_col_dims = 1,
                          y_num_col_dims = 1)