from paddle import fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph import LSTMCell
import numpy as np
batch_size = 64
input_size = 128
hidden_size = 256
step_input_np = np.random.uniform(-0.1, 0.1, (
    batch_size, input_size)).astype('float64')
pre_hidden_np = np.random.uniform(-0.1, 0.1, (
    batch_size, hidden_size)).astype('float64')
pre_cell_np = np.random.uniform(-0.1, 0.1, (
    batch_size, hidden_size)).astype('float64')
if core.is_compiled_with_cuda():
    place = core.CUDAPlace(0)
else:
    place = core.CPUPlace()
with fluid.dygraph.guard(place):
    cudnn_lstm = LSTMCell(hidden_size, input_size)
    step_input_var = fluid.dygraph.to_variable(step_input_np)
    pre_hidden_var = fluid.dygraph.to_variable(pre_hidden_np)
    pre_cell_var = fluid.dygraph.to_variable(pre_cell_np)
    new_hidden, new_cell = cudnn_lstm(step_input_var, pre_hidden_var, pre_cell_var)