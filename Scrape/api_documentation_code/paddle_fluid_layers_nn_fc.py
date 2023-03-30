import paddle.fluid as fluid
import paddle
paddle.enable_static()
# when input is single tensor
data = fluid.data(name="data", shape=[-1, 32], dtype="float32")
fc = fluid.layers.fc(input=data, size=1000, act="tanh")

# when input are multiple tensors
data_1 = fluid.data(name="data_1", shape=[-1, 32], dtype="float32")
data_2 = fluid.data(name="data_2", shape=[-1, 36], dtype="float32")
fc = fluid.layers.fc(input=[data_1, data_2], size=1000, act="tanh")