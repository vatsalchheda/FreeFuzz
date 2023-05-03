import paddle
paddle.enable_static()
import paddle.fluid as fluid
import numpy as np

DATATYPE='float32'
input_data = np.array([[1.],[2.],[3.],[4.]]).astype(DATATYPE)
label_data = np.array([[3.],[3.],[4.],[4.]]).astype(DATATYPE)

x = fluid.data(name='input', shape=[None, 1], dtype=DATATYPE)
y = fluid.data(name='label', shape=[None, 1], dtype=DATATYPE)
loss = fluid.layers.huber_loss(input=x, label=y, delta=1.0)

place = fluid.CPUPlace()
#place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
HuberLoss, = exe.run(feed={'input':input_data ,'label':label_data}, fetch_list=[loss.name])
print(HuberLoss)  #[[1.5], [0.5], [0.5], [0. ]], dtype=float32