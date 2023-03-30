import paddle.fluid as fluid
import paddle
paddle.enable_static()

data = fluid.data(
    name="words", shape=[-1, 6, 3, 9], dtype="float32")
# shape of output: [-1, 3, 3, 9]
output = fluid.nets.glu(input=data, dim=1)