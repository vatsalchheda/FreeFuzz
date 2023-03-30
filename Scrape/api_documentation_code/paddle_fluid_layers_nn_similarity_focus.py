import paddle.fluid as fluid
import paddle
paddle.enable_static()
data = fluid.data(
    name='data', shape=[-1, 3, 2, 2], dtype='float32')
fluid.layers.similarity_focus(input=data, axis=1, indexes=[0])