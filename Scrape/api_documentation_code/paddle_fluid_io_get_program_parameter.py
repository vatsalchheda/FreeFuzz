import paddle
import paddle.fluid as fluid

paddle.enable_static()
data = fluid.data(name="img", shape=[64, 784])
w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
list_para  = fluid.io.get_program_parameter(  fluid.default_main_program() )