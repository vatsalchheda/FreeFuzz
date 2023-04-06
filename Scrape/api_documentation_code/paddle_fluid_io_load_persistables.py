import paddle
import paddle.fluid as fluid

paddle.enable_static()
exe = fluid.Executor(fluid.CPUPlace())
param_path = "./my_paddle_model"
prog = fluid.default_main_program()
fluid.io.load_persistables(executor=exe, dirname=param_path,
                           main_program=None)