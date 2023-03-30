import paddle
import paddle.fluid as fluid

paddle.enable_static()
main_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(main_prog, startup_prog):
    data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
    w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
    b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
    hidden_w = fluid.layers.matmul(x=data, y=w)
    hidden_b = fluid.layers.elementwise_add(hidden_w, b)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

# The first usage: use `vars` to set the saved variables.
var_list = [w, b]
path = "./my_paddle_vars"
fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                filename="vars_file")
# w and b will be save in a file named "var_file".

# The second usage: use `predicate` to select the saved variable.
def name_has_fc(var):
    res = "fc" in var.name
    return res
param_path = "./my_paddle_model"
fluid.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate = name_has_fc)
# all variables whose names contain "fc " are saved.