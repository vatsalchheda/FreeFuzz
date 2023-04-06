import paddle.fluid as fluid

lr = fluid.layers.create_global_var(
    shape=[1],
    value=0.0,
    dtype='float32',
    persistable=True,
    name="learning_rate")
zero_var = fluid.layers.fill_constant(
    shape=[1], dtype='float32', value=0.0)
one_var = fluid.layers.fill_constant(
    shape=[1], dtype='float32', value=1.0)
two_var = fluid.layers.fill_constant(
    shape=[1], dtype='float32', value=2.0)

global_step = fluid.layers.autoincreased_step_counter(counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)

with fluid.layers.control_flow.Switch() as switch:
    with switch.case(global_step == zero_var):
        fluid.layers.assign(input=one_var, output=lr)
    with switch.default():
        fluid.layers.assign(input=two_var, output=lr)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[lr])
print(res) # [array([1.], dtype=float32)]