import paddle.fluid as fluid
import paddle.fluid.layers as layers

x = layers.fill_constant(shape=[2, 3], dtype='float32', value=2.0)
condition = layers.reduce_max(x) < 1.0 # False
layers.Assert(condition, [x], 10, "example_assert_layer")

exe = fluid.Executor()
try:
    exe.run(fluid.default_main_program())
    # Print x and throws paddle.fluid.core.EnforceNotMet exception
    # Example printed message for x:
    #
    # Variable: fill_constant_0.tmp_0
    #   - lod: {}
    #   - place: CPUPlace()
    #   - shape: [2, 3]
    #   - layout: NCHW
    #   - dtype: float
    #   - data: [2 2 2 2 2 2]
except fluid.core.EnforceNotMet as e:
    print("Assert Exception Example")