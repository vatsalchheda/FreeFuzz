import paddle

paddle.fluid.contrib.mixed_precision.bf16.amp_utils.cast_parameters_to_bf16(paddle.CUDAPlace(0), 
paddle.fluid.default_main_program(), scope=None, to_bf16_var_names=None)
