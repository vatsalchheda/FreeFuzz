import paddle

paddle.fluid.contrib.mixed_precision.fp16_utils.cast_parameters_to_fp16(paddle.CUDAPlace(0), 
paddle.fluid.default_main_program(), scope=None, to_fp16_var_names=None)
