import paddle

paddle.enable_static()

with paddle.fluid.contrib.mixed_precision.bf16.amp_utils.bf16_guard():
  paddle.fluid.contrib.mixed_precision.bf16.amp_utils.AutoMixedPrecisionListsBF16(custom_fp32_list={'lstm'})