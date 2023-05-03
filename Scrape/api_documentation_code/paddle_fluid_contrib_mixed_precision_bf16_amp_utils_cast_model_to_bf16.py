import paddle

paddle.fluid.contrib.mixed_precision.bf16.amp_utils.cast_model_to_bf16(paddle.fluid.default_main_program(), startup_prog=None, amp_lists=None, use_bf16_guard=True)