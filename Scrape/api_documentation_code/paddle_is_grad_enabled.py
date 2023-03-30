import paddle

# Dygraph gradient calculation mode is enabled by default.
paddle.is_grad_enabled() # True

with paddle.set_grad_enabled(False):
    paddle.is_grad_enabled() # False

paddle.enable_static()
paddle.is_grad_enabled() # False