import paddle
paddle.device.get_all_custom_device_type()

# Case 1: paddlepaddle-gpu package installed, and no custom device registerd.
# Output: None

# Case 2: paddlepaddle-gpu package installed, and custom deivce 'CustomCPU' and 'CustomGPU' is registerd.
# Output: ['CustomCPU', 'CustomGPU']