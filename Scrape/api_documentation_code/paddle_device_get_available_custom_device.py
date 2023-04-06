import paddle
paddle.device.get_available_custom_device()

# Case 1: paddlepaddle-gpu package installed, and no custom device registerd.
# Output: None

# Case 2: paddlepaddle-gpu package installed, and custom deivce 'CustomCPU' and 'CustomGPU' is registerd.
# Output: ['CustomCPU', 'CustomGPU:0', 'CustomGPU:1']