import paddle
paddle.device.get_all_device_type()

# Case 1: paddlepaddle-cpu package installed, and no custom device registerd.
# Output: ['cpu']

# Case 2: paddlepaddle-gpu package installed, and no custom device registerd.
# Output: ['cpu', 'gpu']

# Case 3: paddlepaddle-cpu package installed, and custom deivce 'CustomCPU' is registerd.
# Output: ['cpu', 'CustomCPU']

# Case 4: paddlepaddle-gpu package installed, and custom deivce 'CustomCPU' and 'CustomGPU' is registerd.
# Output: ['cpu', 'gpu', 'CustomCPU', 'CustomGPU']