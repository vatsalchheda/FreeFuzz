import paddle.fluid as fluid
fluid.install_check.run_check()

# If installed successfully, output may be
# Running Verify Fluid Program ...
# W0805 04:24:59.496919 35357 device_context.cc:268] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.2, Runtime API Version: 10.1
# W0805 04:24:59.505594 35357 device_context.cc:276] device: 0, cuDNN Version: 7.6.
# Your Paddle Fluid works well on SINGLE GPU or CPU.
# Your Paddle Fluid works well on MUTIPLE GPU or CPU.
# Your Paddle Fluid is installed successfully! Let's start deep Learning with Paddle Fluid now