import paddle
gain = paddle.nn.initializer.calculate_gain('tanh') # 5.0 / 3
gain = paddle.nn.initializer.calculate_gain('leaky_relu', param=1.0) # 1.0 = math.sqrt(2.0 / (1+param^2))
initializer = paddle.nn.initializer.Orthogonal(gain)