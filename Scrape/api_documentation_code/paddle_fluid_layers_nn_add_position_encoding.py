import paddle

tensor = paddle.randn([16, 32, 64])
position_tensor = paddle.fluid.layers.add_position_encoding(
      input=tensor, alpha=1.0, beta=1.0)