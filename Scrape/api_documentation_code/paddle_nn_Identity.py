import paddle

input_tensor = paddle.randn(shape=[3, 2])
layer = paddle.nn.Identity()
out = layer(input_tensor)
# input_tensor: [[-0.32342386 -1.200079  ]
#                [ 0.7979031  -0.90978354]
#                [ 0.40597573  1.8095392 ]]
# out: [[-0.32342386 -1.200079  ]
#      [ 0.7979031  -0.90978354]
#      [ 0.40597573  1.8095392 ]]