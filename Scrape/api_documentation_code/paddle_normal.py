import paddle

out1 = paddle.normal(shape=[2, 3])
# [[ 0.17501129  0.32364586  1.561118  ]  # random
#  [-1.7232178   1.1545963  -0.76156676]]  # random

mean_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
out2 = paddle.normal(mean=mean_tensor)
# [ 0.18644847 -1.19434458  3.93694787]  # random

std_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
out3 = paddle.normal(mean=mean_tensor, std=std_tensor)
# [1.00780561 3.78457445 5.81058198]  # random