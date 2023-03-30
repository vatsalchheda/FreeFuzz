import paddle
import paddle.nn.functional as F

x = paddle.randn([2,3*2*2,12])
y = F.fold(x, output_sizes=[4, 5], kernel_sizes=2)
# y.shape = [2,3,4,5]