import paddle

case1 = paddle.randn([2, 3])
case2 = paddle.randn([3, 10, 10])
case3 = paddle.randn([3, 10, 5, 10])
data1 = paddle.trace(case1) # data1.shape = [1]
data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) # data2.shape = [3]
data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) # data2.shape = [3, 5]