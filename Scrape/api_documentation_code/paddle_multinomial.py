import paddle

paddle.seed(100) # on CPU device
x = paddle.rand([2,4])
print(x)
# [[0.5535528  0.20714243 0.01162981 0.51577556]
# [0.36369765 0.2609165  0.18905126 0.5621971 ]]

paddle.seed(200) # on CPU device
out1 = paddle.multinomial(x, num_samples=5, replacement=True)
print(out1)
# [[3 3 0 0 0]
# [3 3 3 1 0]]

# out2 = paddle.multinomial(x, num_samples=5)
# InvalidArgumentError: When replacement is False, number of samples
#  should be less than non-zero categories

paddle.seed(300) # on CPU device
out3 = paddle.multinomial(x, num_samples=3)
print(out3)
# [[3 0 1]
# [3 1 0]]