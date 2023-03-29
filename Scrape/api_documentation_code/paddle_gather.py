import paddle

input = paddle.to_tensor([[1,2],[3,4],[5,6]])
index = paddle.to_tensor([0,1])
output = paddle.gather(input, index, axis=0)
# expected output: [[1,2],[3,4]]