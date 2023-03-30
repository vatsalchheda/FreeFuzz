import paddle

t = paddle.to_tensor([0,1,2,3,4])
expectlist = t.tolist()
print(expectlist)   #[0, 1, 2, 3, 4]

expectlist = paddle.tolist(t)
print(expectlist)   #[0, 1, 2, 3, 4]