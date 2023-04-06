import paddle
paddle.set_device('cpu')

input = paddle.uniform([4, 3])
# [[0.56194401  -0.22450298  -0.10741806] # random
#  [0.36136317  0.23556745  0.88748658] # random
#  [0.18151939  0.80947340  -0.31078976] # random
#  [0.68886101  -0.14239830  -0.41297770]] # random
label = paddle.to_tensor([0, 1, 4, 5])
m = paddle.nn.HSigmoidLoss(3, 5)
out = m(input, label)
# [[2.42524505]
#  [1.74917245]
#  [3.14571381]
#  [2.34564662]]