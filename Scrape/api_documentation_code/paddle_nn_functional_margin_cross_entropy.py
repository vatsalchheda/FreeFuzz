# required: distributed
# Multi GPU, test_margin_cross_entropy.py
import paddle
import paddle.distributed as dist
strategy = dist.fleet.DistributedStrategy()
dist.fleet.init(is_collective=True, strategy=strategy)
rank_id = dist.get_rank()
m1 = 1.0
m2 = 0.5
m3 = 0.0
s = 64.0
batch_size = 2
feature_length = 4
num_class_per_card = [4, 8]
num_classes = paddle.sum(paddle.to_tensor(num_class_per_card))

label = paddle.randint(low=0, high=num_classes.item(), shape=[batch_size], dtype='int64')
label_list = []
dist.all_gather(label_list, label)
label = paddle.concat(label_list, axis=0)

X = paddle.randn(
    shape=[batch_size, feature_length],
    dtype='float64')
X_list = []
dist.all_gather(X_list, X)
X = paddle.concat(X_list, axis=0)
X_l2 = paddle.sqrt(paddle.sum(paddle.square(X), axis=1, keepdim=True))
X = paddle.divide(X, X_l2)

W = paddle.randn(
    shape=[feature_length, num_class_per_card[rank_id]],
    dtype='float64')
W_l2 = paddle.sqrt(paddle.sum(paddle.square(W), axis=0, keepdim=True))
W = paddle.divide(W, W_l2)

logits = paddle.matmul(X, W)
loss, softmax = paddle.nn.functional.margin_cross_entropy(
    logits, label, margin1=m1, margin2=m2, margin3=m3, scale=s, return_softmax=True, reduction=None)

print(logits)
print(label)
print(loss)
print(softmax)

# python -m paddle.distributed.launch --gpus=0,1 test_margin_cross_entropy.py
## for rank0 input
#Tensor(shape=[4, 4], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
#       [[ 0.32888934,  0.02408748, -0.02763289,  0.18173063],
#        [-0.52893978, -0.10623845, -0.21596515, -0.06432517],
#        [-0.00536345, -0.03924667,  0.66735314, -0.28640926],
#        [-0.09907366, -0.48534973, -0.10365338, -0.39472322]])
#Tensor(shape=[4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#       [11, 1 , 10, 11])

## for rank1 input
#Tensor(shape=[4, 8], dtype=float64, place=CUDAPlace(1), stop_gradient=True,
#       [[ 0.68654754,  0.28137170,  0.69694954, -0.60923933, -0.57077653,  0.54576703, -0.38709028,  0.56028204],
#        [-0.80360371, -0.03042448, -0.45107338,  0.49559349,  0.69998950, -0.45411693,  0.61927630, -0.82808600],
#        [ 0.11457570, -0.34785879, -0.68819499, -0.26189226, -0.48241491, -0.67685711,  0.06510185,  0.49660849],
#        [ 0.31604851,  0.52087884,  0.53124749, -0.86176582, -0.43426329,  0.34786144, -0.10850784,  0.51566383]])
#Tensor(shape=[4], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
#       [11, 1 , 10, 11])

## for rank0 output
#Tensor(shape=[4, 1], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
#       [[38.96608230],
#        [81.28152394],
#        [69.67229865],
#        [31.74197251]])
#Tensor(shape=[4, 4], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
#       [[0.00000000, 0.00000000, 0.00000000, 0.00000000],
#        [0.00000000, 0.00000000, 0.00000000, 0.00000000],
#        [0.00000000, 0.00000000, 0.99998205, 0.00000000],
#        [0.00000000, 0.00000000, 0.00000000, 0.00000000]])
## for rank1 output
#Tensor(shape=[4, 1], dtype=float64, place=CUDAPlace(1), stop_gradient=True,
#       [[38.96608230],
#        [81.28152394],
#        [69.67229865],
#        [31.74197251]])
#Tensor(shape=[4, 8], dtype=float64, place=CUDAPlace(1), stop_gradient=True,
#       [[0.33943993, 0.00000000, 0.66051859, 0.00000000, 0.00000000, 0.00004148, 0.00000000, 0.00000000],
#        [0.00000000, 0.00000000, 0.00000000, 0.00000207, 0.99432097, 0.00000000, 0.00567696, 0.00000000],
#        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00001795],
#        [0.00000069, 0.33993085, 0.66006319, 0.00000000, 0.00000000, 0.00000528, 0.00000000, 0.00000000]])