# required: distributed
# Multi GPU, test_class_center_sample.py
import paddle
import paddle.distributed as dist
strategy = dist.fleet.DistributedStrategy()
dist.fleet.init(is_collective=True, strategy=strategy)
batch_size = 10
num_samples = 6
rank_id = dist.get_rank()
# num_classes of each GPU can be different, e.g num_classes_list = [10, 8]
num_classes_list = [10, 10]
num_classes = paddle.sum(paddle.to_tensor(num_classes_list))
label = paddle.randint(low=0, high=num_classes.item(), shape=[batch_size], dtype='int64')
label_list = []
dist.all_gather(label_list, label)
label = paddle.concat(label_list, axis=0)
remapped_label, sampled_class_index = paddle.nn.functional.class_center_sample(label, num_classes_list[rank_id], num_samples)

print(label)
print(remapped_label)
print(sampled_class_index)

#python -m paddle.distributed.launch --gpus=0,1 test_class_center_sample.py
# rank 0 output:
#Tensor(shape=[20], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#       [10, 17, 15, 11, 9 , 12, 18, 18, 17, 18, 19, 2 , 8 , 13, 11, 13, 9 , 10, 0 , 4 ])
#Tensor(shape=[20], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#       [6 , 11, 10, 7 , 4 , 8 , 12, 12, 11, 12, 13, 1 , 3 , 9 , 7 , 9 , 4 , 6 , 0 , 2 ])
#Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#       [0, 2, 4, 8, 9, 3])

# rank 1 output:
#Tensor(shape=[20], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
#       [10, 17, 15, 11, 9 , 12, 18, 18, 17, 18, 19, 2 , 8 , 13, 11, 13, 9 , 10, 0 , 4 ])
#Tensor(shape=[20], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
#       [6 , 11, 10, 7 , 4 , 8 , 12, 12, 11, 12, 13, 1 , 3 , 9 , 7 , 9 , 4 , 6 , 0 , 2 ])
#Tensor(shape=[7], dtype=int64, place=CUDAPlace(1), stop_gradient=True,
#       [0, 1, 2, 3, 5, 7, 8])