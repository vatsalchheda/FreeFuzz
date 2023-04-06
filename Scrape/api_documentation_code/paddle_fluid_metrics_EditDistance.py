import paddle.fluid as fluid
import numpy as np

# suppose that batch_size is 128
batch_size = 128

# init the edit distance manager
distance_evaluator = fluid.metrics.EditDistance("EditDistance")

# generate the edit distance across 128 sequence pairs, the max distance is 10 here
edit_distances_batch0 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
seq_num_batch0 = batch_size

distance_evaluator.update(edit_distances_batch0, seq_num_batch0)
avg_distance, wrong_instance_ratio = distance_evaluator.eval()
print("the average edit distance for batch0 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))

edit_distances_batch1 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
seq_num_batch1 = batch_size

distance_evaluator.update(edit_distances_batch1, seq_num_batch1)
avg_distance, wrong_instance_ratio = distance_evaluator.eval()
print("the average edit distance for batch0 and batch1 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))

distance_evaluator.reset()

edit_distances_batch2 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
seq_num_batch2 = batch_size

distance_evaluator.update(edit_distances_batch2, seq_num_batch2)
avg_distance, wrong_instance_ratio = distance_evaluator.eval()
print("the average edit distance for batch2 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))