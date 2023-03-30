import paddle.fluid as fluid
import numpy as np

metric = fluid.metrics.Precision()

# generate the preds and labels

preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
         [0.2], [0.3], [0.5], [0.8], [0.6]]

labels = [[0], [1], [1], [1], [1],
          [0], [0], [0], [0], [0]]

preds = np.array(preds)
labels = np.array(labels)

metric.update(preds=preds, labels=labels)
numpy_precision = metric.eval()

print("expect precision: %.2f and got %.2f" % ( 3.0 / 5.0, numpy_precision))