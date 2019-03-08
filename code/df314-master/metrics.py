import numpy as np
import tensorflow as tf
from keras import backend as K


def iou_metric(num_classes=2):
    def iou_metric_func(y_true, y_pred):
        def calculate(gt, pt):
            counts = np.zeros((num_classes, 2), dtype=np.float32)
            gt, pt = np.reshape(gt, -1), np.reshape(pt, -1)
            for i in range(num_classes):
                t, p = gt == i, pt == i
                counts[i, 0] += np.sum(np.logical_and(t, p))
                counts[i, 1] += np.sum(np.logical_or(t, p))

            joint, union = counts[:, 0], counts[:, 1]
            union[union == 0] = 1e-7

            ious = joint / union
            return np.mean(ious[1:])

        y_true, y_pred = K.argmax(y_true), K.argmax(y_pred)
        return tf.py_func(calculate, [y_true, y_pred], tf.float32)

    return iou_metric_func
