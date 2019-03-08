import tensorflow as tf
from keras import backend as K


def focal_loss(gamma=2., per_image=True, num_classes=2, loss_weight=None):
    """
    Use Focal Loss, check out for more to link https://arxiv.org/abs/1708.02002
    :param gamma: The parameter gamma
    :param per_image: Process for each image
    :return: Loss
    """
    def focal_loss_fixed(y_true, y_pred):
        if loss_weight is not None:
            assert  len(loss_weight) == num_classes
            klossweight = K.stack(loss_weight)
        else:
            klossweight = None

        def calculate_loss(t_p):
            lss = []
            gt, pt = t_p
            for i in range(1, num_classes):
                t, p = gt[:, :, i], pt[:, :, i]
                t, p = K.flatten(t), K.flatten(p)
                p = K.clip(p, K.epsilon(), 1 - K.epsilon())

                p_t = tf.where(tf.equal(t, 1), p, 1. - p)
                alpha = K.sum(t) / K.sum(K.ones_like(t))
                alpha_t = tf.where(tf.equal(t, 1), K.ones_like(p) * (1. - alpha), K.ones_like(p) * alpha)
                ls = -alpha_t * K.pow((1 - p_t), gamma) * K.log(p_t)
                lss.append(K.sum(ls) * klossweight[i])
            return K.sum(K.stack(lss))

        if per_image:
            losses = tf.map_fn(calculate_loss, (y_true, y_pred), dtype=tf.float32)
            loss = tf.reduce_mean(losses)
        else:
            loss = calculate_loss((y_true, y_pred))

        return loss
    return focal_loss_fixed
