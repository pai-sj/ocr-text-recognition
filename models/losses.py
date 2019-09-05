"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from models.layers import JamoDeCompose
from .jamo import 초성, 중성, 종성


def ctc_loss(y_true, y_pred):
    """
    Runs CTC Loss Algorithm on each batch element

    :param y_true: tensor (samples, max_string_length) containing the truth labels.
    :param y_pred: tensor (samples, time_steps, num_categories) containing the prediction, or output of the softmax.

    * caution

    input_length : tensor (samples, 1) containing the sequence length for each batch item in y_pred
    label_length : tensor (samples, 1) containing the sequence length for each batch item in y_true

    y_true는 [3,7,12,1,2,-1,-1,-1,-1] 와 같은 형태로 구성되어 있음. -1은 Blank를 의미
    처음 등장하는 -1의 인덱스가 y_true의 sequnece length와 동일

    y_pred의 총 width와 input_length는 동일

    """

    # Get the Length of Prediction
    shape = tf.shape(y_pred)
    batch_size = shape[0]
    max_length = shape[1, None, None]
    input_length = tf.tile(max_length, [batch_size, 1])

    # Get the Length of Input
    label_length = tf.argmin(y_true, axis=-1)[:, None]

    return K.ctc_batch_cost(y_true, y_pred,
                            input_length, label_length)


def masking_sparse_categorical_crossentropy(mask_value):
    """
    Runs sparse Categorical Crossentropy Loss Algorithm on each batch element Without Masking Value

    :param mask_value: masking value for preventing Back Propagation
    :return:
    """
    mask_value = K.variable(mask_value)

    def loss(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        mask = K.equal(y_true, mask_value)
        mask = 1 - K.cast(mask, K.floatx())
        y_true = y_true * mask

        loss = K.sparse_categorical_crossentropy(y_true, y_pred) * mask
        return K.sum(loss) / K.sum(mask)

    return loss


def jamo_categorical_crossentropy(y_true, y_pred):
    y_true = K.cast(y_true, tf.int32)
    eos_mask = tf.not_equal(y_true, ord('\n'))
    blank_mask = tf.not_equal(y_true, -1)

    y_true_초성 = ((y_true - 44032) // 28) // 21
    y_true_초성 = tf.where(eos_mask, y_true_초성, tf.ones_like(y_true_초성)*len(초성))
    y_true_초성 = tf.where(blank_mask, y_true_초성, tf.zeros_like(y_true_초성))

    y_true_중성 = ((y_true - 44032) // 28) % 21
    y_true_중성 = tf.where(eos_mask, y_true_중성, tf.ones_like(y_true_중성)*len(중성))
    y_true_중성 = tf.where(blank_mask, y_true_중성, tf.zeros_like(y_true_중성))

    y_true_종성 = (y_true - 44032) % 28
    y_true_종성 = tf.where(eos_mask, y_true_종성, tf.ones_like(y_true_종성)*len(종성))
    y_true_종성 = tf.where(blank_mask, y_true_종성, tf.zeros_like(y_true_종성))

    y_pred_초성,  y_pred_중성, y_pred_종성 = tf.split(y_pred,
                                                [len(초성) + 1, len(중성) + 1, len(종성) + 1],
                                                axis=-1)

    mask = tf.cast(blank_mask, dtype=K.floatx())
    loss_초성 = K.sparse_categorical_crossentropy(y_true_초성, y_pred_초성) * mask
    loss_중성 = K.sparse_categorical_crossentropy(y_true_중성, y_pred_중성) * mask
    loss_종성 = K.sparse_categorical_crossentropy(y_true_종성, y_pred_종성) * mask

    return  K.sum(loss_초성 + loss_중성 + loss_종성) / K.sum(mask)


class JamoCategoricalCrossEntropy(Layer):
    def __init__(self, blank_value, **kwargs):
        self.blank_value = blank_value
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        y_true = inputs[0]
        y_pred = inputs[1]

        y_true = K.cast(y_true, tf.int32)
        blank_mask = tf.not_equal(y_true, tf.to_int32(self.blank_value))

        y_true_초성, y_true_중성, y_true_종성 = JamoDeCompose()(y_true)
        y_pred_초성,  y_pred_중성, y_pred_종성 = tf.split(y_pred,
                                                    [len(초성) + 1, len(중성) + 1, len(종성) + 1],
                                                    axis=-1)

        mask = tf.cast(blank_mask, dtype=K.floatx())
        loss_초성 = K.sparse_categorical_crossentropy(y_true_초성, y_pred_초성) * mask
        loss_중성 = K.sparse_categorical_crossentropy(y_true_중성, y_pred_중성) * mask
        loss_종성 = K.sparse_categorical_crossentropy(y_true_종성, y_pred_종성) * mask

        mask = K.sum(mask, axis=1)
        loss_jamo = K.sum(loss_초성+loss_중성+loss_종성, axis=1)
        return loss_jamo / mask


__all__ = [
    "ctc_loss",
    "jamo_categorical_crossentropy",
    "JamoCategoricalCrossEntropy"
]
