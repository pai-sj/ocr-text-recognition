"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import get_custom_objects

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



get_custom_objects().update({'ctc_loss' : ctc_loss})