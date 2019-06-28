"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.python.keras.layers import Conv2D, Layer
from tensorflow.python.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.python.keras import backend as K

import tensorflow as tf

"""
Reference : 

1. CRNN Paper, 
   An End-to-End Trainable Neural Network for Image-based Sequence Recognition 
   and its Applications to Scene Text Recognition

2. GRU Attention Decoder,
   Robust Scene Text Recognition With Automatic Rectification

"""


class ConvFeatureExtractor(Layer):
    """
    CRNN 중 Convolution Layers에 해당하는 Module Class

    | Layer Name | #maps | Filter | Stride | Padding |
    | ----       | ---   | -----  | ------ | -----   |
    | conv1      | 64    | (3,3)  | (1,1)  | same    |
    | maxpool1   | -     | (2,2)  | (2,2)  | same    |
    | conv2      | 128   | (3,3)  | (1,1)  | same    |
    | maxpool2   | -     | (2,2)  | (2,2)  | same    |
    | conv3      | 256   | (3,3)  | (1,1)  | same    |
    | conv4      | 256   | (3,3)  | (1,1)  | same    |
    | maxpool3   | -     | (2,1)  | (2,1)  | same    |
    | batchnorm1 | -     | -      | -      | -       |
    | conv5      | 512   | (3,3)  | (1,1)  | same    |
    | batchnorm2 | -     | -      | -      | -       |
    | conv6      | 512   | (3,3)  | (1,1)  | same    |
    | maxpool4   | -     | (2,1)  | (2,1)  | same    |
    | conv7      | 512   | (3,3)  | (1,1)  | valid   |


    특징
     1. Maxpool3와 Maxpool4는 Rectangular Pooling의 형태를 띄고 있음.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # for builing Layer and Weight
        self.conv1 = Conv2D(64, (3, 3), padding='same')
        self.maxpool1 = MaxPooling2D((2, 2), (2, 2), padding='same')
        self.conv2 = Conv2D(128, (3, 3), padding='same')
        self.maxpool2 = MaxPooling2D((2, 2), (2, 2), padding='same')
        self.conv3 = Conv2D(256, (3, 3), padding='same')
        self.conv4 = Conv2D(256, (3, 3), padding='same')
        self.maxpool3 = MaxPooling2D((2, 1), (2, 1), padding='same')
        self.batchnorm1 = BatchNormalization()
        self.conv5 = Conv2D(512, (3, 3), padding='same')
        self.batchnorm2 = BatchNormalization()
        self.conv6 = Conv2D(512, (3, 3), padding='same')
        self.maxpool4 = MaxPooling2D((2, 1), (2, 1), padding='same')
        self.conv7 = Conv2D(512, (2, 2), padding='valid')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool3(x)
        x = self.batchnorm1(x)
        x = self.conv5(x)
        x = self.batchnorm2(x)
        x = self.conv6(x)
        x = self.maxpool4(x)
        return self.conv7(x)


class Map2Sequence(Layer):
    """
    CRNN 중 CNN Layer의 출력값을 RNN Layer의 입력값으로 변환하는 Module Class

    CNN output shape  ->  RNN Input Shape

    (batch size, width, height, channels)
    -> (batch size, width, height * channels)

    * Caution
        이 때 batch size와 width는 입력 데이터에 따라 변하는 Dynamic Shape이고
        height와 channels는 Convolution Layer에 의해 정해진 Static Shape이다

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # Get the Dynamic Shape
        shape = tf.shape(inputs)
        batch_size = shape[0]
        f_width = shape[1]

        # Get the Static Shape
        _, _, f_height, f_num = inputs.shape.as_list()

        return tf.reshape(inputs,
                          shape=[batch_size, f_width,
                                 f_height * f_num])


class BLSTMEncoder(Layer):
    """
    CRNN 중 Recurrent Layers에 해당하는 Module Class
    Convolution Layer의 Image Feature Sequence를 Encoding하여,
    우리가 원하는 Text Feature Sequence로 만듦

    | Layer Name | #Hidden Units |
    | ----       | ------ |
    | Bi-LSTM1   | 256    |
    | Bi-LSTM2   | 256    |

    """
    def __init__(self, n_units=256, **kwargs):
        super().__init__(**kwargs)
        self.lstm1 = Bidirectional(LSTM(n_units, return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(n_units, return_sequences=True))

    def call(self, inputs, **kwargs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        return x


class CTCDecoder(Layer):
    """
    CRNN 중 Transcription Layer에 해당하는 Module Class

    * beam_width :
      클수록 탐색하는 Candidate Sequence가 많아져 정확도는 올라가나,
      연산 비용도 함께 올라가기 때문에 ACC <-> Speed의 Trade-Off 관계에 있음

    """
    def __init__(self, beam_width=100, **kwargs):
        self.beam_width = beam_width
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        max_length = shape[1, None]
        input_length = tf.tile(max_length, [batch_size])

        prediction, scores = K.ctc_decode(inputs, input_length, beam_width=self.beam_width)
        return [prediction, scores]


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


__all__ = ["ConvFeatureExtractor",
           "Map2Sequence",
           "BLSTMEncoder",
           "CTCDecoder",
           "ctc_loss"]


if __name__ == "__main__":
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import Input

    height = 32
    num_classes = 11172
    K.clear_session()

    # For Gray Scale Image & Dynamic width
    inputs = Input(shape=(height, None, 1))

    # (batch size, height, width, channels) -> (batch size, width, height, channels)
    transposed = tf.transpose(inputs, (0, 2, 1, 3))

    # CRNN Layer
    conv_maps = ConvFeatureExtractor(name='feature_extractor')(transposed)
    feature_seqs = Map2Sequence(name='map_to_sequence')(conv_maps)
    lstm_seqs = BLSTMEncoder()(feature_seqs)
    # 우리의 출력 형태는 class 수에 Blank Label을 하나 더해 #classes + 1 만큼을 출력
    output_seqs = Dense(num_classes+1, activation='softmax')(lstm_seqs)


    # 모델 구성하기
    # (1) 학습 모델 구성하기
    trainer = Model(inputs, output_seqs, name='trainer')

    # Caution :
    # CTC Loss의 경우 y_pred와 y_true의 형태가 다릅니다.
    # Keras는 y_pred과 y_true의 shape가 동일하다고 가정하기 때문에, 다른 경우
    # compile 시 target_tensor에 직접 y_true의 형태를 지정해주어야 합니다.
    y_true = tf.placeholder(dtype=tf.int32, shape=(None, None))

    trainer.compile('adam',
                    loss=ctc_loss,
                    target_tensors=[y_true])

    # (2) 예측 모델 구성하기
    predictions = CTCDecoder(beam_width=100)(output_seqs)
    predictor = Model(inputs, predictions, name='predictor')
