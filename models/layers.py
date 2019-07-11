"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.python.keras.layers import Conv2D, Layer
from tensorflow.python.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.python.keras.layers import LayerNormalization
from tensorflow.python.keras.layers import Bidirectional, LSTM
from tensorflow.python.keras.layers import Softmax, Dense
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import get_custom_objects
import tensorflow as tf

"""
Reference : 

1. CRNN Paper, 
   An End-to-End Trainable Neural Network for Image-based Sequence Recognition 
   and its Applications to Scene Text Recognition

2. GRU Attention Decoder,
   Robust Scene Text Recognition With Automatic Rectification

3. KaKao OCR Blog,
   카카오 OCR 시스템 구성과 모델, https://brunch.co.kr/@kakao-it/318
"""


class ConvFeatureExtractor(Layer):
    """
    CRNN 중 Convolution Layers에 해당하는 Module Class

    | Layer Name | #maps | Filter | Stride | Padding | activation |
    | ----       | ---   | -----  | ------ | -----   | ---- |
    | conv1      | 64    | (3,3)  | (1,1)  | same    | relu |
    | maxpool1   | -     | (2,2)  | (2,2)  | same    | -    |
    | conv2      | 128   | (3,3)  | (1,1)  | same    | relu |
    | maxpool2   | -     | (2,2)  | (2,2)  | same    | -    |
    | conv3      | 256   | (3,3)  | (1,1)  | same    | relu |
    | conv4      | 256   | (3,3)  | (1,1)  | same    | relu |
    | maxpool3   | -     | (2,1)  | (2,1)  | same    | -    |
    | batchnorm1 | -     | -      | -      | -       | -    |
    | conv5      | 512   | (3,3)  | (1,1)  | same    | relu |
    | batchnorm2 | -     | -      | -      | -       | -    |
    | conv6      | 512   | (3,3)  | (1,1)  | same    | relu |
    | maxpool4   | -     | (2,1)  | (2,1)  | same    | -    |
    | conv7      | 512   | (3,3)  | (1,1)  | valid   | relu |


    특징
     1. Maxpool3와 Maxpool4는 Rectangular Pooling의 형태를 띄고 있음.

    """
    def __init__(self, n_hidden=64, **kwargs):
        self.n_hidden = n_hidden
        super().__init__(**kwargs)
        # for builing Layer and Weight
        self.conv1 = Conv2D(n_hidden, (3, 3), activation='relu', padding='same')
        self.maxpool1 = MaxPooling2D((2, 2), (2, 2), padding='same')
        self.conv2 = Conv2D(n_hidden*2, (3, 3), activation='relu', padding='same')
        self.maxpool2 = MaxPooling2D((2, 2), (2, 2), padding='same')
        self.conv3 = Conv2D(n_hidden*4, (3, 3), activation='relu', padding='same')
        self.conv4 = Conv2D(n_hidden*4, (3, 3), activation='relu', padding='same')
        self.maxpool3 = MaxPooling2D((2, 1), (2, 1), padding='same')
        self.batchnorm1 = BatchNormalization()
        self.conv5 = Conv2D(n_hidden*8, (3, 3), activation='relu', padding='same')
        self.batchnorm2 = BatchNormalization()
        self.conv6 = Conv2D(n_hidden*8, (3, 3), activation='relu', padding='same')
        self.maxpool4 = MaxPooling2D((2, 1), (2, 1), padding='same')
        self.conv7 = Conv2D(n_hidden*8, (2, 2), activation='relu', padding='valid')

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

    def get_config(self):
        config = {
            "n_hidden": self.n_hidden
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResidualConvFeatureExtractor(Layer):
    """
    Residual Block & KAKAO OCR Like Text Recognition Model
    [ conv2d - layer norm - conv2d - layer norm - maxpool ] * 3

    특징
     1. Batch Normalization 대신 Layer Normalization을 이용(높이와 채널 축으로만 진행)
        RNN 모델에서는 주로 BN 보다는 LN을 쓴다고 함.
        Layer Normalization은 Hidden unit들에 대해서 Mean과 Variance를 구함
     2. KAKAO와 달리 Block 수를 4개로 진행 (보다 넓은 범위를 탐색하기 위함)
     3. Residual Block을 두어서 보다 빠르게 학습가능하도록 설정

    """
    def __init__(self, n_hidden=64, **kwargs):
        self.n_hidden = n_hidden
        super().__init__(**kwargs)
        # for builing Layer and Weight
        self.conv1_1 = Conv2D(n_hidden, (3, 3), activation='relu', padding='same')
        self.lnorm1_1 = LayerNormalization(axis=(1, 3)) # Normalizing height & Channel
        self.conv1_2 = Conv2D(n_hidden, (3, 3), padding='same')
        self.lnorm1_2 = LayerNormalization(axis=(1, 3)) # Normalizing height & Channel
        self.maxpool1 = MaxPooling2D((2, 2), (2, 2), padding='same')

        self.conv2_skip = Conv2D(n_hidden*2, (1, 1), activation='relu', padding='same')
        self.conv2_1 = Conv2D(n_hidden*2, (3, 3), activation='relu', padding='same')
        self.lnorm2_1 = LayerNormalization(axis=(1, 3)) # Normalizing height & Channel
        self.conv2_2 = Conv2D(n_hidden*2, (3, 3), padding='same')
        self.lnorm2_2 = LayerNormalization(axis=(1, 3)) # Normalizing height & Channel
        self.maxpool2 = MaxPooling2D((2, 2), (2, 2), padding='same')

        self.conv3_skip = Conv2D(n_hidden * 4, (1, 1), activation='relu', padding='same')
        self.conv3_1 = Conv2D(n_hidden*4, (3, 3), activation='relu', padding='same')
        self.lnorm3_1 = LayerNormalization(axis=(1, 3)) # Normalizing height & Channel
        self.conv3_2 = Conv2D(n_hidden*4, (3, 3), padding='same')
        self.lnorm3_2 = LayerNormalization(axis=(1, 3)) # Normalizing height & Channel
        self.maxpool3 = MaxPooling2D((2, 1), (2, 1), padding='same')

        self.conv4_skip = Conv2D(n_hidden * 4, (1, 1), activation='relu', padding='same')
        self.conv4_1 = Conv2D(n_hidden*4, (3, 3), activation='relu', padding='same')
        self.lnorm4_1 = LayerNormalization(axis=(1, 3)) # Normalizing height & Channel
        self.conv4_2 = Conv2D(n_hidden*4, (3, 3), padding='same')
        self.lnorm4_2 = LayerNormalization(axis=(1, 3)) # Normalizing height & Channel
        self.maxpool4 = MaxPooling2D((2, 1), (2, 1), padding='same')
        self.built = True

    def call(self, inputs, **kwargs):
        x = self.conv1_1(inputs)
        x = self.lnorm1_1(x)
        x = self.conv1_2(x)
        x = self.lnorm1_2(x)
        x = self.maxpool1(x)

        skip = self.conv2_skip(x)
        x = self.conv2_1(x)
        x = self.lnorm2_1(x)
        x = self.conv2_2(x)
        x = self.lnorm2_2(x)
        x = K.relu(skip + x)
        x = self.maxpool2(x)

        skip = self.conv3_skip(x)
        x = self.conv3_1(x)
        x = self.lnorm3_1(x)
        x = self.conv3_2(x)
        x = self.lnorm3_2(x)
        x = K.relu(skip + x)
        x = self.maxpool3(x)

        skip = self.conv4_skip(x)
        x = self.conv4_1(x)
        x = self.lnorm4_1(x)
        x = self.conv4_2(x)
        x = self.lnorm4_2(x)
        x = K.relu(skip + x)
        x = self.maxpool4(x)
        return x

    def get_config(self):
        config = {
            "n_hidden": self.n_hidden
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Map2Sequence(Layer):
    """ CNN Layer의 출력값을 RNN Layer의 입력값으로 변환하는 Module Class
    Transpose & Reshape을 거쳐서 진행

    CNN output shape  ->  RNN Input Shape

    (batch size, height, width, channels)
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
        f_width = shape[2]

        # Get the Static Shape
        _, f_height, _, f_num = inputs.shape.as_list()
        inputs = K.permute_dimensions(inputs, (0, 2, 1, 3))
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
        self.n_units = n_units
        super().__init__(**kwargs)
        self.lstm1 = Bidirectional(LSTM(n_units, return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(n_units, return_sequences=True))

    def call(self, inputs, **kwargs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        return x

    def get_config(self):
        config = {
            "n_units": self.n_units
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

    def get_config(self):
        config = {
            "beam_width": self.beam_width
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DotAttention(Layer):
    """ General Dot-Product Attention Network (Luong, 2015)

    * n_state :
       if n_state is None, Dot-Product Attention(s_t * h_i)
       if n_state is number, general Dot-Product Attention(s_t * W_a * h_i)

    """
    def __init__(self, n_state=None, **kwargs):
        super().__init__(**kwargs)
        self.n_state = n_state
        if isinstance(self.n_state, int):
            self.dense1 = Dense(self.n_state)

    def call(self, inputs, **kwargs):
        states_encoder = inputs[0]
        states_decoder = inputs[1]

        # (0) adjust the size of encoder state to the size of decoder state
        if isinstance(self.n_state, int):
            states_encoder = self.dense1(states_encoder)

        # (1) Calculate Score
        expanded_states_encoder = states_encoder[:, None, ...]
        # >>> (batch size, 1, length of encoder sequence, num hidden)
        expanded_states_decoder = states_decoder[..., None, :]
        # >>> (batch size, length of decoder sequence, 1, num hidden)
        score = K.sum(expanded_states_encoder * expanded_states_decoder,
                      axis=-1)
        # >>> (batch size, length of decoder input, length of encoder input)
        # (2) Normalize score
        attention = Softmax(axis=-1, name='attention')(score)

        # (3) Calculate Context Vector
        context = K.sum(expanded_states_encoder * attention[..., None], axis=2)
        # >>> (batch size, length of decoder input, num hidden)

        return context, attention

    def get_config(self):
        config = {
            "n_state": self.n_state
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


__all__ = ["ConvFeatureExtractor",
           "ResidualConvFeatureExtractor",
           "Map2Sequence",
           "BLSTMEncoder",
           "CTCDecoder",
           "DotAttention"]

get_custom_objects().update({
    "ConvFeatureExtractor" : ConvFeatureExtractor,
    "ResidualConvFeatureExtractor": ResidualConvFeatureExtractor,
    "Map2Sequence" : Map2Sequence,
    "BLSTMEncoder" : BLSTMEncoder,
    "CTCDecoder" : CTCDecoder,
    "DotAttention": DotAttention,
})


