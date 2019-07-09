"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import backend as K
from .layers import ConvFeatureExtractor, Map2Sequence, BLSTMEncoder, CTCDecoder
from .losses import ctc_loss


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
