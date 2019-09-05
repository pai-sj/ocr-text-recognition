from tensorflow.python.keras.engine.base_layer import AddMetric, AddLoss
from tensorflow.python.keras.utils import get_custom_objects
from models.layers import *
from models.optimizer import *
from models.losses import *

# Custom Layer 구성하기
get_custom_objects().update({
    "ConvFeatureExtractor" : ConvFeatureExtractor,
    "ResidualConvFeatureExtractor": ResidualConvFeatureExtractor,
    "Map2Sequence" : Map2Sequence,
    "BGRUEncoder" : BGRUEncoder,
    "CTCDecoder" : CTCDecoder,
    "DotAttention": DotAttention,
    "JamoCompose": JamoCompose,
    'JamoDeCompose': JamoDeCompose,
    "JamoEmbedding": JamoEmbedding,
    "JamoClassifier": JamoClassifier,
    "TeacherForcing": TeacherForcing
})

# Custom Optimizer 구성하기
get_custom_objects().update({'AdamW': AdamW,
                             'RectifiedAdam': RectifiedAdam})

# Custom Loss 구성하기
get_custom_objects().update({
    'ctc_loss': ctc_loss,
    "JamoCategoricalCrossEntropy": JamoCategoricalCrossEntropy})

# BUGS!!!> Keras 기본 인자인데, 세팅이 안되어 있어서, save Model & Load Model에서
# 따로 지정해주어야 함
get_custom_objects().update({'AddMetric': AddMetric,
                             'AddLoss': AddLoss})
