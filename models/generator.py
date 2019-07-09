from tensorflow.python.keras.utils import Sequence
from hgtk.text import compose, decompose
import numpy as np
import pandas as pd

가 = ord('가')
힣 = ord('힣')

KOR_CHARS = [chr(idx) for idx in range(가,힣+1)]
KOR2IDX = { char : idx for idx, char in enumerate(KOR_CHARS )}

# 한글 자모자를 인덱스로 만드는 Map 구현
초성 = (
    u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', u'ㅃ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

중성 = (
    u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', u'ㅗ', u'ㅘ',
    u'ㅙ', u'ㅚ', u'ㅛ', u'ㅜ', u'ㅝ', u'ㅞ', u'ㅟ', u'ㅠ', u'ㅡ', u'ㅢ', u'ㅣ'
)

종성 = (
    u'', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ', u'ㄺ',
    u'ㄻ', u'ㄼ', u'ㄽ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

미포함종성 = tuple(set(종성) - set(초성))

# 초성, 중성, 종성, 그리고 "ᴥ","\n"(EOS Token), " "(Blank Token)을 포함
JAMOS = list(초성 + 중성 + 미포함종성 + ("ᴥ","\n","",))
# jamo에 매칭되는 인덱스
JAMO2IDX = {jamo : idx for idx, jamo in enumerate(JAMOS)}


class OCRGenerator(Sequence):
    "Generates OCR TEXT Recognition Dataset for Keras"

    def __init__(self, dataset, char_list=None,
                 batch_size=32, blank_value=-1, shuffle=True):
        """
        Initialization

        param
        :param dataset : instance of class 'OCRDataset'
        :param char_list : unique character list (for Embedding)
        :param batch_size : the number of batch
        :param blank_value : the value of `blank` label
        :param shuffle : whether shuffle dataset or not
        """
        self.dataset = dataset
        if char_list is None:
            self.char_list = KOR_CHARS
            self.char2idx = KOR2IDX
        else:
            self.char_list = char_list
            self.char2idx = {char: idx
                             for idx, char
                             in enumerate(self.char_list)}

        self.batch_size = batch_size
        self.max_length = self.dataset.max_word + 1 # With blank time step for Last label
        self.blank_value = blank_value
        self.shuffle = shuffle
        self.num_classes = len(self.char_list) + 1 # With Blank Token
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, texts = self.dataset[self.batch_size * index:
                                     self.batch_size * (index + 1)]
        # label sequence
        labels = np.ones([self.batch_size, self.max_length], np.int32)
        labels *= -1  # BLANK Token value : -1
        for idx, text in enumerate(texts):
            labels[idx, :len(text)] = text2label(text, self.char2idx)
        return images, labels

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()


class OCRSeq2SeqGenerator(Sequence):
    "Generates OCR TEXT Recognition Dataset for Keras"

    def __init__(self, dataset, char_list=None,
                 batch_size=32, blank_value=-1, shuffle=True,
                 return_initial_state=True, state_size=512):
        """
        Initialization

        param
        :param dataset : instance of class 'OCRDataset'
        :param char_list : unique character list (for Embedding)
        :param batch_size : the number of batch
        :param blank_value : the value of `blank` label
        :param shuffle : whether shuffle dataset or not
        :param return_initial_state : Whether return Initial state(Zero state) or not
        :param state_size : if return_initial_state is True, the size of initial state
        """
        self.dataset = dataset
        if char_list is None:
            self.char_list = KOR_CHARS
            self.char2idx = KOR2IDX
        else:
            self.char_list = char_list
            self.char2idx = {char: idx
                             for idx, char
                             in enumerate(self.char_list)}

        self.batch_size = batch_size
        self.max_length = self.dataset.max_word + 1 # With <EOS> time step for Last label
        self.blank_value = blank_value
        self.shuffle = shuffle
        self.num_classes = len(self.char_list) + 1 # With <EOS> Token
        self.return_initial_state = return_initial_state
        self.state_size = state_size
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, texts = self.dataset[self.batch_size * index:
                                     self.batch_size * (index + 1)]
        # label sequence
        labels = np.ones([self.batch_size, self.max_length], np.int32)
        labels *= -1  # BLANK Token value : -1
        for idx, text in enumerate(texts):
            labels[idx, :len(text)] = text2label(text, self.char2idx)
            labels[idx, len(text)] = self.num_classes # <EOS> Token

        target_inputs = np.roll(labels, 1, axis=1)
        target_inputs[:, 0] = self.num_classes # <EOS> Token
        target_inputs[target_inputs==-1] = self.num_classes # <EOS> Token

        X = {
            "images" : images,
            "decoder_inputs" : target_inputs
        }
        # return initial state
        if self.return_initial_state:
            batch_size = images.shape[0]
            X['decoder_state'] = np.zeros([batch_size, self.state_size])

        Y = {
            "output_seqs" : labels
        }

        return X, Y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()


def text2label(text, char2idx=KOR2IDX):
    return np.array([char2idx[char] for char in text])


class JAMOSeq2SeqGenerator(Sequence):
    "Generates OCR TEXT Recognition Dataset for Keras"

    def __init__(self, dataset, batch_size=32,
                 blank_value=-1, shuffle=True,
                 return_initial_state=True, state_size=512):
        """
        Initialization

        param
        :param dataset : instance of class 'OCRDataset'
        :param batch_size : the number of batch
        :param blank_value : the value of `blank` label
        :param shuffle : whether shuffle dataset or not
        :param return_initial_state : Whether return Initial state(Zero state) or not
        :param state_size : if return_initial_state is True, the size of initial state
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = (self.dataset.max_word*4) + 1 #word * 4(초성/중성/종성/ᴥ)+1(<EOS> token)
        self.blank_value = blank_value
        self.shuffle = shuffle
        self.num_classes = len(JAMOS) - 2 # Drop <EOS> Token & <BLANK> Token
        self.return_initial_state = return_initial_state
        self.state_size = state_size
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, texts = self.dataset[self.batch_size * index:
                                     self.batch_size * (index + 1)]
        # label sequence
        labels = np.ones([self.batch_size, self.max_length], np.int32)
        labels *= -1  # BLANK Token value : -1
        for idx, text in enumerate(texts):
            jamos = decompose(text)
            labels[idx, :len(jamos)] = jamos2label(jamos)
            labels[idx, len(jamos)] = self.num_classes # <EOS> Token

        target_inputs = np.roll(labels, 1, axis=1)
        target_inputs[:, 0] = self.num_classes # <EOS> Token
        target_inputs[target_inputs==-1] = self.num_classes # <EOS> Token

        X = {
            "images" : images,
            "decoder_inputs" : target_inputs
        }
        # return initial state
        if self.return_initial_state:
            batch_size = images.shape[0]
            X['decoder_state'] = np.zeros([batch_size, self.state_size])

        Y = {
            "output_seqs" : labels
        }

        return X, Y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()

    @classmethod
    def convert2text(cls, arr: np.ndarray):
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        df = pd.DataFrame(arr)
        df = df.applymap(lambda x : JAMOS[x])
        texts = df.apply(lambda x: compose("".join(x)).replace("\n", ""), axis=1).values
        return texts


def jamos2label(jamos):
    return np.array([JAMO2IDX[char] for char in jamos])


class DataGenerator(Sequence):
    "Generates Text Recognition Dataset for Keras"

    def __init__(self, dataset, batch_size=32, blank_value=-1, shuffle=True):
        "Initialization"
        self.dataset = dataset
        self.batch_size = batch_size
        self.blank_value = blank_value
        self.shuffle = shuffle
        self.num_classes = self.dataset.labels.max() + 1  # With Blank
        self.max_length = self.dataset.digit_range[-1] - 1
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, labels, _ = self.dataset[self.batch_size * index:
                                         self.batch_size * (index + 1)]
        # Add Channel axis (batch, width, height) -> (batch, width, height, 1)
        batch_images = images[..., np.newaxis]

        # label sequence
        batch_labels = np.ones([self.batch_size, self.max_length+1], np.int32)
        batch_labels *= self.blank_value  # EOS Token value : -1
        for idx, label in enumerate(labels):
            batch_labels[idx, :len(label)] = label

        return batch_images, batch_labels

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()


class Seq2SeqGenerator(Sequence):
    "Generates Text Recognition Dataset for Keras"

    def __init__(self, dataset, batch_size=32, blank_value=-1, shuffle=True,
                 return_initial_state=True, state_size=512):
        "Initialization"
        self.dataset = dataset
        self.batch_size = batch_size
        self.blank_value = blank_value
        self.shuffle = shuffle
        self.return_initial_state = return_initial_state
        self.state_size = state_size

        self.num_classes = self.dataset.labels.max() + 1  # With Blank
        self.max_length = self.dataset.digit_range[-1] - 1
        self.on_epoch_end()

        self.idx2char = {i: str(i) for i in range(10)}
        self.idx2char[10] = '<EOS>'
        self.idx2char[-1] = ""


    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, labels, _ = self.dataset[self.batch_size * index:
                                         self.batch_size * (index + 1)]
        # Add Channel axis (batch, width, height) -> (batch, width, height, 1)
        batch_images = images[..., np.newaxis]

        # label sequence
        batch_labels = np.ones([self.batch_size, self.max_length+1], np.int32)
        batch_labels *= -1  # BLANK Token value : -1
        for idx, label in enumerate(labels):
            batch_labels[idx, :len(label)] = label
            batch_labels[idx, len(label)] = self.num_classes # <EOS> Token

        target_inputs = np.roll(batch_labels, 1, axis=1)
        target_inputs[:, 0] = self.num_classes # <EOS> Token
        target_inputs[target_inputs==-1] = self.num_classes # <EOS> Token

        X = {
            "images" : batch_images,
            "decoder_inputs" : target_inputs
        }
        # return initial state
        if self.return_initial_state:
            batch_size = batch_images.shape[0]
            X['decoder_state'] = np.zeros([batch_size, self.state_size])

        Y = {
            "output_seqs" : batch_labels
        }

        return X, Y

    def convert2text(self, arr):
        df = pd.DataFrame(arr)
        df = df.applymap(lambda x: self.idx2char[x])
        texts = df.apply(lambda x: "".join(x), axis=1).values
        return texts

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()