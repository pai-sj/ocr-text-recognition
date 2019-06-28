from tensorflow.python.keras.utils import Sequence
import numpy as np

가 = ord('가')
힣 = ord('힣')

KOR_CHARS = [chr(idx) for idx in range(가,힣+1)]
KOR2IDX = { char : idx for idx, char in enumerate(KOR_CHARS )}


class OCRGenerator(Sequence):
    "Generates OCR TEXT Recognition Dataset for Keras"

    def __init__(self, dataset, char_list=KOR_CHARS, batch_size=32, shuffle=True):
        """
        Initialization

        param
        :param dataset : instance of class 'OCRDataset'
        :param char_list : unique character list (for Embedding)
        :param batch_size : the number of batch
        :param shuffle : whether shuffle dataset or not
        """
        self.dataset = dataset
        self.char_list = char_list
        self.char2idx = {char: idx
                         for idx, char
                         in enumerate(self.char_list)}

        self.batch_size = batch_size
        self.max_length = self.dataset.max_word + 1 # With Blank for Last label
        self.shuffle = shuffle
        self.num_classes = len(self.char_list) + 1  # With Blank for CTC LOSS
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
        labels *= -1  # EOS Token value : -1
        for idx, text in enumerate(texts):
            labels[idx, :len(text)] = text2label(text, self.char2idx)

        return images, labels

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()


def text2label(text, char2idx=KOR2IDX):
    return np.array([char2idx[char] for char in text])


class DataGenerator(Sequence):
    "Generates Text Recognition Dataset for Keras"

    def __init__(self, dataset, batch_size=32, shuffle=True):
        "Initialization"
        self.dataset = dataset
        self.batch_size = batch_size
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
        batch_labels = np.ones([self.batch_size, self.max_length], np.int32)
        batch_labels *= -1  # EOS Token value : -1
        for idx, label in enumerate(labels):
            batch_labels[idx, :len(label)] = label

        return batch_images, batch_labels

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()