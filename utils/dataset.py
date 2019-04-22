import numpy as np
import pandas as pd
import os
"""
# All about MNIST Style DataSet

> We include the following dataset list
    - mnist : handwritten digits dataset

    - fashionmnist : dataset of Zalando's article images

    - handwritten : handwritten a ~ z Alphabet dataset

"""
dirnames = []
for dirname in os.getcwd().split('/'):
    dirnames.append(dirname)
    if dirname == 'CRNN-text-recognition':
        break
DATASET_DIR = "/".join(dirnames+['datasets'])

DOWNLOAD_URL_FORMAT = "https://s3.ap-northeast-2.amazonaws.com/pai-datasets/all-about-mnist/{}/{}.csv"


class SerializationDataset:
    """
    generate data for Serialization

    이 class는 단순히 숫자를 나열하는 것

    :param dataset : Select one, (mnist, fashionmnist, handwritten)
    :param data_type : Select one, (train, test, validation)
    :param digit : the length of number (몇개의 숫자를 serialize할 것인지 결정)
    :param bg_noise : the background noise of image, bg_noise = (gaussian mean, gaussian stddev)
    :param pad_range : the padding length between two number (두 숫자 간 거리, 값의 범위로 주어 랜덤하게 결정)
    """

    def __init__(self, dataset="mnist", data_type="train",
                 digit=5, bg_noise=(0, 0.2), pad_range=(3, 10)):
        """
        generate data for Serialization

        :param dataset: Select one, (mnist, fashionmnist, handwritten)
        :param data_type: Select one, (train, test, validation)
        :param digit : the length of number
          if digit is integer, the length of number is always same value.
          if digit is tuple(low_value, high_value), the length of number will be determined within the range
        :param bg_noise : the background noise of image, bg_noise = (gaussian mean, gaussian stddev)
        :param pad_range : the padding length between two number (두 숫자 간 거리, 값의 범위로 주어 랜덤하게 결정)
        """
        self.images, self.labels = load_dataset(dataset, data_type)
        if isinstance(digit, int):
            self.digit_range = (digit, digit+1)
        else:
            self.digit_range = digit
        self.num_data = len(self.labels) // (self.digit_range[1]-1)
        self.index_list = np.arange(len(self.labels))

        self.bg_noise = bg_noise
        self.pad_range = pad_range

        self.max_length = int((15 + pad_range[1]) * self.digit_range[1])

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if isinstance(index, int):
            num_digit = np.random.randint(*self.digit_range)
            start_index = (self.digit_range[1]-1) * index
            digits = self.index_list[start_index :start_index + num_digit]

            digit_images = self.images[digits]
            digit_labels = self.labels[digits].values
            series_image, series_len = self._serialize_random(digit_images)

            return series_image, digit_labels, series_len

        else:
            batch_images, batch_labels, batch_length = [], [], []
            indexes = np.arange(self.num_data)[index]
            for _index in indexes:
                num_digit = np.random.randint(*self.digit_range)
                start_index = (self.digit_range[1] - 1) * _index
                digits = self.index_list[start_index :start_index + num_digit]

                digit_images = self.images[digits]
                digit_labels = self.labels[digits].values
                series_image, series_len = self._serialize_random(digit_images)
                batch_images.append(series_image)
                batch_labels.append(digit_labels)
                batch_length.append(series_len)

            return np.stack(batch_images), \
                batch_labels, \
                np.stack(batch_length)

    def shuffle(self):
        indexes = np.arange(len(self.images))
        np.random.shuffle(indexes)

        self.images = self.images[indexes]
        self.labels = self.labels[indexes]
        self.labels.index = np.arange(len(self.labels))

    def _serialize_random(self, images):
        """
        복수의 이미지를 직렬로 붙임

        :param images:
        :return:
        """
        pad_height = images.shape[1]
        pad_width = np.random.randint(*self.pad_range)

        serialized_image = np.zeros([pad_height, pad_width])
        for image in images:
            serialized_image = self._place_random(image, serialized_image)

        full_image = np.random.normal(*self.bg_noise,
                                      size=(pad_height, self.max_length))

        if serialized_image.shape[1] < self.max_length:
            series_length = serialized_image.shape[1]
            full_image[:, :serialized_image.shape[1]] += serialized_image
        else:
            series_length = full_image.shape[1]
            full_image += serialized_image[:, :full_image.shape[1]]

        full_image = np.clip(full_image, 0., 1.)
        return full_image, series_length

    def _place_random(self, image, serialized_image):
        """
        가운데 정렬된 이미지를 떼어서 재정렬함

        :param image:
        :param serialized_image:
        :return:
        """
        x_min, x_max, _, _ = crop_fit_position(image)
        cropped = image[:, x_min:x_max]

        pad_height = cropped.shape[0]
        pad_width = np.random.randint(*self.pad_range)
        pad = np.zeros([pad_height, pad_width])

        serialized_image = np.concatenate(
            [serialized_image, cropped, pad], axis=1)
        return serialized_image


def crop_fit_position(image):
    """
    get the coordinates to fit object in image

    :param image:
    :return:
    """
    positions = np.argwhere(
        image >= 0.1)  # set the threshold to 0.1 for reducing the noise

    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)

    return np.array([x_min, x_max, y_min, y_max])


def load_dataset(dataset, data_type):
    """
    Load the MNIST-Style dataset
    if you don't have dataset, download the file automatically

    :param dataset: Select one, (mnist, fashionmnist, handwritten)
    :param data_type: Select one, (train, test, validation)
    :return:
    """
    if dataset not in ["mnist", "fashionmnist", "handwritten"]:
        raise ValueError(
            "allowed dataset: mnist, fashionmnist, handwritten")
    if data_type not in ["train", "test", "validation"]:
        raise ValueError(
            "allowed data_type: train, test, validation")

    file_path = os.path.join(
        DATASET_DIR, "{}/{}.csv".format(dataset, data_type))

    if not os.path.exists(file_path):
        import wget
        os.makedirs(os.path.split(file_path)[0], exist_ok=True)
        url = DOWNLOAD_URL_FORMAT.format(dataset, data_type)
        wget.download(url, out=file_path)

    df = pd.read_csv(file_path)

    images = df.values[:, 1:].reshape(-1, 28, 28)
    images = images / 255  # normalization, 0~1
    labels = df.label  # label information
    return images, labels


if __name__ == '__main__':
    pass
