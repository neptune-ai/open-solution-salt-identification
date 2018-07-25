import logging
import os
import pathlib
import random
import sys
import time
from itertools import chain
from collections import Iterable

from deepsense import neptune
import numpy as np
import pandas as pd
import torch
from PIL import Image
from attrdict import AttrDict
from tqdm import tqdm
from pycocotools import mask as cocomask
from sklearn.model_selection import BaseCrossValidator
from steppy.base import BaseTransformer
import yaml
from imgaug import augmenters as iaa
import imgaug as ia

NEPTUNE_CONFIG_PATH = str(pathlib.Path(__file__).resolve().parents[1] / 'configs' / 'neptune_local.yaml')


# Alex Martelli's 'Borg'
# http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class _Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class NeptuneContext(_Borg):
    def __init__(self, fallback_file=NEPTUNE_CONFIG_PATH):
        _Borg.__init__(self)

        self.ctx = neptune.Context()
        self.fallback_file = fallback_file
        self.params = self._read_params()
        self.numeric_channel = neptune.ChannelType.NUMERIC
        self.image_channel = neptune.ChannelType.IMAGE
        self.text_channel = neptune.ChannelType.TEXT

    def channel_send(self, *args, **kwargs):
        self.ctx.channel_send(*args, **kwargs)

    def _read_params(self):
        if self.ctx.params.__class__.__name__ == 'OfflineContextParams':
            params = self._read_yaml().parameters
        else:
            params = self.ctx.params
        return params

    def _read_yaml(self):
        with open(self.fallback_file) as f:
            config = yaml.load(f)
        return AttrDict(config)


def init_logger():
    logger = logging.getLogger('salt-detection')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def get_logger():
    return logging.getLogger('salt-detection')


def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)

    if not masks:
        return [labeled]
    else:
        return masks


def create_submission(experiments_dir, meta, predictions, logger):
    image_ids, encodings = [], []
    output = []
    for image_id, prediction in zip(meta['id'].values, predictions):
        for mask in decompose(prediction):
            rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask > 128.))
            if len(rle_encoded) != 0:
                image_ids.append(image_id)
                encodings.append(rle_encoded)
                output.append([image_id, rle_encoded])
            else:
                logger.info('*** image_id {}'.format(image_id))
                logger.info('*** rle_encoded {} is empty'.format(rle_encoded))

    submission = pd.DataFrame(output, columns=['id', 'rle_mask']).astype(str)
    submission = submission[submission['rle_mask'] != 'nan']
    submission_filepath = os.path.join(experiments_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission.head()))


def read_masks(masks_filepaths):
    masks = []
    for mask_filepath in tqdm(masks_filepaths):
        mask = Image.open(mask_filepath)
        mask = np.asarray(mask.convert('L').point(lambda x: 0 if x < 128 else 1)).astype(np.uint8)
        masks.append(mask)
    return masks


def read_masks_from_csv(image_ids, solution_file_path):
    solution = pd.read_csv(solution_file_path)
    masks = []
    for image_id in image_ids:
        mask_shape = (solution[solution['ImageId'] == image_id]['Height'].iloc[0],
                      solution[solution['ImageId'] == image_id]['Width'].iloc[0])
        mask = np.zeros(mask_shape, dtype=np.uint8)
        for i, rle in enumerate(solution[solution['ImageId'] == image_id]['EncodedPixels']):
            mask += (i + 1) * run_length_decoding(rle, mask_shape)
        masks.append(mask)
    return masks


def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    if len(rle) != 0 and rle[-1] + rle[-2] == x.size:
        rle[-2] = rle[-2] - 1

    return rle


def run_length_decoding(mask_rle, shape):
    """
    Based on https://www.kaggle.com/msl23518/visualize-the-stage1-test-solution and modified
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T


def generate_metadata(train_images_dir, test_images_dir, depths_filepath):
    depths = pd.read_csv(depths_filepath)

    metadata = {}
    for filename in tqdm(os.listdir(os.path.join(train_images_dir, 'images'))):
        image_filepath = os.path.join(train_images_dir, 'images', filename)
        mask_filepath = os.path.join(train_images_dir, 'masks', filename)
        image_id = filename.split('.')[0]
        depth = depths[depths['id'] == image_id]['z'].values[0]

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(mask_filepath)
        metadata.setdefault('is_train', []).append(1)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('z', []).append(depth)

    for filename in tqdm(os.listdir(os.path.join(test_images_dir, 'images'))):
        image_filepath = os.path.join(test_images_dir, 'images', filename)
        image_id = filename.split('.')[0]
        depth = depths[depths['id'] == image_id]['z'].values[0]

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(None)
        metadata.setdefault('is_train', []).append(0)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('z', []).append(depth)

    return pd.DataFrame(metadata)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(X, theta=1.0, axis=None):
    """
    https://nolanbconaway.github.io/blog/2017/softmax-numpy
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def from_pil(*images):
    images = [np.array(image) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images


def to_pil(*images):
    images = [Image.fromarray((image).astype(np.uint8)) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images


def make_apply_transformer(func, output_name='output', apply_on=None):
    class StaticApplyTransformer(BaseTransformer):
        def transform(self, *args, **kwargs):
            self.check_input(*args, **kwargs)

            if not apply_on:
                iterator = zip(*args, *kwargs.values())
            else:
                iterator = zip(*args, *[kwargs[key] for key in apply_on])

            output = []
            for func_args in tqdm(iterator, total=self.get_arg_length(*args, **kwargs)):
                output.append(func(*func_args))
            return {output_name: output}

        @staticmethod
        def check_input(*args, **kwargs):
            if len(args) and len(kwargs) == 0:
                raise Exception('Input must not be empty')

            arg_length = None
            for arg in chain(args, kwargs.values()):
                if not isinstance(arg, Iterable):
                    raise Exception('All inputs must be iterable')
                arg_length_loc = None
                try:
                    arg_length_loc = len(arg)
                except:
                    pass
                if arg_length_loc is not None:
                    if arg_length is None:
                        arg_length = arg_length_loc
                    elif arg_length_loc != arg_length:
                        raise Exception('All inputs must be the same length')

        @staticmethod
        def get_arg_length(*args, **kwargs):
            arg_length = None
            for arg in chain(args, kwargs.values()):
                if arg_length is None:
                    try:
                        arg_length = len(arg)
                    except:
                        pass
                if arg_length is not None:
                    return arg_length

    return StaticApplyTransformer()


def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)


def binary_from_rle(rle):
    return cocomask.decode(rle)


def get_segmentations(labeled):
    nr_true = labeled.max()
    segmentations = []
    for i in range(1, nr_true + 1):
        msk = labeled == i
        segmentation = rle_from_binary(msk.astype('uint8'))
        segmentation['counts'] = segmentation['counts'].decode("UTF-8")
        segmentations.append(segmentation)
    return segmentations


def get_crop_pad_sequence(vertical, horizontal):
    top = int(vertical / 2)
    bottom = vertical - top
    right = int(horizontal / 2)
    left = horizontal - right
    return (top, right, bottom, left)


def get_list_of_image_predictions(batch_predictions):
    image_predictions = []
    for batch_pred in batch_predictions:
        image_predictions.extend(list(batch_pred))
    return image_predictions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImgAug:
    def __init__(self, augmenters):
        if not isinstance(augmenters, list):
            augmenters = [augmenters]
        self.augmenters = augmenters
        self.seq_det = None

    def _pre_call_hook(self):
        seq = iaa.Sequential(self.augmenters)
        seq = reseed(seq, deterministic=True)
        self.seq_det = seq

    def transform(self, *images):
        images = [self.seq_det.augment_image(image) for image in images]
        if len(images) == 1:
            return images[0]
        else:
            return images

    def __call__(self, *args):
        self._pre_call_hook()
        return self.transform(*args)


def get_seed():
    seed = int(time.time()) + int(os.getpid())
    return seed


def reseed(augmenter, deterministic=True):
    augmenter.random_state = ia.new_random_state(get_seed())
    if deterministic:
        augmenter.deterministic = True

    for lists in augmenter.get_children_lists():
        for aug in lists:
            aug = reseed(aug, deterministic=True)
    return augmenter


class KFoldBySortedValue(BaseCrossValidator):
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        sorted_idx_vals = sorted(zip(indices, X), key=lambda x: x[1])
        indices = [idx for idx, val in sorted_idx_vals]

        for split_start in range(self.n_splits):
            split_indeces = indices[split_start::self.n_splits]
            yield split_indeces

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_split
