import logging
import os
import random
import sys
import time
from itertools import chain
from collections import Iterable
import gc

import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from attrdict import AttrDict
from tqdm import tqdm
from pycocotools import mask as cocomask
from sklearn.model_selection import BaseCrossValidator
from sklearn.externals import joblib
from steppy.base import BaseTransformer, Step
from steppy.utils import get_logger
from skimage.transform import resize
import yaml
from imgaug import augmenters as iaa
import imgaug as ia
import torch

logger = get_logger()


def read_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f)
    return AttrDict(config)


def check_env_vars():
    assert os.getenv('NEPTUNE_API_TOKEN'), """You must put your Neptune API token in the \
NEPTUNE_API_TOKEN env variable. You should run:
    $ export NEPTUNE_API_TOKEN=your_neptune_api_token"""
    assert os.getenv('CONFIG_PATH'), """You must specify path to the config file in \
CONFIG_PATH env variable. For example run:
    $ export CONFIG_PATH=neptune.yaml"""


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


def create_submission(meta, predictions):
    output = []
    for image_id, mask in zip(meta['id'].values, predictions):
        rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask))
        output.append([image_id, rle_encoded])

    submission = pd.DataFrame(output, columns=['id', 'rle_mask']).astype(str)
    return submission


def encode_rle(predictions):
    return [run_length_encoding(mask) for mask in predictions]


def read_masks(masks_filepaths):
    masks = []
    for mask_filepath in tqdm(masks_filepaths):
        mask = Image.open(mask_filepath)
        mask = np.asarray(mask.convert('L').point(lambda x: 0 if x < 128 else 1)).astype(np.uint8)
        masks.append(mask)
    return masks


def read_images(filepaths):
    images = []
    for filepath in filepaths:
        image = np.array(Image.open(filepath))
        images.append(image)
    return images


def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1):
            rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

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
        size = (np.array(Image.open(mask_filepath)) > 0).astype(np.uint8).sum()
        is_not_empty = int(size != 0)

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(mask_filepath)
        metadata.setdefault('is_train', []).append(1)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('z', []).append(depth)
        metadata.setdefault('size', []).append(size)
        metadata.setdefault('is_not_empty', []).append(is_not_empty)

    for filename in tqdm(os.listdir(os.path.join(test_images_dir, 'images'))):
        image_filepath = os.path.join(test_images_dir, 'images', filename)
        image_id = filename.split('.')[0]
        depth = depths[depths['id'] == image_id]['z'].values[0]
        size = np.nan
        is_not_empty = np.nan

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(None)
        metadata.setdefault('is_train', []).append(0)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('z', []).append(depth)
        metadata.setdefault('size', []).append(size)
        metadata.setdefault('is_not_empty', []).append(is_not_empty)

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
        return self.n_splits


def plot_list(images=[], labels=[], vmin=0.0, vmax=1.0):
    n_img = len(images)
    n_lab = len(labels)
    n = n_lab + n_img
    fig, axs = plt.subplots(1, n, figsize=(16, 12))
    for i, image in enumerate(images):
        axs[i].imshow(image, vmin=vmin, vmax=vmax)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    for j, label in enumerate(labels):
        axs[n_img + j].imshow(label, cmap='nipy_spectral')
        axs[n_img + j].set_xticks([])
        axs[n_img + j].set_yticks([])
    plt.show()


def clean_object_from_memory(obj):
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class FineTuneStep(Step):
    def __init__(self,
                 name,
                 transformer,
                 experiment_directory,
                 input_data=None,
                 input_steps=None,
                 adapter=None,
                 is_trainable=False,
                 cache_output=False,
                 persist_output=False,
                 load_persisted_output=False,
                 force_fitting=False,
                 fine_tuning=False,
                 persist_upstream_pipeline_structure=False):
        super().__init__(name,
                         transformer,
                         experiment_directory,
                         input_data=input_data,
                         input_steps=input_steps,
                         adapter=adapter,
                         is_trainable=is_trainable,
                         cache_output=cache_output,
                         persist_output=persist_output,
                         load_persisted_output=load_persisted_output,
                         force_fitting=force_fitting,
                         persist_upstream_pipeline_structure=persist_upstream_pipeline_structure)
        self.fine_tuning = fine_tuning

    def _cached_fit_transform(self, step_inputs):
        if self.is_trainable:
            if self.transformer_is_cached:
                if self.force_fitting and self.fine_tuning:
                    raise ValueError('only one of force_fitting or fine_tuning can be True')
                elif self.force_fitting:
                    logger.info('Step {}, fitting and transforming...'.format(self.name))
                    step_output_data = self.transformer.fit_transform(**step_inputs)
                    logger.info('Step {}, persisting transformer to the {}'
                                .format(self.name, self.exp_dir_transformers_step))
                    self.transformer.persist(self.exp_dir_transformers_step)
                elif self.fine_tuning:
                    logger.info('Step {}, loading transformer from the {}'
                                .format(self.name, self.exp_dir_transformers_step))
                    self.transformer.load(self.exp_dir_transformers_step)
                    logger.info('Step {}, transforming...'.format(self.name))
                    step_output_data = self.transformer.fit_transform(**step_inputs)
                    self.transformer.persist(self.exp_dir_transformers_step)
                else:
                    logger.info('Step {}, loading transformer from the {}'
                                .format(self.name, self.exp_dir_transformers_step))
                    self.transformer.load(self.exp_dir_transformers_step)
                    logger.info('Step {}, transforming...'.format(self.name))
                    step_output_data = self.transformer.transform(**step_inputs)
            else:
                logger.info('Step {}, fitting and transforming...'.format(self.name))
                step_output_data = self.transformer.fit_transform(**step_inputs)
                logger.info('Step {}, persisting transformer to the {}'
                            .format(self.name, self.exp_dir_transformers_step))
                self.transformer.persist(self.exp_dir_transformers_step)
        else:
            logger.info('Step {}, transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)

        if self.cache_output:
            logger.info('Step {}, caching output to the {}'
                        .format(self.name, self.exp_dir_cache_step))
            self._persist_output(step_output_data, self.exp_dir_cache_step)
        if self.persist_output:
            logger.info('Step {}, persisting output to the {}'
                        .format(self.name, self.exp_dir_outputs_step))
            self._persist_output(step_output_data, self.exp_dir_outputs_step)
        return step_output_data


def pytorch_where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1 - cond) * x_2)


class AddDepthChannels:
    def __call__(self, tensor):
        _, h, w = tensor.size()
        for row, const in enumerate(np.linspace(0, 1, h)):
            tensor[1, row, :] = const
        tensor[2] = tensor[0] * tensor[1]
        return tensor

    def __repr__(self):
        return self.__class__.__name__


def load_image(filepath, is_mask=False):
    if is_mask:
        img = (np.array(Image.open(filepath)) > 0).astype(np.uint8)
    else:
        img = np.array(Image.open(filepath)).astype(np.uint8)
    return img


def save_image(img, filepath):
    img = Image.fromarray((img))
    img.save(filepath)


def resize_image(image, target_shape, is_mask=False):
    if is_mask:
        image = (resize(image, target_shape, preserve_range=True) > 0).astype(int)
    else:
        image = resize(image, target_shape)
    return image


def get_cut_coordinates(mask, step=4, min_img_crop=20, min_size=50, max_size=300):
    h, w = mask.shape
    ts = []
    rots = [1, 2, 3, 0]
    for rot in rots:
        mask = np.rot90(mask)
        for t in range(min_img_crop, h, step):
            crop = mask[:t, :t]
            size = crop.mean() * h * w
            if min_size < size <= max_size:
                break
        ts.append((t, rot))
    try:
        ts = [(t, r) for t, r in ts if t < 99]
        best_t, best_rot = sorted(ts, key=lambda x: x[0], reverse=True)[0]
    except IndexError:
        return (0, w), (0, h), False
    if best_t < min_img_crop:
        return (0, w), (0, h), False

    if best_rot == 0:
        x1, x2, y1, y2 = 0, best_t, 0, best_t
    elif best_rot == 1:
        x1, x2, y1, y2 = 0, best_t, h - best_t, h
    elif best_rot == 2:
        x1, x2, y1, y2 = w - best_t, w, h - best_t, h
    elif best_rot == 3:
        x1, x2, y1, y2 = w - best_t, w, 0, best_t
    else:
        raise ValueError
    return (x1, x2), (y1, y2), True


def group_predictions_by_id(raw_dir, grouped_by_id_dir):
    experiments = sorted(os.listdir(raw_dir))
    for experiment in tqdm(experiments):
        for train_test in ['train', 'test']:
            oof_predictions = joblib.load(os.path.join(raw_dir,
                                                       experiment,
                                                       'out_of_fold_{}_predictions.pkl'.format(train_test)))
            ids, images = oof_predictions['ids'], oof_predictions['images']
            images = [image[1, :, :] for image in images]
            for idx, image in zip(ids, images):
                os.makedirs(os.path.join(grouped_by_id_dir, idx), exist_ok=True)
                joblib.dump(image, os.path.join(grouped_by_id_dir, idx, '{}.pkl'.format(experiment)))


def join_id_predictions(grouped_by_id_dir, joined_predictions_dir):
    for idx in tqdm(os.listdir(grouped_by_id_dir)):
        predictions = []
        for prediction_filepath in glob.glob('{}/{}/*'.format(grouped_by_id_dir, idx)):
            prediction = joblib.load(prediction_filepath)
            predictions.append(prediction)
        predictions = np.stack(predictions, axis=-1)
        joblib.dump(predictions, os.path.join(joined_predictions_dir, '{}.pkl'.format(idx)))


def generate_metadata_stacking(metadata_filepath, joined_predictions_dir, colname='file_path_stacked_predictions'):
    meta = pd.read_csv(metadata_filepath)
    meta[colname] = meta['id'].apply(lambda x: os.path.join(joined_predictions_dir, '{}.pkl'.format(x)))
    return meta
