import os
import shutil

import pandas as pd

from .metrics import intersection_over_union, intersection_over_union_thresholds
from .pipeline_config import SOLUTION_CONFIG, SEED, DEPTH_COLUMN, Y_COLUMNS
from .pipelines import PIPELINES
from .utils import NeptuneContext, init_logger, read_masks, read_masks_from_csv, create_submission, \
    generate_metadata, set_seed, KFoldBySortedValue

LOGGER = init_logger()
CTX = NeptuneContext()
PARAMS = CTX.params
set_seed(SEED)


class PipelineManager():
    def prepare_metadata(self):
        prepare_metadata()

    def train(self, pipeline_name, dev_mode):
        train(pipeline_name, dev_mode)

    def evaluate(self, pipeline_name, dev_mode):
        evaluate(pipeline_name, dev_mode)

    def predict(self, pipeline_name, dev_mode):
        predict(pipeline_name, dev_mode)


def prepare_metadata():
    LOGGER.info('creating metadata')
    meta = generate_metadata(train_images_dir=PARAMS.train_images_dir,
                             test_images_dir=PARAMS.test_images_dir,
                             depths_filepath=PARAMS.depths_filepath
                             )
    meta.to_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'), index=None)


def train(pipeline_name, dev_mode):
    LOGGER.info('training')
    if bool(PARAMS.overwrite) and os.path.isdir(PARAMS.experiment_dir):
        shutil.rmtree(PARAMS.experiment_dir)

    meta = pd.read_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'))
    meta_train = meta[meta['is_train'] == 1]

    cv = KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)
    for train_idx, valid_idx in cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1)):
        break

    meta_train_split, meta_valid_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]

    if dev_mode:
        meta_train_split = meta_train_split.sample(PARAMS.dev_mode_size, random_state=SEED)
        meta_valid_split = meta_valid_split.sample(int(PARAMS.dev_mode_size / 2), random_state=SEED)

    data = {'input': {'meta': meta_train_split
                      },
            'callback_input': {'meta_valid': meta_valid_split
                               }
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


def evaluate(pipeline_name, dev_mode):
    LOGGER.info('evaluating')
    meta = pd.read_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'))
    meta_train = meta[meta['is_train'] == 1]

    cv = KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)
    for train_idx, valid_idx in cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1)):
        break

    meta_valid_split = meta_train.iloc[valid_idx]
    y_true = read_masks(meta_valid_split[Y_COLUMNS[0]].values)

    if dev_mode:
        meta_valid_split = meta_valid_split.sample(PARAMS.dev_mode_size, random_state=SEED)

    data = {'input': {'meta': meta_valid_split,
                      },
            'callback_input': {'meta_valid': None
                               }
            }
    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    LOGGER.info('Calculating IOU and IOUT Scores')
    iou_score = intersection_over_union(y_true, y_pred)
    LOGGER.info('IOU score on validation is {}'.format(iou_score))
    CTX.channel_send('IOU', 0, iou_score)

    iout_score = intersection_over_union_thresholds(y_true, y_pred)
    LOGGER.info('IOUT score on validation is {}'.format(iout_score))
    CTX.channel_send('IOUT', 0, iout_score)


def predict(pipeline_name, dev_mode):
    LOGGER.info('predicting')
    meta = pd.read_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'))
    meta_test = meta[meta['is_train'] == 0]

    if dev_mode:
        meta_test = meta_test.sample(PARAMS.dev_mode_size, random_state=SEED)

    data = {'input': {'meta': meta_test,
                      },
            'callback_input': {'meta_valid': None
                               }
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    create_submission(PARAMS.experiment_dir, meta_test, y_pred, LOGGER)
