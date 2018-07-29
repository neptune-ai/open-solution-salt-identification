import os
import shutil

import numpy as np
import pandas as pd

from .metrics import intersection_over_union, intersection_over_union_thresholds
from . import pipeline_config as cfg
from .pipelines import PIPELINES
from .utils import NeptuneContext, init_logger, read_masks, read_masks_from_csv, create_submission, \
    generate_metadata, set_seed, KFoldBySortedValue

LOGGER = init_logger()
CTX = NeptuneContext()
PARAMS = CTX.params
set_seed(cfg.SEED)


class PipelineManager:
    def prepare_metadata(self):
        prepare_metadata()

    def train(self, pipeline_name, dev_mode):
        train(pipeline_name, dev_mode)

    def evaluate(self, pipeline_name, dev_mode):
        evaluate(pipeline_name, dev_mode)

    def predict(self, pipeline_name, dev_mode, submit_predictions):
        predict(pipeline_name, dev_mode, submit_predictions)

    def train_evaluate_cv(self, pipeline_name, dev_mode):
        train_evaluate_cv(pipeline_name, dev_mode)


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

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=True)
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
    y_true = read_masks(meta_valid_split[cfg.Y_COLUMNS[0]].values)

    if dev_mode:
        meta_valid_split = meta_valid_split.sample(PARAMS.dev_mode_size, random_state=SEED)

    data = {'input': {'meta': meta_valid_split,
                      },
            'callback_input': {'meta_valid': None
                               }
            }
    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False)
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


def make_submission(submission_filepath):
    LOGGER.info('Making Kaggle submit...')
    os.system('kaggle competitions submit -c tgs-salt-identification-challenge -f {} -m {}'.format(submission_filepath,
                                                                                                   PARAMS.kaggle_message))
    LOGGER.info('Kaggle submit completed')


def predict(pipeline_name, submit_predictions, dev_mode):
    LOGGER.info('predicting')
    meta = pd.read_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'))
    meta_test = meta[meta['is_train'] == 0]

    if dev_mode:
        meta_test = meta_test.sample(PARAMS.dev_mode_size, random_state=cfg.SEED)

    data = {'input': {'meta': meta_test,
                      },
            'callback_input': {'meta_valid': None
                               }
            }

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    submission = create_submission(meta_test, y_pred)

    submission_filepath = os.path.join(PARAMS.experiment_dir, 'submission.csv')

    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    LOGGER.info('submission saved to {}'.format(submission_filepath))
    LOGGER.info('submission head \n\n{}'.format(submission.head()))

    if submit_predictions:
        make_submission(submission_filepath)


def train_evaluate_cv(pipeline_name, dev_mode):
    LOGGER.info('training')
    if bool(PARAMS.overwrite) and os.path.isdir(PARAMS.experiment_dir):
        shutil.rmtree(PARAMS.experiment_dir)

    if dev_mode:
        meta = pd.read_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'), nrows=PARAMS.dev_mode_size)
    else:
        meta = pd.read_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'))
    meta_train = meta[meta['is_train'] == 1]

    cv = KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=cfg.SEED)

    fold_iou, fold_iout = [], []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[cfg.DEPTH_COLUMN].values.reshape(-1))):
        train_data_split, valid_data_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]

        LOGGER.info('Started fold {}'.format(fold_id))
        iou, iout, _, _ = _fold_fit_evaluate_loop(train_data_split,
                                                  valid_data_split,
                                                  fold_id,
                                                  pipeline_name)

        LOGGER.info('Fold {} IOU {}'.format(fold_id, iou))
        CTX.channel_send('Fold {} IOU'.format(fold_id), 0, iou)
        LOGGER.info('Fold {} IOUT {}'.format(fold_id, iout))
        CTX.channel_send('Fold {} IOUT'.format(fold_id), 0, iout)

        fold_iou.append(iou)
        fold_iout.append(iout)

    iou_mean, iou_std = np.mean(fold_iou), np.std(fold_iou)
    iout_mean, iout_std = np.mean(fold_iout), np.std(fold_iout)

    LOGGER.info('IOU mean {}, IOU std {}'.format(iou_mean, iou_std))
    CTX.channel_send('IOU', 0, iou_mean)
    CTX.channel_send('IOU STD', 0, iou_std)

    LOGGER.info('IOUT mean {}, IOUT std {}'.format(iout_mean, iout_std))
    CTX.channel_send('IOUT', 0, iout_mean)
    CTX.channel_send('IOUT STD', 0, iout_std)


def _fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id, pipeline_name):
    train_pipe_input = {'input': {'meta': train_data_split
                                  },
                        'callback_input': {'meta_valid': valid_data_split
                                           }
                        }

    valid_pipe_input = {'input': {'meta': valid_data_split
                                  },
                        'callback_input': {'meta_valid': None
                                           }
                        }

    config = cfg.SOLUTION_CONFIG
    model_name = pipeline_name
    config['model'][model_name]['callbacks_config']['neptune_monitor']['model_name'] = '{}_{}'.format(model_name,
                                                                                                      fold_id)
    pipeline = PIPELINES[pipeline_name](config=config, train_mode=True, suffix='_fold_{}'.format(fold_id))
    LOGGER.info('Start pipeline fit and transform on train')
    pipeline.clean_cache()
    pipeline.fit_transform(train_pipe_input)
    pipeline.clean_cache()

    pipeline = PIPELINES[pipeline_name](config=config, train_mode=False, suffix='_fold_{}'.format(fold_id))
    LOGGER.info('Start pipeline transform on valid')
    pipeline.clean_cache()
    output_valid = pipeline.transform(valid_pipe_input)
    pipeline.clean_cache()

    y_pred = output_valid['y_pred']
    y_true = read_masks(valid_data_split[cfg.Y_COLUMNS[0]].values)

    iou = intersection_over_union(y_true, y_pred)
    iout = intersection_over_union_thresholds(y_true, y_pred)

    return iou, iout, y_pred, pipeline
