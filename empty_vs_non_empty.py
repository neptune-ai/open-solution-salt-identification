from functools import partial
import os
import shutil

from attrdict import AttrDict
import neptune
from neptunecontrib.api.utils import get_filepaths
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from steppy.base import Step, IdentityOperation
from steppy.adapter import Adapter, E

from common_blocks import augmentation as aug
from common_blocks import models
from common_blocks import loaders
from common_blocks import utils
from common_blocks import postprocessing

utils.check_env_vars()
CONFIG = utils.read_config(config_path=os.getenv('CONFIG_PATH'))
LOGGER = utils.init_logger()

neptune.init(project_qualified_name=CONFIG.project)

#    ______   ______   .__   __.  _______  __    _______      _______.
#   /      | /  __  \  |  \ |  | |   ____||  |  /  _____|    /       |
#  |  ,----'|  |  |  | |   \|  | |  |__   |  | |  |  __     |   (----`
#  |  |     |  |  |  | |  . `  | |   __|  |  | |  | |_ |     \   \
#  |  `----.|  `--'  | |  |\   | |  |     |  | |  |__| | .----)   |
#   \______| \______/  |__| \__| |__|     |__|  \______| |_______/
#

EXPERIMENT_NAME = 'empty_vs_non_empty'
EXPERIMENT_DIR = 'data/experiments/{}'.format(EXPERIMENT_NAME)
CLONE_EXPERIMENT_DIR_FROM = ''  # When running eval in the cloud specify this as for example /input/SAL-14/output/experiment
OVERWRITE_EXPERIMENT_DIR = False
DEV_MODE = False
SECOND_LEVEL = False
USE_DEPTH = False
USE_AUXILIARY_DATA = False
TAGS = ['first-level', 'training', 'empty_vs_non_empty']

if OVERWRITE_EXPERIMENT_DIR and os.path.isdir(EXPERIMENT_DIR):
    shutil.rmtree(EXPERIMENT_DIR)
if CLONE_EXPERIMENT_DIR_FROM != '':
    if os.path.exists(EXPERIMENT_DIR):
        shutil.rmtree(EXPERIMENT_DIR)
    shutil.copytree(CLONE_EXPERIMENT_DIR_FROM, EXPERIMENT_DIR)

PARAMS = CONFIG.parameters

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 1234
ID_COLUMN = 'id'
DEPTH_COLUMN = 'z'
X_COLUMN = 'file_path_image'
Y_COLUMN = 'is_not_empty'

if USE_DEPTH:
    x_columns = [X_COLUMN, DEPTH_COLUMN]
else:
    x_columns = [X_COLUMN]

CONFIG = AttrDict({
    'execution': {'experiment_dir': EXPERIMENT_DIR,
                  'num_workers': PARAMS.num_workers,
                  },
    'general': {'img_H-W': (PARAMS.image_h, PARAMS.image_w),
                'loader_mode': PARAMS.loader_mode,
                'num_classes': 2,
                'original_size': (101, 101),
                },
    'xy_splitter': {
        'network': {'x_columns': x_columns,
                    'y_columns': [Y_COLUMN],
                    },
    },
    'reader': {
        'network': {'x_columns': x_columns,
                    'y_columns': [Y_COLUMN],
                    },
    },
    'loaders': {'stacking': {'dataset_params': {'h': PARAMS.image_h,
                                                'w': PARAMS.image_w,
                                                'image_source': PARAMS.image_source,
                                                'target_format': PARAMS.target_format,
                                                'use_depth': USE_DEPTH,
                                                'MEAN': MEAN,
                                                'STD': STD
                                                },
                             'loader_params': {'training': {'batch_size': PARAMS.batch_size_train,
                                                            'shuffle': True,
                                                            'num_workers': PARAMS.num_workers,
                                                            'pin_memory': PARAMS.pin_memory
                                                            },
                                               'inference': {'batch_size': PARAMS.batch_size_inference,
                                                             'shuffle': False,
                                                             'num_workers': PARAMS.num_workers,
                                                             'pin_memory': PARAMS.pin_memory
                                                             },
                                               },
                             'augmentation_params': {},
                             },
                'resize_and_pad': {'dataset_params': {'h': PARAMS.image_h,
                                                      'w': PARAMS.image_w,
                                                      'image_source': PARAMS.image_source,
                                                      'use_depth': USE_DEPTH,
                                                      'MEAN': MEAN,
                                                      'STD': STD
                                                      },
                                   'loader_params': {'training': {'batch_size': PARAMS.batch_size_train,
                                                                  'shuffle': True,
                                                                  'num_workers': PARAMS.num_workers,
                                                                  'pin_memory': PARAMS.pin_memory
                                                                  },
                                                     'inference': {'batch_size': PARAMS.batch_size_inference,
                                                                   'shuffle': False,
                                                                   'num_workers': PARAMS.num_workers,
                                                                   'pin_memory': PARAMS.pin_memory
                                                                   },
                                                     },

                                   'augmentation_params': {'image_augment_train': aug.intensity_seq,
                                                           'image_augment_with_target_train': aug.resize_pad_seq(
                                                               resize_target_size=PARAMS.resize_target_size,
                                                               pad_method=PARAMS.pad_method,
                                                               pad_size=PARAMS.pad_size),
                                                           'image_augment_inference': aug.pad_to_fit_net(64,
                                                                                                         PARAMS.pad_method),
                                                           'image_augment_with_target_inference': aug.pad_to_fit_net(64,
                                                                                                                     PARAMS.pad_method)
                                                           },
                                   },
                'pad_tta': {'dataset_params': {'h': PARAMS.image_h,
                                               'w': PARAMS.image_w,
                                               'image_source': PARAMS.image_source,
                                               'use_depth': USE_DEPTH,
                                               'MEAN': MEAN,
                                               'STD': STD
                                               },
                            'loader_params': {'training': {'batch_size': PARAMS.batch_size_train,
                                                           'shuffle': True,
                                                           'num_workers': PARAMS.num_workers,
                                                           'pin_memory': PARAMS.pin_memory
                                                           },
                                              'inference': {'batch_size': PARAMS.batch_size_inference,
                                                            'shuffle': False,
                                                            'num_workers': PARAMS.num_workers,
                                                            'pin_memory': PARAMS.pin_memory
                                                            },
                                              },

                            'augmentation_params': {
                                'image_augment_inference': aug.pad_to_fit_net(64, PARAMS.pad_method),
                                'image_augment_with_target_inference': aug.pad_to_fit_net(64,
                                                                                          PARAMS.pad_method),
                                'tta_transform': aug.test_time_augmentation_transform
                            },
                            },
                'resize': {'dataset_params': {'h': PARAMS.image_h,
                                              'w': PARAMS.image_w,
                                              'image_source': PARAMS.image_source,
                                              'use_depth': USE_DEPTH,
                                              'MEAN': MEAN,
                                              'STD': STD
                                              },
                           'loader_params': {'training': {'batch_size': PARAMS.batch_size_train,
                                                          'shuffle': True,
                                                          'num_workers': PARAMS.num_workers,
                                                          'pin_memory': PARAMS.pin_memory
                                                          },
                                             'inference': {'batch_size': PARAMS.batch_size_inference,
                                                           'shuffle': False,
                                                           'num_workers': PARAMS.num_workers,
                                                           'pin_memory': PARAMS.pin_memory
                                                           },
                                             },

                           'augmentation_params': {'image_augment_train': aug.intensity_seq,
                                                   'image_augment_with_target_train': aug.resize_seq(
                                                       resize_target_size=PARAMS.resize_target_size),
                                                   'image_augment_inference': aug.resize_to_fit_net(
                                                       resize_target_size=PARAMS.resize_target_size),
                                                   'image_augment_with_target_inference': aug.resize_to_fit_net(
                                                       resize_target_size=PARAMS.resize_target_size)
                                                   },
                           },
                'resize_tta': {'dataset_params': {'h': PARAMS.image_h,
                                                  'w': PARAMS.image_w,
                                                  'image_source': PARAMS.image_source,
                                                  'use_depth': USE_DEPTH,
                                                  'MEAN': MEAN,
                                                  'STD': STD
                                                  },
                               'loader_params': {'training': {'batch_size': PARAMS.batch_size_train,
                                                              'shuffle': True,
                                                              'num_workers': PARAMS.num_workers,
                                                              'pin_memory': PARAMS.pin_memory
                                                              },
                                                 'inference': {'batch_size': PARAMS.batch_size_inference,
                                                               'shuffle': False,
                                                               'num_workers': PARAMS.num_workers,
                                                               'pin_memory': PARAMS.pin_memory
                                                               },
                                                 },

                               'augmentation_params': {
                                   'image_augment_inference': aug.resize_to_fit_net(
                                       resize_target_size=PARAMS.resize_target_size),
                                   'image_augment_with_target_inference': aug.resize_to_fit_net(
                                       resize_target_size=PARAMS.resize_target_size),
                                   'tta_transform': aug.test_time_augmentation_transform
                               },
                               },
                },
    'model': {
        'network': {
            'architecture_config': {'model_params': {'n_filters': PARAMS.n_filters,
                                                     'conv_kernel': PARAMS.conv_kernel,
                                                     'pool_kernel': PARAMS.pool_kernel,
                                                     'pool_stride': PARAMS.pool_stride,
                                                     'repeat_blocks': PARAMS.repeat_blocks,
                                                     'batch_norm': PARAMS.use_batch_norm,
                                                     'dropout': PARAMS.dropout_conv,
                                                     'in_channels': PARAMS.image_channels,
                                                     'out_channels': PARAMS.network_output_channels,
                                                     'nr_outputs': PARAMS.nr_network_outputs,
                                                     'architecture': PARAMS.architecture,
                                                     'activation': PARAMS.network_activation,
                                                     },
                                    'optimizer_params': {'lr': PARAMS.lr,
                                                         },
                                    'regularizer_params': {'regularize': True,
                                                           'weight_decay_conv2d': PARAMS.l2_reg_conv,
                                                           },
                                    'weights_init': {'function': 'xavier',
                                                     },
                                    },
            'training_config': {'epochs': PARAMS.epochs_nr,
                                'shuffle': True,
                                'batch_size': PARAMS.batch_size_train,
                                'fine_tuning': PARAMS.fine_tuning,
                                },
            'callbacks_config': {'model_checkpoint': {
                'filepath': os.path.join(EXPERIMENT_DIR, 'checkpoints', 'network', 'best.torch'),
                'epoch_every': 1,
                'metric_name': PARAMS.validation_metric_name,
                'minimize': PARAMS.minimize_validation_metric},
                'exponential_lr_scheduler': {'gamma': PARAMS.gamma,
                                             'epoch_every': 1},
                'reduce_lr_on_plateau_scheduler': {'metric_name': PARAMS.validation_metric_name,
                                                   'minimize': PARAMS.minimize_validation_metric,
                                                   'reduce_factor': PARAMS.reduce_factor,
                                                   'reduce_patience': PARAMS.reduce_patience,
                                                   'min_lr': PARAMS.min_lr},
                'training_monitor': {'batch_every': 0,
                                     'epoch_every': 1},
                'experiment_timing': {'batch_every': 0,
                                      'epoch_every': 1},
                'validation_monitor': {'epoch_every': 1,
                                       'data_dir': PARAMS.train_images_dir,
                                       'loader_mode': PARAMS.loader_mode,
                                       'use_depth': USE_DEPTH},
                'neptune_monitor': {'model_name': 'network',
                                    'image_nr': 16,
                                    'image_resize': 1.0,
                                    'image_every': None,
                                    'use_depth': USE_DEPTH},
                'early_stopping': {'patience': PARAMS.patience,
                                   'metric_name': PARAMS.validation_metric_name,
                                   'minimize': PARAMS.minimize_validation_metric},
            }
        },
    },
    'tta_generator': {'flip_ud': False,
                      'flip_lr': True,
                      'rotation': False,
                      'color_shift_runs': 0},
    'tta_aggregator': {'tta_inverse_transform': aug.test_time_augmentation_inverse_transform,
                       'method': PARAMS.tta_aggregation_method,
                       'nthreads': PARAMS.num_threads
                       },
    'thresholder': {'threshold_masks': PARAMS.threshold_masks,
                    },
})


#  .______    __  .______    _______  __       __  .__   __.  _______     _______.
#  |   _  \  |  | |   _  \  |   ____||  |     |  | |  \ |  | |   ____|   /       |
#  |  |_)  | |  | |  |_)  | |  |__   |  |     |  | |   \|  | |  |__     |   (----`
#  |   ___/  |  | |   ___/  |   __|  |  |     |  | |  . `  | |   __|     \   \
#  |  |      |  | |  |      |  |____ |  `----.|  | |  |\   | |  |____.----)   |
#  | _|      |__| | _|      |_______||_______||__| |__| \__| |_______|_______/
#


def emptiness_preprocessing_train(config, model_name='network', suffix=''):
    reader_train = Step(name='xy_train{}'.format(suffix),
                        transformer=loaders.XYSplit(train_mode=True, **config.xy_splitter[model_name]),
                        input_data=['input'],
                        adapter=Adapter({'meta': E('input', 'meta')}),
                        experiment_directory=config.execution.experiment_dir)

    reader_inference = Step(name='xy_inference{}'.format(suffix),
                            transformer=loaders.XYSplit(train_mode=True, **config.xy_splitter[model_name]),
                            input_data=['callback_input'],
                            adapter=Adapter({'meta': E('callback_input', 'meta_valid')}),
                            experiment_directory=config.execution.experiment_dir)

    loader = Step(name='loader{}'.format(suffix),
                  transformer=loaders.EmptinessLoader(train_mode=True, **config.loaders.resize),
                  input_steps=[reader_train, reader_inference],
                  adapter=Adapter({'X': E(reader_train.name, 'X'),
                                   'y': E(reader_train.name, 'y'),
                                   'X_valid': E(reader_inference.name, 'X'),
                                   'y_valid': E(reader_inference.name, 'y'),
                                   }),
                  experiment_directory=config.execution.experiment_dir)
    return loader


def emptiness_preprocessing_inference(config, model_name='network', suffix=''):
    reader_inference = Step(name='xy_inference{}'.format(suffix),
                            transformer=loaders.XYSplit(train_mode=False, **config.xy_splitter[model_name]),
                            input_data=['input'],
                            adapter=Adapter({'meta': E('input', 'meta')}),
                            experiment_directory=config.execution.experiment_dir)

    loader = Step(name='loader{}'.format(suffix),
                  transformer=loaders.EmptinessLoader(train_mode=False, **config.loaders.resize),
                  input_steps=[reader_inference],
                  adapter=Adapter({'X': E(reader_inference.name, 'X'),
                                   'y': E(reader_inference.name, 'y'),
                                   }),
                  experiment_directory=config.execution.experiment_dir,
                  cache_output=True)
    return loader


def network(config, suffix='', train_mode=True):
    if train_mode:
        preprocessing = emptiness_preprocessing_train(config, model_name='network', suffix=suffix)
    else:
        preprocessing = emptiness_preprocessing_inference(config, suffix=suffix)

    network = utils.FineTuneStep(name='network{}'.format(suffix),
                                 transformer=models.SegmentationModel(**config.model['network']),
                                 input_data=['callback_input'],
                                 input_steps=[preprocessing],
                                 adapter=Adapter({'datagen': E(preprocessing.name, 'datagen'),
                                                  'validation_datagen': E(preprocessing.name, 'validation_datagen'),
                                                  'meta_valid': E('callback_input', 'meta_valid'),
                                                  }),
                                 is_trainable=True,
                                 fine_tuning=config.model.network.training_config.fine_tuning,
                                 experiment_directory=config.execution.experiment_dir)

    mask_resize = Step(name='mask_resize{}'.format(suffix),
                       transformer=utils.make_apply_transformer(partial(postprocessing.resize_emptiness_predictions,
                                                                        target_size=config.general.original_size),
                                                                output_name='resized_images',
                                                                apply_on=['images']),
                       input_steps=[network],
                       adapter=Adapter({'images': E(network.name, 'mask_prediction'),
                                        }),
                       experiment_directory=config.execution.experiment_dir)

    return mask_resize


#   __________   ___  _______   ______  __    __  .___________. __    ______   .__   __.
#  |   ____\  \ /  / |   ____| /      ||  |  |  | |           ||  |  /  __  \  |  \ |  |
#  |  |__   \  V  /  |  |__   |  ,----'|  |  |  | `---|  |----`|  | |  |  |  | |   \|  |
#  |   __|   >   <   |   __|  |  |     |  |  |  |     |  |     |  | |  |  |  | |  . `  |
#  |  |____ /  .  \  |  |____ |  `----.|  `--'  |     |  |     |  | |  `--'  | |  |\   |
#  |_______/__/ \__\ |_______| \______| \______/      |__|     |__|  \______/  |__| \__|
#


def train_evaluate_cv():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    if DEV_MODE:
        meta = meta.sample(PARAMS.dev_mode_size, random_state=SEED)

    meta_train = meta[meta['is_train'] == 1]

    with neptune.create_experiment(name=EXPERIMENT_NAME,
                                   params=PARAMS,
                                   tags=TAGS + ['train', 'evaluate', 'on_cv_folds'],
                                   upload_source_files=get_filepaths(),
                                   properties={'experiment_dir': EXPERIMENT_DIR}):

        cv = utils.KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)

        fold_auc = []
        for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1))):
            train_data_split, valid_data_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]

            if USE_AUXILIARY_DATA:
                auxiliary = pd.read_csv(PARAMS.auxiliary_metadata_filepath)
                train_auxiliary = auxiliary[auxiliary[ID_COLUMN].isin(valid_data_split[ID_COLUMN].tolist())]
                train_data_split = pd.concat([train_data_split, train_auxiliary], axis=0)

            LOGGER.info('Started fold {}'.format(fold_id))
            auc, _ = fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id)
            LOGGER.info('Fold {} AUC {}'.format(fold_id, auc))
            neptune.send_metric('Fold {} AUC'.format(fold_id), auc)

            fold_auc.append(auc)

        auc_mean, auc_std = np.mean(fold_auc), np.std(fold_auc)
        log_scores(auc_mean, auc_std)


def train_evaluate_predict_cv():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    if DEV_MODE:
        meta = meta.sample(PARAMS.dev_mode_size, random_state=SEED)

    meta_train = meta[meta['is_train'] == 1]
    meta_test = meta[meta['is_train'] == 0]

    with neptune.create_experiment(name=EXPERIMENT_NAME,
                                   params=PARAMS,
                                   tags=TAGS + ['train', 'evaluate', 'predict', 'on_cv_folds'],
                                   upload_source_files=get_filepaths(),
                                   properties={'experiment_dir': EXPERIMENT_DIR}):

        cv = utils.KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)

        fold_auc, out_of_fold_train_predictions, out_of_fold_test_predictions = [], [], []
        for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1))):
            train_data_split, valid_data_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]

            if USE_AUXILIARY_DATA:
                auxiliary = pd.read_csv(PARAMS.auxiliary_metadata_filepath)
                train_auxiliary = auxiliary[auxiliary[ID_COLUMN].isin(valid_data_split[ID_COLUMN].tolist())]
                train_data_split = pd.concat([train_data_split, train_auxiliary], axis=0)

            LOGGER.info('Started fold {}'.format(fold_id))
            auc, out_of_fold_prediction, test_prediction = fold_fit_evaluate_predict_loop(train_data_split,
                                                                                          valid_data_split,
                                                                                          meta_test,
                                                                                          fold_id)

            LOGGER.info('Fold {} AUC {}'.format(fold_id, auc))
            neptune.send_metric('Fold {} AUC'.format(fold_id), auc)

            fold_auc.append(auc)
            out_of_fold_train_predictions.append(out_of_fold_prediction)
            out_of_fold_test_predictions.append(test_prediction)

        train_ids, train_predictions = [], []
        for idx_fold, train_pred_fold in out_of_fold_train_predictions:
            train_ids.extend(idx_fold)
            train_predictions.extend(train_pred_fold)

        auc_mean, auc_std = np.mean(fold_auc), np.std(fold_auc)
        log_scores(auc_mean, auc_std)
        save_predictions(train_ids, train_predictions, meta_test, out_of_fold_test_predictions)


def evaluate_cv():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    if DEV_MODE:
        meta = meta.sample(PARAMS.dev_mode_size, random_state=SEED)

    meta_train = meta[meta['is_train'] == 1]

    with neptune.create_experiment(name=EXPERIMENT_NAME,
                                   params=PARAMS,
                                   tags=TAGS + ['evaluate', 'on_cv_folds'],
                                   upload_source_files=get_filepaths(),
                                   properties={'experiment_dir': EXPERIMENT_DIR}):

        cv = utils.KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)

        fold_auc = []
        for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1))):
            valid_data_split = meta_train.iloc[valid_idx]

            LOGGER.info('Started fold {}'.format(fold_id))
            auc, _ = fold_evaluate_loop(valid_data_split, fold_id)
            LOGGER.info('Fold {} AUC {}'.format(fold_id, auc))
            neptune.send_metric('Fold {} AUC'.format(fold_id), auc)

            fold_auc.append(auc)

        auc_mean, auc_std = np.mean(fold_auc), np.std(fold_auc)
        log_scores(auc_mean, auc_std)


def evaluate_predict_cv():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    if DEV_MODE:
        meta = meta.sample(PARAMS.dev_mode_size, random_state=SEED)

    meta_train = meta[meta['is_train'] == 1]
    meta_test = meta[meta['is_train'] == 0]

    with neptune.create_experiment(name=EXPERIMENT_NAME,
                                   params=PARAMS,
                                   tags=TAGS + ['evaluate', 'predict', 'on_cv_folds'],
                                   upload_source_files=get_filepaths(),
                                   properties={'experiment_dir': EXPERIMENT_DIR}):

        cv = utils.KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)

        fold_auc, out_of_fold_train_predictions, out_of_fold_test_predictions = [], [], []
        for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1))):
            valid_data_split = meta_train.iloc[valid_idx]

            LOGGER.info('Started fold {}'.format(fold_id))
            auc, out_of_fold_prediction, test_prediction = fold_evaluate_predict_loop(valid_data_split,
                                                                                      meta_test,
                                                                                      fold_id)

            LOGGER.info('Fold {} AUC {}'.format(fold_id, auc))
            neptune.send_metric('Fold {} AUC'.format(fold_id), auc)

            fold_auc.append(auc)
            out_of_fold_train_predictions.append(out_of_fold_prediction)
            out_of_fold_test_predictions.append(test_prediction)

        train_ids, train_predictions = [], []
        for idx_fold, train_pred_fold in out_of_fold_train_predictions:
            train_ids.extend(idx_fold)
            train_predictions.extend(train_pred_fold)

        auc_mean, auc_std = np.mean(fold_auc), np.std(fold_auc)
        log_scores(auc_mean, auc_std)
        save_predictions(train_ids, train_predictions, meta_test, out_of_fold_test_predictions)


def fold_fit_evaluate_predict_loop(train_data_split, valid_data_split, test, fold_id):
    auc, predicted_masks_valid = fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id)

    test_pipe_input = {'input': {'meta': test
                                 },
                       'callback_input': {'meta_valid': None
                                          }
                       }
    pipeline_network = network(config=CONFIG, suffix='_fold_{}'.format(fold_id), train_mode=False)
    LOGGER.info('Start pipeline transform on test')
    pipeline_network.clean_cache()
    predicted_masks_test = pipeline_network.transform(test_pipe_input)
    utils.clean_object_from_memory(pipeline_network)

    predicted_masks_test = predicted_masks_test['resized_images']
    return auc, predicted_masks_valid, predicted_masks_test


def fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id):
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
    valid_ids = valid_data_split[ID_COLUMN].tolist()

    LOGGER.info('Start pipeline fit and transform on train')

    config = add_fold_id_suffix(CONFIG, fold_id)

    pipeline_network = network(config=config, suffix='_fold_{}'.format(fold_id), train_mode=True)
    pipeline_network.clean_cache()
    pipeline_network.fit_transform(train_pipe_input)
    utils.clean_object_from_memory(pipeline_network)

    LOGGER.info('Start pipeline transform on valid')
    pipeline_network = network(config=config, suffix='_fold_{}'.format(fold_id), train_mode=False)
    pipeline_network.clean_cache()
    predicted_masks_valid = pipeline_network.transform(valid_pipe_input)
    utils.clean_object_from_memory(pipeline_network)

    y_pred_valid = predicted_masks_valid['resized_images']
    y_true_valid = valid_data_split[Y_COLUMN].values

    auc = calculate_scores(y_true_valid, y_pred_valid)
    return auc, (valid_ids, y_pred_valid)


def fold_evaluate_predict_loop(valid_data_split, test, fold_id):
    auc, predicted_masks_valid = fold_evaluate_loop(valid_data_split, fold_id)

    test_pipe_input = {'input': {'meta': test
                                 },
                       'callback_input': {'meta_valid': None
                                          }
                       }
    pipeline_network = network(config=CONFIG, suffix='_fold_{}'.format(fold_id), train_mode=False)
    LOGGER.info('Start pipeline transform on test')
    pipeline_network.clean_cache()
    predicted_masks_test = pipeline_network.transform(test_pipe_input)
    utils.clean_object_from_memory(pipeline_network)

    predicted_masks_test = predicted_masks_test['resized_images']
    return auc, predicted_masks_valid, predicted_masks_test


def fold_evaluate_loop(valid_data_split, fold_id):
    valid_pipe_input = {'input': {'meta': valid_data_split
                                  },
                        'callback_input': {'meta_valid': None
                                           }
                        }
    valid_ids = valid_data_split[ID_COLUMN].tolist()

    LOGGER.info('Start pipeline transform on valid')
    pipeline_network = network(config=CONFIG, suffix='_fold_{}'.format(fold_id), train_mode=False)
    pipeline_network.clean_cache()
    predicted_masks_valid = pipeline_network.transform(valid_pipe_input)
    utils.clean_object_from_memory(pipeline_network)

    y_pred_valid = predicted_masks_valid['resized_images']
    y_true_valid = valid_data_split[Y_COLUMN].values

    auc = calculate_scores(y_true_valid, y_pred_valid)
    return auc, (valid_ids, y_pred_valid)


#   __    __  .___________. __   __          _______.
#  |  |  |  | |           ||  | |  |        /       |
#  |  |  |  | `---|  |----`|  | |  |       |   (----`
#  |  |  |  |     |  |     |  | |  |        \   \
#  |  `--'  |     |  |     |  | |  `----.----)   |
#   \______/      |__|     |__| |_______|_______/
#

def calculate_scores(y_true, y_pred):
    y_pred = np.array([y[1, 0, 0] for y in y_pred])
    auc = roc_auc_score(y_true, y_pred)
    return auc


def add_fold_id_suffix(config, fold_id):
    config['model']['network']['callbacks_config']['neptune_monitor']['model_name'] = 'network_{}'.format(fold_id)
    checkpoint_filepath = config['model']['network']['callbacks_config']['model_checkpoint']['filepath']
    fold_checkpoint_filepath = checkpoint_filepath.replace('network/best.torch',
                                                           'network_{}/best.torch'.format(fold_id))
    config['model']['network']['callbacks_config']['model_checkpoint']['filepath'] = fold_checkpoint_filepath
    return config


def log_scores(auc_mean, auc_std):
    LOGGER.info('AUC mean {}, AUC std {}'.format(auc_mean, auc_std))
    neptune.send_metric('AUC', auc_mean)
    neptune.send_metric('AUC STD', auc_std)


def save_predictions(train_ids, train_predictions, meta_test, out_of_fold_test_predictions):
    averaged_mask_predictions_test = np.mean(np.array(out_of_fold_test_predictions), axis=0)

    LOGGER.info('Saving predictions')
    out_of_fold_train_predictions_path = os.path.join(EXPERIMENT_DIR, 'out_of_fold_train_predictions.pkl')
    joblib.dump({'ids': train_ids,
                 'images': train_predictions}, out_of_fold_train_predictions_path)

    out_of_fold_test_predictions_path = os.path.join(EXPERIMENT_DIR, 'out_of_fold_test_predictions.pkl')
    joblib.dump({'ids': meta_test[ID_COLUMN].tolist(),
                 'images': averaged_mask_predictions_test}, out_of_fold_test_predictions_path)


#  .___  ___.      ___       __  .__   __.
#  |   \/   |     /   \     |  | |  \ |  |
#  |  \  /  |    /  ^  \    |  | |   \|  |
#  |  |\/|  |   /  /_\  \   |  | |  . `  |
#  |  |  |  |  /  _____  \  |  | |  |\   |
#  |__|  |__| /__/     \__\ |__| |__| \__|
#

if __name__ == '__main__':
    train_evaluate_predict_cv()
