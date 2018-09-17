from functools import partial
import os
import shutil

from attrdict import AttrDict
import neptune
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from steppy.base import Step, IdentityOperation
from steppy.adapter import Adapter, E

from common_blocks import augmentation as aug
from common_blocks import metrics
from common_blocks import models
from common_blocks import pipelines
from common_blocks import utils
from common_blocks import postprocessing

CTX = neptune.Context()
LOGGER = utils.init_logger()

#    ______   ______   .__   __.  _______  __    _______      _______.
#   /      | /  __  \  |  \ |  | |   ____||  |  /  _____|    /       |
#  |  ,----'|  |  |  | |   \|  | |  |__   |  | |  |  __     |   (----`
#  |  |     |  |  |  | |  . `  | |   __|  |  | |  | |_ |     \   \
#  |  `----.|  `--'  | |  |\   | |  |     |  | |  |__| | .----)   |
#   \______| \______/  |__| \__| |__|     |__|  \______| |_______/
#

EXPERIMENT_DIR = '/output/experiment'
CLONE_EXPERIMENT_DIR_FROM = ''  # When running eval in the cloud specify this as for example /input/SAL-14/output/experiment
OVERWRITE_EXPERIMENT_DIR = False
DEV_MODE = False

if OVERWRITE_EXPERIMENT_DIR and os.path.isdir(EXPERIMENT_DIR):
    shutil.rmtree(EXPERIMENT_DIR)
if CLONE_EXPERIMENT_DIR_FROM != '':
    if os.path.exists(EXPERIMENT_DIR):
        shutil.rmtree(EXPERIMENT_DIR)
    shutil.copytree(CLONE_EXPERIMENT_DIR_FROM, EXPERIMENT_DIR)

if CTX.params.__class__.__name__ == 'OfflineContextParams':
    PARAMS = utils.read_yaml().parameters
else:
    PARAMS = CTX.params

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 1234
ID_COLUMN = 'id'
DEPTH_COLUMN = 'z'
X_COLUMN = 'file_path_image'
Y_COLUMN = 'file_path_mask'

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
        'unet': {'x_columns': [X_COLUMN],
                 'y_columns': [Y_COLUMN],
                 },
    },
    'reader': {
        'unet': {'x_columns': [X_COLUMN],
                 'y_columns': [Y_COLUMN],
                 },
    },
    'loaders': {'resize_and_pad': {'dataset_params': {'h': PARAMS.image_h,
                                                      'w': PARAMS.image_w,
                                                      'image_source': PARAMS.image_source,
                                                      'target_format': PARAMS.target_format,
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
                                               'target_format': PARAMS.target_format,
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
                                              'target_format': PARAMS.target_format,
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
                                                   'image_augment_with_target_train': aug.affine_seq
                                                   },
                           },
                'resize_tta': {'dataset_params': {'h': PARAMS.image_h,
                                                  'w': PARAMS.image_w,
                                                  'image_source': PARAMS.image_source,
                                                  'target_format': PARAMS.target_format,
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

                               'augmentation_params': {'tta_transform': aug.test_time_augmentation_transform
                                                       },
                               },
                },
    'model': {
        'unet': {
            'architecture_config': {'model_params': {'n_filters': PARAMS.n_filters,
                                                     'conv_kernel': PARAMS.conv_kernel,
                                                     'pool_kernel': PARAMS.pool_kernel,
                                                     'pool_stride': PARAMS.pool_stride,
                                                     'repeat_blocks': PARAMS.repeat_blocks,
                                                     'batch_norm': PARAMS.use_batch_norm,
                                                     'dropout': PARAMS.dropout_conv,
                                                     'in_channels': PARAMS.image_channels,
                                                     'out_channels': PARAMS.unet_output_channels,
                                                     'nr_outputs': PARAMS.nr_unet_outputs,
                                                     'encoder': PARAMS.encoder,
                                                     'activation': PARAMS.unet_activation,
                                                     'dice_weight': PARAMS.dice_weight,
                                                     'bce_weight': PARAMS.bce_weight,
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
                'filepath': os.path.join(EXPERIMENT_DIR, 'checkpoints', 'unet', 'best.torch'),
                'epoch_every': 1,
                'metric_name': PARAMS.validation_metric_name,
                'minimize': PARAMS.minimize_validation_metric},
                'lr_scheduler': {'gamma': PARAMS.gamma,
                                 'epoch_every': 1},
                'training_monitor': {'batch_every': 0,
                                     'epoch_every': 1},
                'experiment_timing': {'batch_every': 0,
                                      'epoch_every': 1},
                'validation_monitor': {'epoch_every': 1,
                                       'data_dir': PARAMS.train_images_dir,
                                       'loader_mode': PARAMS.loader_mode},
                'neptune_monitor': {'model_name': 'unet',
                                    'image_nr': 4,
                                    'image_resize': 1.0},
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

def unet(config, suffix='', train_mode=True):
    if train_mode:
        preprocessing = pipelines.preprocessing_train(config, model_name='unet', suffix=suffix)
    else:
        preprocessing = pipelines.preprocessing_inference(config, suffix=suffix)

    unet = utils.FineTuneStep(name='unet{}'.format(suffix),
                              transformer=models.PyTorchUNet(**config.model['unet']),
                              input_data=['callback_input'],
                              input_steps=[preprocessing],
                              adapter=Adapter({'datagen': E(preprocessing.name, 'datagen'),
                                               'validation_datagen': E(preprocessing.name, 'validation_datagen'),
                                               'meta_valid': E('callback_input', 'meta_valid'),
                                               }),
                              is_trainable=True,
                              fine_tuning=config.model.unet.training_config.fine_tuning,
                              experiment_directory=config.execution.experiment_dir)

    if config.general.loader_mode == 'resize_and_pad':
        size_adjustment_function = partial(postprocessing.crop_image, target_size=config.general.original_size)
    elif config.general.loader_mode == 'resize':
        size_adjustment_function = partial(postprocessing.resize_image, target_size=config.general.original_size)
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize{}'.format(suffix),
                       transformer=utils.make_apply_transformer(size_adjustment_function,
                                                                output_name='resized_images',
                                                                apply_on=['images']),
                       input_steps=[unet],
                       adapter=Adapter({'images': E(unet.name, 'mask_prediction'),
                                        }),
                       experiment_directory=config.execution.experiment_dir)

    return mask_resize


def unet_tta(config, suffix=''):
    preprocessing, tta_generator = pipelines.preprocessing_inference_tta(config, model_name='unet')

    unet = Step(name='unet{}'.format(suffix),
                transformer=models.PyTorchUNet(**config.model['unet']),
                input_data=['callback_input'],
                input_steps=[preprocessing],
                is_trainable=True,
                experiment_directory=config.execution.experiment_dir)

    tta_aggregator = pipelines.aggregator('tta_aggregator{}'.format(suffix), unet,
                                          tta_generator=tta_generator,
                                          experiment_directory=config.execution.experiment_dir,
                                          config=config.tta_aggregator)

    prediction_renamed = Step(name='prediction_renamed{}'.format(suffix),
                              transformer=IdentityOperation(),
                              input_steps=[tta_aggregator],
                              adapter=Adapter({'mask_prediction': E(tta_aggregator.name, 'aggregated_prediction')
                                               }),
                              experiment_directory=config.execution.experiment_dir)

    if config.general.loader_mode == 'resize_and_pad':
        size_adjustment_function = partial(postprocessing.crop_image, target_size=config.general.original_size)
    elif config.general.loader_mode == 'resize':
        size_adjustment_function = partial(postprocessing.resize_image, target_size=config.general.original_size)
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize{}'.format(suffix),
                       transformer=utils.make_apply_transformer(size_adjustment_function,
                                                                output_name='resized_images',
                                                                apply_on=['images']),
                       input_steps=[prediction_renamed],
                       adapter=Adapter({'images': E(prediction_renamed.name, 'mask_prediction'),
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

def prepare_metadata():
    LOGGER.info('creating metadata')
    meta = utils.generate_metadata(train_images_dir=PARAMS.train_images_dir,
                                   test_images_dir=PARAMS.test_images_dir,
                                   depths_filepath=PARAMS.depths_filepath
                                   )
    meta.to_csv(PARAMS.metadata_filepath, index=None)


def train():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    cv = utils.KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)
    for train_idx, valid_idx in cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1)):
        break

    meta_train_split, meta_valid_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]

    if DEV_MODE:
        meta_train_split = meta_train_split.sample(PARAMS.dev_mode_size, random_state=SEED)
        meta_valid_split = meta_valid_split.sample(int(PARAMS.dev_mode_size / 2), random_state=SEED)

    data = {'input': {'meta': meta_train_split
                      },
            'callback_input': {'meta_valid': meta_valid_split
                               }
            }

    pipeline_network = unet(config=CONFIG, train_mode=True)
    pipeline_network.clean_cache()
    pipeline_network.fit_transform(data)
    pipeline_network.clean_cache()


def evaluate():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    cv = utils.KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)
    for train_idx, valid_idx in cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1)):
        break

    meta_valid_split = meta_train.iloc[valid_idx]
    y_true_valid = utils.read_masks(meta_valid_split[Y_COLUMN].values)

    if DEV_MODE:
        meta_valid_split = meta_valid_split.sample(PARAMS.dev_mode_size, random_state=SEED)

    data = {'input': {'meta': meta_valid_split,
                      },
            'callback_input': {'meta_valid': None
                               }
            }
    pipeline_network = unet(config=CONFIG, train_mode=False)
    pipeline_postprocessing = pipelines.mask_postprocessing(config=CONFIG)
    pipeline_network.clean_cache()
    output = pipeline_network.transform(data)
    valid_masks = {'input_masks': output
                   }
    output = pipeline_postprocessing.transform(valid_masks)
    pipeline_network.clean_cache()
    pipeline_postprocessing.clean_cache()
    y_pred_valid = output['binarized_images']

    LOGGER.info('Calculating IOU and IOUT Scores')
    iou_score, iout_score = calculate_scores(y_true_valid, y_pred_valid)
    LOGGER.info('IOU score on validation is {}'.format(iou_score))
    CTX.channel_send('IOU', 0, iou_score)
    LOGGER.info('IOUT score on validation is {}'.format(iout_score))
    CTX.channel_send('IOUT', 0, iout_score)

    results_filepath = os.path.join(EXPERIMENT_DIR, 'validation_results.pkl')
    LOGGER.info('Saving validation results to {}'.format(results_filepath))
    joblib.dump((meta_valid_split, y_true_valid, y_pred_valid), results_filepath)


def predict():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_test = meta[meta['is_train'] == 0]

    if DEV_MODE:
        meta_test = meta_test.sample(PARAMS.dev_mode_size, random_state=SEED)

    data = {'input': {'meta': meta_test,
                      },
            'callback_input': {'meta_valid': None
                               }
            }

    pipeline_network = unet(config=CONFIG, train_mode=False)
    pipeline_postprocessing = pipelines.mask_postprocessing(config=CONFIG)
    pipeline_network.clean_cache()
    predicted_masks = pipeline_network.transform(data)
    test_masks = {'input_masks': predicted_masks
                  }
    output = pipeline_postprocessing.transform(test_masks)
    pipeline_network.clean_cache()
    pipeline_postprocessing.clean_cache()
    y_pred_test = output['binarized_images']

    submission = utils.create_submission(meta_test, y_pred_test)

    submission_filepath = os.path.join(EXPERIMENT_DIR, 'submission.csv')

    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    LOGGER.info('submission saved to {}'.format(submission_filepath))
    LOGGER.info('submission head \n\n{}'.format(submission.head()))


def train_evaluate_cv():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    if DEV_MODE:
        meta = meta.sample(PARAMS.dev_mode_size, random_state=SEED)

    meta_train = meta[meta['is_train'] == 1]

    cv = utils.KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)

    fold_iou, fold_iout = [], []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1))):
        train_data_split, valid_data_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]

        LOGGER.info('Started fold {}'.format(fold_id))
        iou, iout, _ = fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id)
        LOGGER.info('Fold {} IOU {}'.format(fold_id, iou))
        CTX.channel_send('Fold {} IOU'.format(fold_id), 0, iou)
        LOGGER.info('Fold {} IOUT {}'.format(fold_id, iout))
        CTX.channel_send('Fold {} IOUT'.format(fold_id), 0, iout)

        fold_iou.append(iou)
        fold_iout.append(iout)

    iou_mean, iou_std = np.mean(fold_iou), np.std(fold_iou)
    iout_mean, iout_std = np.mean(fold_iout), np.std(fold_iout)

    log_scores(iou_mean, iou_std, iout_mean, iout_std)


def train_evaluate_predict_cv():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    if DEV_MODE:
        meta = meta.sample(PARAMS.dev_mode_size, random_state=SEED)

    meta_train = meta[meta['is_train'] == 1]
    meta_test = meta[meta['is_train'] == 0]

    cv = utils.KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)

    fold_iou, fold_iout, out_of_fold_train_predictions, out_of_fold_test_predictions = [], [], [], []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1))):
        train_data_split, valid_data_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]

        LOGGER.info('Started fold {}'.format(fold_id))
        iou, iout, out_of_fold_prediction, test_prediction = fold_fit_evaluate_predict_loop(train_data_split,
                                                                                            valid_data_split,
                                                                                            meta_test,
                                                                                            fold_id)

        LOGGER.info('Fold {} IOU {}'.format(fold_id, iou))
        CTX.channel_send('Fold {} IOU'.format(fold_id), 0, iou)
        LOGGER.info('Fold {} IOUT {}'.format(fold_id, iout))
        CTX.channel_send('Fold {} IOUT'.format(fold_id), 0, iout)

        fold_iou.append(iou)
        fold_iout.append(iout)
        out_of_fold_train_predictions.append(out_of_fold_prediction)
        out_of_fold_test_predictions.append(test_prediction)

    train_ids, train_predictions = [], []
    for idx_fold, train_pred_fold in out_of_fold_train_predictions:
        train_ids.extend(idx_fold)
        train_predictions.extend(train_pred_fold)

    iou_mean, iou_std = np.mean(fold_iou), np.std(fold_iou)
    iout_mean, iout_std = np.mean(fold_iout), np.std(fold_iout)

    log_scores(iou_mean, iou_std, iout_mean, iout_std)

    save_predictions(train_ids, train_predictions, meta_test, out_of_fold_test_predictions)


def evaluate_cv():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    if DEV_MODE:
        meta = meta.sample(PARAMS.dev_mode_size, random_state=SEED)

    meta_train = meta[meta['is_train'] == 1]

    cv = utils.KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)

    fold_iou, fold_iout = [], []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1))):
        valid_data_split = meta_train.iloc[valid_idx]

        LOGGER.info('Started fold {}'.format(fold_id))
        iou, iout, _ = fold_evaluate_loop(valid_data_split, fold_id)
        LOGGER.info('Fold {} IOU {}'.format(fold_id, iou))
        CTX.channel_send('Fold {} IOU'.format(fold_id), 0, iou)
        LOGGER.info('Fold {} IOUT {}'.format(fold_id, iout))
        CTX.channel_send('Fold {} IOUT'.format(fold_id), 0, iout)

        fold_iou.append(iou)
        fold_iout.append(iout)

    iou_mean, iou_std = np.mean(fold_iou), np.std(fold_iou)
    iout_mean, iout_std = np.mean(fold_iout), np.std(fold_iout)

    log_scores(iou_mean, iou_std, iout_mean, iout_std)


def evaluate_predict_cv():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    if DEV_MODE:
        meta = meta.sample(PARAMS.dev_mode_size, random_state=SEED)

    meta_train = meta[meta['is_train'] == 1]
    meta_test = meta[meta['is_train'] == 0]

    cv = utils.KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=SEED)

    fold_iou, fold_iout, out_of_fold_train_predictions, out_of_fold_test_predictions = [], [], [], []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[DEPTH_COLUMN].values.reshape(-1))):
        valid_data_split = meta_train.iloc[valid_idx]

        LOGGER.info('Started fold {}'.format(fold_id))
        iou, iout, out_of_fold_prediction, test_prediction = fold_evaluate_predict_loop(valid_data_split,
                                                                                        meta_test,
                                                                                        fold_id)

        LOGGER.info('Fold {} IOU {}'.format(fold_id, iou))
        CTX.channel_send('Fold {} IOU'.format(fold_id), 0, iou)
        LOGGER.info('Fold {} IOUT {}'.format(fold_id, iout))
        CTX.channel_send('Fold {} IOUT'.format(fold_id), 0, iout)

        fold_iou.append(iou)
        fold_iout.append(iout)
        out_of_fold_train_predictions.append(out_of_fold_prediction)
        out_of_fold_test_predictions.append(test_prediction)

    train_ids, train_predictions = [], []
    for idx_fold, train_pred_fold in out_of_fold_train_predictions:
        train_ids.extend(idx_fold)
        train_predictions.extend(train_pred_fold)

    iou_mean, iou_std = np.mean(fold_iou), np.std(fold_iou)
    iout_mean, iout_std = np.mean(fold_iout), np.std(fold_iout)

    log_scores(iou_mean, iou_std, iout_mean, iout_std)
    save_predictions(train_ids, train_predictions, meta_test, out_of_fold_test_predictions)


def fold_fit_evaluate_predict_loop(train_data_split, valid_data_split, test, fold_id):
    iou, iout, predicted_masks_valid = fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id)

    test_pipe_input = {'input': {'meta': test
                                 },
                       'callback_input': {'meta_valid': None
                                          }
                       }
    pipeline_network = unet(config=CONFIG, suffix='_fold_{}'.format(fold_id), train_mode=False)
    LOGGER.info('Start pipeline transform on test')
    pipeline_network.clean_cache()
    predicted_masks_test = pipeline_network.transform(test_pipe_input)
    utils.clean_object_from_memory(pipeline_network)

    predicted_masks_test = predicted_masks_test['resized_images']
    return iou, iout, predicted_masks_valid, predicted_masks_test


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

    pipeline_network = unet(config=config, suffix='_fold_{}'.format(fold_id), train_mode=True)
    pipeline_network.clean_cache()
    pipeline_network.fit_transform(train_pipe_input)
    utils.clean_object_from_memory(pipeline_network)

    LOGGER.info('Start pipeline transform on valid')
    pipeline_network = unet(config=config, suffix='_fold_{}'.format(fold_id), train_mode=False)
    pipeline_postprocessing = pipelines.mask_postprocessing(config=config, suffix='_fold_{}'.format(fold_id))
    pipeline_network.clean_cache()
    pipeline_postprocessing.clean_cache()
    predicted_masks_valid = pipeline_network.transform(valid_pipe_input)
    valid_pipe_masks = {'input_masks': predicted_masks_valid
                        }
    output_valid = pipeline_postprocessing.transform(valid_pipe_masks)
    utils.clean_object_from_memory(pipeline_network)

    y_pred_valid = output_valid['binarized_images']
    y_true_valid = utils.read_masks(valid_data_split[Y_COLUMN].values)

    iou, iout = calculate_scores(y_true_valid, y_pred_valid)

    predicted_masks_valid = predicted_masks_valid['resized_images']
    return iou, iout, (valid_ids, predicted_masks_valid)


def fold_evaluate_predict_loop(valid_data_split, test, fold_id):
    iou, iout, predicted_masks_valid = fold_evaluate_loop(valid_data_split, fold_id)

    test_pipe_input = {'input': {'meta': test
                                 },
                       'callback_input': {'meta_valid': None
                                          }
                       }
    pipeline_network = unet(config=CONFIG, suffix='_fold_{}'.format(fold_id), train_mode=False)
    LOGGER.info('Start pipeline transform on test')
    pipeline_network.clean_cache()
    predicted_masks_test = pipeline_network.transform(test_pipe_input)
    utils.clean_object_from_memory(pipeline_network)

    predicted_masks_test = predicted_masks_test['resized_images']
    return iou, iout, predicted_masks_valid, predicted_masks_test


def fold_evaluate_loop(valid_data_split, fold_id):
    valid_pipe_input = {'input': {'meta': valid_data_split
                                  },
                        'callback_input': {'meta_valid': None
                                           }
                        }
    valid_ids = valid_data_split[ID_COLUMN].tolist()

    LOGGER.info('Start pipeline transform on valid')
    pipeline_network = unet(config=CONFIG, suffix='_fold_{}'.format(fold_id), train_mode=False)
    pipeline_postprocessing = pipelines.mask_postprocessing(config=CONFIG, suffix='_fold_{}'.format(fold_id))
    pipeline_network.clean_cache()
    pipeline_postprocessing.clean_cache()
    predicted_masks_valid = pipeline_network.transform(valid_pipe_input)
    valid_pipe_masks = {'input_masks': predicted_masks_valid
                        }
    output_valid = pipeline_postprocessing.transform(valid_pipe_masks)
    utils.clean_object_from_memory(pipeline_network)

    y_pred_valid = output_valid['binarized_images']
    y_true_valid = utils.read_masks(valid_data_split[Y_COLUMN].values)

    iou, iout = calculate_scores(y_true_valid, y_pred_valid)
    predicted_masks_valid = predicted_masks_valid['resized_images']
    return iou, iout, (valid_ids, predicted_masks_valid)


#   __    __  .___________. __   __          _______.
#  |  |  |  | |           ||  | |  |        /       |
#  |  |  |  | `---|  |----`|  | |  |       |   (----`
#  |  |  |  |     |  |     |  | |  |        \   \
#  |  `--'  |     |  |     |  | |  `----.----)   |
#   \______/      |__|     |__| |_______|_______/
#

def calculate_scores(y_true, y_pred):
    iou = metrics.intersection_over_union(y_true, y_pred)
    iout = metrics.intersection_over_union_thresholds(y_true, y_pred)
    return iou, iout


def add_fold_id_suffix(config, fold_id):
    config['model']['unet']['callbacks_config']['neptune_monitor']['model_name'] = 'unet_{}'.format(fold_id)
    checkpoint_filepath = config['model']['unet']['callbacks_config']['model_checkpoint']['filepath']
    fold_checkpoint_filepath = checkpoint_filepath.replace('unet/best.torch', 'unet_{}/best.torch'.format(fold_id))
    config['model']['unet']['callbacks_config']['model_checkpoint']['filepath'] = fold_checkpoint_filepath
    return config


def log_scores(iou_mean, iou_std, iout_mean, iout_std):
    LOGGER.info('IOU mean {}, IOU std {}'.format(iou_mean, iou_std))
    CTX.channel_send('IOU', 0, iou_mean)
    CTX.channel_send('IOU STD', 0, iou_std)

    LOGGER.info('IOUT mean {}, IOUT std {}'.format(iout_mean, iout_std))
    CTX.channel_send('IOUT', 0, iout_mean)
    CTX.channel_send('IOUT STD', 0, iout_std)


def save_predictions(train_ids, train_predictions, meta_test, out_of_fold_test_predictions):
    averaged_mask_predictions_test = np.mean(np.array(out_of_fold_test_predictions), axis=0)
    pipeline_postprocessing = pipelines.mask_postprocessing(config=CONFIG)
    pipeline_postprocessing.clean_cache()
    test_pipe_masks = {'input_masks': {'resized_images': averaged_mask_predictions_test}
                       }
    y_pred_test = pipeline_postprocessing.transform(test_pipe_masks)['binarized_images']

    LOGGER.info('Saving predictions')
    out_of_fold_train_predictions_path = os.path.join(EXPERIMENT_DIR, 'out_of_fold_train_predictions.pkl')
    joblib.dump({'ids': train_ids,
                 'images': train_predictions}, out_of_fold_train_predictions_path)

    out_of_fold_test_predictions_path = os.path.join(EXPERIMENT_DIR, 'out_of_fold_test_predictions.pkl')
    joblib.dump({'ids': meta_test[ID_COLUMN].tolist(),
                 'images': averaged_mask_predictions_test}, out_of_fold_test_predictions_path)

    submission = utils.create_submission(meta_test, y_pred_test)
    submission_filepath = os.path.join(EXPERIMENT_DIR, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    LOGGER.info('submission saved to {}'.format(submission_filepath))
    LOGGER.info('submission head \n\n{}'.format(submission.head()))


#  .___  ___.      ___       __  .__   __.
#  |   \/   |     /   \     |  | |  \ |  |
#  |  \  /  |    /  ^  \    |  | |   \|  |
#  |  |\/|  |   /  /_\  \   |  | |  . `  |
#  |  |  |  |  /  _____  \  |  | |  |\   |
#  |__|  |__| /__/     \__\ |__| |__| \__|
#

if __name__ == '__main__':
    prepare_metadata()
    train_evaluate_predict_cv()
