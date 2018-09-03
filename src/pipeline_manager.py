import os
import shutil

import neptune
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from .metrics import intersection_over_union, intersection_over_union_thresholds
from . import pipeline_config as cfg
from .pipelines import PIPELINES
from .utils import init_logger, read_masks, create_submission, \
    generate_metadata, set_seed, KFoldBySortedValue, clean_object_from_memory, read_params

LOGGER = init_logger()
CTX = neptune.Context()
PARAMS = read_params(CTX)
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

    def train_evaluate_predict_cv(self, pipeline_name, submit_predictions, dev_mode):
        train_evaluate_predict_cv(pipeline_name, submit_predictions, dev_mode)

    def evaluate_cv(self, pipeline_name, dev_mode):
        evaluate_cv(pipeline_name, dev_mode)

    def evaluate_predict_cv(self, pipeline_name, submit_predictions, dev_mode):
        evaluate_predict_cv(pipeline_name, submit_predictions, dev_mode)


def prepare_metadata():
    LOGGER.info('creating metadata')
    meta = generate_metadata(train_images_dir=PARAMS.train_images_dir,
                             test_images_dir=PARAMS.test_images_dir,
                             depths_filepath=PARAMS.depths_filepath
                             )
    meta.to_csv(PARAMS.metadata_filepath, index=None)


def train(pipeline_name, dev_mode):
    LOGGER.info('training')

    if PARAMS.clone_experiment_dir_from != '':
        if os.path.exists(PARAMS.experiment_dir):
            shutil.rmtree(PARAMS.experiment_dir)
        shutil.copytree(PARAMS.clone_experiment_dir_from, PARAMS.experiment_dir)

    if bool(PARAMS.overwrite) and os.path.isdir(PARAMS.experiment_dir):
        shutil.rmtree(PARAMS.experiment_dir)

    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    cv = KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=cfg.SEED)
    for train_idx, valid_idx in cv.split(meta_train[cfg.DEPTH_COLUMN].values.reshape(-1)):
        break

    meta_train_split, meta_valid_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]

    if dev_mode:
        meta_train_split = meta_train_split.sample(PARAMS.dev_mode_size, random_state=cfg.SEED)
        meta_valid_split = meta_valid_split.sample(int(PARAMS.dev_mode_size / 2), random_state=cfg.SEED)

    data = {'input': {'meta': meta_train_split
                      },
            'callback_input': {'meta_valid': meta_valid_split
                               }
            }

    pipeline_network = PIPELINES[pipeline_name]['network'](config=cfg.SOLUTION_CONFIG, train_mode=True)
    pipeline_network.clean_cache()
    pipeline_network.fit_transform(data)
    pipeline_network.clean_cache()


def evaluate(pipeline_name, dev_mode):
    LOGGER.info('evaluating')

    if PARAMS.clone_experiment_dir_from != '':
        if os.path.exists(PARAMS.experiment_dir):
            shutil.rmtree(PARAMS.experiment_dir)
        shutil.copytree(PARAMS.clone_experiment_dir_from, PARAMS.experiment_dir)

    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    cv = KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=cfg.SEED)
    for train_idx, valid_idx in cv.split(meta_train[cfg.DEPTH_COLUMN].values.reshape(-1)):
        break

    meta_valid_split = meta_train.iloc[valid_idx]
    y_true_valid = read_masks(meta_valid_split[cfg.Y_COLUMNS[0]].values)

    if dev_mode:
        meta_valid_split = meta_valid_split.sample(PARAMS.dev_mode_size, random_state=cfg.SEED)

    data = {'input': {'meta': meta_valid_split,
                      },
            'callback_input': {'meta_valid': None
                               }
            }
    pipeline_network = PIPELINES[pipeline_name]['network'](config=cfg.SOLUTION_CONFIG, train_mode=False)
    pipeline_postprocessing = PIPELINES[pipeline_name]['postprocessing'](config=cfg.SOLUTION_CONFIG)
    pipeline_network.clean_cache()
    output = pipeline_network.transform(data)
    valid_masks = {'input_masks': output
                   }
    output = pipeline_postprocessing.transform(valid_masks)
    pipeline_network.clean_cache()
    pipeline_postprocessing.clean_cache()
    y_pred_valid = output['binarized_images']

    LOGGER.info('Calculating IOU and IOUT Scores')
    iou_score = intersection_over_union(y_true_valid, y_pred_valid)
    LOGGER.info('IOU score on validation is {}'.format(iou_score))
    CTX.channel_send('IOU', 0, iou_score)

    iout_score = intersection_over_union_thresholds(y_true_valid, y_pred_valid)
    LOGGER.info('IOUT score on validation is {}'.format(iout_score))
    CTX.channel_send('IOUT', 0, iout_score)

    results_filepath = os.path.join(PARAMS.experiment_dir, 'validation_results.pkl')
    LOGGER.info('Saving validation results to {}'.format(results_filepath))
    joblib.dump((meta_valid_split, y_true_valid, y_pred_valid), results_filepath)


def predict(pipeline_name, submit_predictions, dev_mode):
    LOGGER.info('predicting')

    if PARAMS.clone_experiment_dir_from != '':
        if os.path.exists(PARAMS.experiment_dir):
            shutil.rmtree(PARAMS.experiment_dir)
        shutil.copytree(PARAMS.clone_experiment_dir_from, PARAMS.experiment_dir)

    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_test = meta[meta['is_train'] == 0]

    if dev_mode:
        meta_test = meta_test.sample(PARAMS.dev_mode_size, random_state=cfg.SEED)

    data = {'input': {'meta': meta_test,
                      },
            'callback_input': {'meta_valid': None
                               }
            }

    pipeline_network = PIPELINES[pipeline_name]['network'](config=cfg.SOLUTION_CONFIG, train_mode=False)
    pipeline_postprocessing = PIPELINES[pipeline_name]['postprocessing'](config=cfg.SOLUTION_CONFIG)
    pipeline_network.clean_cache()
    predicted_masks = pipeline_network.transform(data)
    test_masks = {'input_masks': predicted_masks
                  }
    output = pipeline_postprocessing.transform(test_masks)
    pipeline_network.clean_cache()
    pipeline_postprocessing.clean_cache()
    y_pred_test = output['binarized_images']

    submission = create_submission(meta_test, y_pred_test)

    submission_filepath = os.path.join(PARAMS.experiment_dir, 'submission.csv')

    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    LOGGER.info('submission saved to {}'.format(submission_filepath))
    LOGGER.info('submission head \n\n{}'.format(submission.head()))

    if submit_predictions:
        _make_submission(submission_filepath)


def train_evaluate_cv(pipeline_name, dev_mode):
    LOGGER.info('training')

    if PARAMS.clone_experiment_dir_from != '':
        if os.path.exists(PARAMS.experiment_dir):
            shutil.rmtree(PARAMS.experiment_dir)
        shutil.copytree(PARAMS.clone_experiment_dir_from, PARAMS.experiment_dir)

    if bool(PARAMS.overwrite) and os.path.isdir(PARAMS.experiment_dir):
        shutil.rmtree(PARAMS.experiment_dir)

    if dev_mode:
        meta = pd.read_csv(PARAMS.metadata_filepath, nrows=PARAMS.dev_mode_size)
    else:
        meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    cv = KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=cfg.SEED)

    fold_iou, fold_iout = [], []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[cfg.DEPTH_COLUMN].values.reshape(-1))):
        train_data_split, valid_data_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]

        LOGGER.info('Started fold {}'.format(fold_id))
        iou, iout, _ = _fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id, pipeline_name)
        LOGGER.info('Fold {} IOU {}'.format(fold_id, iou))
        CTX.channel_send('Fold {} IOU'.format(fold_id), 0, iou)
        LOGGER.info('Fold {} IOUT {}'.format(fold_id, iout))
        CTX.channel_send('Fold {} IOUT'.format(fold_id), 0, iout)

        fold_iou.append(iou)
        fold_iout.append(iout)

    iou_mean, iou_std = np.mean(fold_iou), np.std(fold_iou)
    iout_mean, iout_std = np.mean(fold_iout), np.std(fold_iout)

    _log_scores(iou_mean, iou_std, iout_mean, iout_std)


def train_evaluate_predict_cv(pipeline_name, submit_predictions, dev_mode):
    LOGGER.info('training')
    if bool(PARAMS.overwrite) and os.path.isdir(PARAMS.experiment_dir):
        shutil.rmtree(PARAMS.experiment_dir)

    if dev_mode:
        meta = pd.read_csv(PARAMS.metadata_filepath, nrows=PARAMS.dev_mode_size)
    else:
        meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]
    meta_test = meta[meta['is_train'] == 0]

    cv = KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=cfg.SEED)

    fold_iou, fold_iout, out_of_fold_train_predictions, out_of_fold_test_predictions = [], [], [], []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[cfg.DEPTH_COLUMN].values.reshape(-1))):
        train_data_split, valid_data_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]

        LOGGER.info('Started fold {}'.format(fold_id))
        iou, iout, out_of_fold_prediction, test_prediction = _fold_fit_evaluate_predict_loop(train_data_split,
                                                                                             valid_data_split,
                                                                                             meta_test,
                                                                                             fold_id,
                                                                                             pipeline_name)

        LOGGER.info('Fold {} IOU {}'.format(fold_id, iou))
        CTX.channel_send('Fold {} IOU'.format(fold_id), 0, iou)
        LOGGER.info('Fold {} IOUT {}'.format(fold_id, iout))
        CTX.channel_send('Fold {} IOUT'.format(fold_id), 0, iout)

        fold_iou.append(iou)
        fold_iout.append(iout)
        out_of_fold_train_predictions.extend(out_of_fold_prediction)
        out_of_fold_test_predictions.append(test_prediction)

    iou_mean, iou_std = np.mean(fold_iou), np.std(fold_iou)
    iout_mean, iout_std = np.mean(fold_iout), np.std(fold_iout)

    _log_scores(iou_mean, iou_std, iout_mean, iout_std)

    _save_predictions(out_of_fold_train_predictions, out_of_fold_test_predictions, meta_train, meta_test,
                      pipeline_name, submit_predictions)


def evaluate_cv(pipeline_name, dev_mode):
    LOGGER.info('training')

    if PARAMS.clone_experiment_dir_from != '':
        if os.path.exists(PARAMS.experiment_dir):
            shutil.rmtree(PARAMS.experiment_dir)
        shutil.copytree(PARAMS.clone_experiment_dir_from, PARAMS.experiment_dir)

    if dev_mode:
        meta = pd.read_csv(PARAMS.metadata_filepath, nrows=PARAMS.dev_mode_size)
    else:
        meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    cv = KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=cfg.SEED)

    fold_iou, fold_iout = [], []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[cfg.DEPTH_COLUMN].values.reshape(-1))):
        valid_data_split = meta_train.iloc[valid_idx]

        LOGGER.info('Started fold {}'.format(fold_id))
        iou, iout, _ = _fold_evaluate_loop(valid_data_split, fold_id, pipeline_name)
        LOGGER.info('Fold {} IOU {}'.format(fold_id, iou))
        CTX.channel_send('Fold {} IOU'.format(fold_id), 0, iou)
        LOGGER.info('Fold {} IOUT {}'.format(fold_id, iout))
        CTX.channel_send('Fold {} IOUT'.format(fold_id), 0, iout)

        fold_iou.append(iou)
        fold_iout.append(iout)

    iou_mean, iou_std = np.mean(fold_iou), np.std(fold_iou)
    iout_mean, iout_std = np.mean(fold_iout), np.std(fold_iout)

    _log_scores(iou_mean, iou_std, iout_mean, iout_std)


def evaluate_predict_cv(pipeline_name, submit_predictions, dev_mode):
    if dev_mode:
        meta = pd.read_csv(PARAMS.metadata_filepath, nrows=PARAMS.dev_mode_size)
    else:
        meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]
    meta_test = meta[meta['is_train'] == 0]

    cv = KFoldBySortedValue(n_splits=PARAMS.n_cv_splits, shuffle=PARAMS.shuffle, random_state=cfg.SEED)

    fold_iou, fold_iout, out_of_fold_train_predictions, out_of_fold_test_predictions = [], [], [], []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(meta_train[cfg.DEPTH_COLUMN].values.reshape(-1))):
        valid_data_split = meta_train.iloc[valid_idx]

        LOGGER.info('Started fold {}'.format(fold_id))
        iou, iout, out_of_fold_prediction, test_prediction = _fold_evaluate_predict_loop(valid_data_split,
                                                                                         meta_test,
                                                                                         fold_id,
                                                                                         pipeline_name)

        LOGGER.info('Fold {} IOU {}'.format(fold_id, iou))
        CTX.channel_send('Fold {} IOU'.format(fold_id), 0, iou)
        LOGGER.info('Fold {} IOUT {}'.format(fold_id, iout))
        CTX.channel_send('Fold {} IOUT'.format(fold_id), 0, iout)

        fold_iou.append(iou)
        fold_iout.append(iout)
        out_of_fold_train_predictions.extend(out_of_fold_prediction)
        out_of_fold_test_predictions.append(test_prediction)

    iou_mean, iou_std = np.mean(fold_iou), np.std(fold_iou)
    iout_mean, iout_std = np.mean(fold_iout), np.std(fold_iout)

    _log_scores(iou_mean, iou_std, iout_mean, iout_std)

    _save_predictions(out_of_fold_train_predictions, out_of_fold_test_predictions, meta_train, meta_test,
                      pipeline_name, submit_predictions)


def _make_submission(submission_filepath):
    LOGGER.info('Making Kaggle submit...')
    os.system('kaggle competitions submit -c tgs-salt-identification-challenge -f {} -m {}'.format(submission_filepath,
                                                                                                   PARAMS.kaggle_message))
    LOGGER.info('Kaggle submit completed')


def _add_fold_id_suffix(config, pipeline_name, fold_id):
    model_name = pipeline_name.split('_')[0]
    config['model'][model_name]['callbacks_config']['neptune_monitor']['model_name'] = '{}_{}'.format(model_name,
                                                                                                      fold_id)
    checkpoint_filepath = config['model'][model_name]['callbacks_config']['model_checkpoint']['filepath']
    fold_checkpoint_filepath = checkpoint_filepath.replace('{}/best.torch'.format(model_name),
                                                           '{}_{}/best.torch'.format(model_name,
                                                                                     fold_id))
    config['model'][model_name]['callbacks_config']['model_checkpoint']['filepath'] = fold_checkpoint_filepath
    return config


def _log_scores(iou_mean, iou_std, iout_mean, iout_std):
    LOGGER.info('IOU mean {}, IOU std {}'.format(iou_mean, iou_std))
    CTX.channel_send('IOU', 0, iou_mean)
    CTX.channel_send('IOU STD', 0, iou_std)

    LOGGER.info('IOUT mean {}, IOUT std {}'.format(iout_mean, iout_std))
    CTX.channel_send('IOUT', 0, iout_mean)
    CTX.channel_send('IOUT STD', 0, iout_std)


def _save_predictions(out_of_fold_train_predictions, out_of_fold_test_predictions, meta_train, meta_test,
                      pipeline_name, submit_predictions):
    averaged_mask_predictions_test = _average_mask_predictions(out_of_fold_test_predictions)
    pipeline_postprocessing = PIPELINES[pipeline_name]['postprocessing'](config=cfg.SOLUTION_CONFIG)
    pipeline_postprocessing.clean_cache()
    test_pipe_masks = {'input_masks': {'mask_prediction': averaged_mask_predictions_test}
                       }
    y_pred_test = pipeline_postprocessing.transform(test_pipe_masks)['binarized_images']

    LOGGER.info('Saving predictions')
    out_of_fold_train_predictions_path = os.path.join(PARAMS.experiment_dir,
                                                      '{}_out_of_fold_train_predictions.pkl'.format(pipeline_name))
    joblib.dump({'ids': meta_train[cfg.ID_COLUMNS[0]].tolist(),
                 'images': out_of_fold_train_predictions}, out_of_fold_train_predictions_path)

    out_of_fold_test_predictions_path = os.path.join(PARAMS.experiment_dir,
                                                     '{}_out_of_fold_test_predictions.pkl'.format(pipeline_name))
    joblib.dump({'ids': meta_test[cfg.ID_COLUMNS[0]].tolist(),
                 'images': averaged_mask_predictions_test}, out_of_fold_test_predictions_path)

    submission = create_submission(meta_test, y_pred_test)
    submission_filepath = os.path.join(PARAMS.experiment_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    LOGGER.info('submission saved to {}'.format(submission_filepath))
    LOGGER.info('submission head \n\n{}'.format(submission.head()))

    if submit_predictions:
        _make_submission(submission_filepath)


def _fold_fit_evaluate_predict_loop(train_data_split, valid_data_split, test, fold_id, pipeline_name):
    iou, iout, predicted_masks_valid = _fold_fit_evaluate_loop(train_data_split, valid_data_split, fold_id,
                                                               pipeline_name)

    test_pipe_input = {'input': {'meta': test
                                 },
                       'callback_input': {'meta_valid': None
                                          }
                       }
    pipeline_network = PIPELINES[pipeline_name]['network'](config=cfg.SOLUTION_CONFIG,
                                                           suffix='_fold_{}'.format(fold_id), train_mode=False)
    LOGGER.info('Start pipeline transform on test')
    pipeline_network.clean_cache()
    predicted_masks_test = pipeline_network.transform(test_pipe_input)
    clean_object_from_memory(pipeline_network)

    predicted_masks_test = predicted_masks_test['mask_prediction']
    return iou, iout, predicted_masks_valid, predicted_masks_test


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
    LOGGER.info('Start pipeline fit and transform on train')

    config = _add_fold_id_suffix(cfg.SOLUTION_CONFIG, pipeline_name, fold_id)

    pipeline_network = PIPELINES[pipeline_name]['network'](config=config,
                                                           suffix='_fold_{}'.format(fold_id), train_mode=True)
    pipeline_network.clean_cache()
    pipeline_network.fit_transform(train_pipe_input)
    clean_object_from_memory(pipeline_network)

    LOGGER.info('Start pipeline transform on valid')
    pipeline_network = PIPELINES[pipeline_name]['network'](config=config,
                                                           suffix='_fold_{}'.format(fold_id), train_mode=False)
    pipeline_postprocessing = PIPELINES[pipeline_name]['postprocessing'](config=config,
                                                                         suffix='_fold_{}'.format(fold_id))
    pipeline_network.clean_cache()
    pipeline_postprocessing.clean_cache()
    predicted_masks_valid = pipeline_network.transform(valid_pipe_input)
    valid_pipe_masks = {'input_masks': predicted_masks_valid
                        }
    output_valid = pipeline_postprocessing.transform(valid_pipe_masks)
    clean_object_from_memory(pipeline_network)

    y_pred_valid = output_valid['binarized_images']
    y_true_valid = read_masks(valid_data_split[cfg.Y_COLUMNS[0]].values)

    iou = intersection_over_union(y_true_valid, y_pred_valid)
    iout = intersection_over_union_thresholds(y_true_valid, y_pred_valid)

    predicted_masks_valid = predicted_masks_valid['mask_prediction']
    return iou, iout, predicted_masks_valid


def _fold_evaluate_predict_loop(valid_data_split, test, fold_id, pipeline_name):
    iou, iout, predicted_masks_valid = _fold_evaluate_loop(valid_data_split, fold_id, pipeline_name)

    test_pipe_input = {'input': {'meta': test
                                 },
                       'callback_input': {'meta_valid': None
                                          }
                       }
    pipeline_network = PIPELINES[pipeline_name]['network'](config=cfg.SOLUTION_CONFIG,
                                                           suffix='_fold_{}'.format(fold_id), train_mode=False)
    LOGGER.info('Start pipeline transform on test')
    pipeline_network.clean_cache()
    predicted_masks_test = pipeline_network.transform(test_pipe_input)
    clean_object_from_memory(pipeline_network)

    predicted_masks_test = predicted_masks_test['mask_prediction']
    return iou, iout, predicted_masks_valid, predicted_masks_test


def _fold_evaluate_loop(valid_data_split, fold_id, pipeline_name):
    valid_pipe_input = {'input': {'meta': valid_data_split
                                  },
                        'callback_input': {'meta_valid': None
                                           }
                        }
    LOGGER.info('Start pipeline transform on valid')
    pipeline_network = PIPELINES[pipeline_name]['network'](config=cfg.SOLUTION_CONFIG,
                                                           suffix='_fold_{}'.format(fold_id), train_mode=False)
    pipeline_postprocessing = PIPELINES[pipeline_name]['postprocessing'](config=cfg.SOLUTION_CONFIG,
                                                                         suffix='_fold_{}'.format(fold_id))
    pipeline_network.clean_cache()
    pipeline_postprocessing.clean_cache()
    predicted_masks_valid = pipeline_network.transform(valid_pipe_input)
    valid_pipe_masks = {'input_masks': predicted_masks_valid
                        }
    output_valid = pipeline_postprocessing.transform(valid_pipe_masks)
    clean_object_from_memory(pipeline_network)

    y_pred_valid = output_valid['binarized_images']
    y_true_valid = read_masks(valid_data_split[cfg.Y_COLUMNS[0]].values)

    iou = intersection_over_union(y_true_valid, y_pred_valid)
    iout = intersection_over_union_thresholds(y_true_valid, y_pred_valid)

    predicted_masks_valid = predicted_masks_valid['mask_prediction']
    return iou, iout, predicted_masks_valid


def _average_mask_predictions(out_of_fold_mask_predictions):
    return np.mean(np.array(out_of_fold_mask_predictions), axis=0)
