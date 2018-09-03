import os

from attrdict import AttrDict
import neptune

from .utils import read_params
from .augmentation import intensity_seq, affine_seq, pad_to_fit_net, resize_pad_seq, test_time_augmentation_transform, \
    test_time_augmentation_inverse_transform

CTX = neptune.Context()
PARAMS = read_params(CTX)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SEED = 1234

ID_COLUMNS = ['id']
X_COLUMNS = ['file_path_image']
Y_COLUMNS = ['file_path_mask']
DEPTH_COLUMN = ['z']
ORIGINAL_SIZE = (101, 101)

GLOBAL_CONFIG = {'exp_root': PARAMS.experiment_dir,
                 'num_workers': PARAMS.num_workers,
                 'num_classes': 2,
                 'img_H-W': (PARAMS.image_h, PARAMS.image_w),
                 'batch_size_train': PARAMS.batch_size_train,
                 'batch_size_inference': PARAMS.batch_size_inference,
                 'loader_mode': PARAMS.loader_mode,
                 }

TRAINING_CONFIG = {'epochs': PARAMS.epochs_nr,
                   'shuffle': True,
                   'batch_size': PARAMS.batch_size_train,
                   'fine_tuning': PARAMS.fine_tuning,
                   }

SOLUTION_CONFIG = AttrDict({
    'env': {'experiment_dir': PARAMS.experiment_dir},
    'execution': GLOBAL_CONFIG,
    'xy_splitter': {
        'unet': {'x_columns': X_COLUMNS,
                 'y_columns': Y_COLUMNS,
                 },
    },
    'reader': {
        'unet': {'x_columns': X_COLUMNS,
                 'y_columns': Y_COLUMNS,
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

                                   'augmentation_params': {'image_augment_train': intensity_seq,
                                                           'image_augment_with_target_train': resize_pad_seq(
                                                               resize_target_size=PARAMS.resize_target_size,
                                                               pad_method=PARAMS.pad_method,
                                                               pad_size=PARAMS.pad_size),
                                                           'image_augment_inference': pad_to_fit_net(64,
                                                                                                     PARAMS.pad_method),
                                                           'image_augment_with_target_inference': pad_to_fit_net(64,
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
                                'image_augment_inference': pad_to_fit_net(64, PARAMS.pad_method),
                                'image_augment_with_target_inference': pad_to_fit_net(64,
                                                                                      PARAMS.pad_method),
                                'tta_transform': test_time_augmentation_transform
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

                           'augmentation_params': {'image_augment_train': intensity_seq,
                                                   'image_augment_with_target_train': affine_seq
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

                               'augmentation_params': {'tta_transform': test_time_augmentation_transform
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
            'training_config': TRAINING_CONFIG,
            'callbacks_config': {'model_checkpoint': {
                'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints', 'unet', 'best.torch'),
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
    'tta_aggregator': {'tta_inverse_transform': test_time_augmentation_inverse_transform,
                       'method': PARAMS.tta_aggregation_method,
                       'nthreads': PARAMS.num_threads
                       },
    'thresholder': {'threshold_masks': PARAMS.threshold_masks,
                    },
})
