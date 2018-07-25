from functools import partial

from steppy.base import Step, IdentityOperation
from steppy.adapter import Adapter, E

from . import loaders
from .models import PyTorchUNet
from .utils import make_apply_transformer
from .postprocessing import crop_image, resize_image, binary_label, binarize


def unet(config, train_mode):
    if train_mode:
        preprocessing = preprocessing_train(config, model_name='unet')
    else:
        preprocessing = preprocessing_inference(config)

    unet = Step(name='unet',
                transformer=PyTorchUNet(**config.model['unet']),
                input_data=['callback_input'],
                input_steps=[preprocessing],
                adapter=Adapter({'datagen': E(preprocessing.name, 'datagen'),
                                 'validation_datagen': E(preprocessing.name, 'validation_datagen'),
                                 'meta_valid': E('callback_input', 'meta_valid'),
                                 }),
                is_trainable=True,
                experiment_directory=config.env.experiment_dir)

    if train_mode:
        return unet

    mask_postprocessed = mask_postprocessing(unet, config)

    output = Step(name='output',
                  transformer=IdentityOperation(),
                  input_steps=[mask_postprocessed],
                  adapter=Adapter({'y_pred': E(mask_postprocessed.name, 'labeled_images'),
                                   }),
                  experiment_directory=config.env.experiment_dir)
    return output


def unet_tta(config):
    preprocessing, tta_generator = preprocessing_inference_tta(config, model_name='unet')

    unet = Step(name='unet',
                transformer=PyTorchUNet(**config.model['unet']),
                input_data=['callback_input'],
                input_steps=[preprocessing],
                is_trainable=True,
                experiment_directory=config.env.experiment_dir)

    tta_aggregator = aggregator('tta_aggregator', unet,
                                tta_generator=tta_generator,
                                experiment_directory=config.env.experiment_dir,
                                config=config.tta_aggregator)

    prediction_renamed = Step(name='prediction_renamed',
                              transformer=IdentityOperation(),
                              input_steps=[tta_aggregator],
                              adapter=Adapter({'mask_prediction': E(tta_aggregator.name, 'aggregated_prediction')
                                               }),
                              experiment_directory=config.env.experiment_dir)

    mask_postprocessed = mask_postprocessing(prediction_renamed, config)

    output = Step(name='output',
                  transformer=IdentityOperation(),
                  input_steps=[mask_postprocessed],
                  adapter=Adapter({'y_pred': E(mask_postprocessed.name, 'labeled_images')}),
                  experiment_directory=config.env.experiment_dir)

    return output


def preprocessing_train(config, model_name='unet'):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.ImageSegmentationLoaderCropPad
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResize
    else:
        raise NotImplementedError

    if config.loader.dataset_params.image_source == 'memory':
        reader_train = Step(name='reader_train',
                            transformer=loaders.ImageReader(train_mode=True, **config.reader[model_name]),
                            input_data=['input'],
                            adapter=Adapter({'meta': E('input', 'meta')}),
                            experiment_directory=config.env.experiment_dir)

        reader_inference = Step(name='reader_inference',
                                transformer=loaders.ImageReader(train_mode=True, **config.reader[model_name]),
                                input_data=['callback_input'],
                                adapter=Adapter({'meta': E('callback_input', 'meta_valid')}),
                                experiment_directory=config.env.experiment_dir)

    elif config.loader.dataset_params.image_source == 'disk':
        reader_train = Step(name='xy_train',
                            transformer=loaders.XYSplit(train_mode=True, **config.xy_splitter[model_name]),
                            input_data=['input'],
                            adapter=Adapter({'meta': E('input', 'meta')}),
                            experiment_directory=config.env.experiment_dir)

        reader_inference = Step(name='xy_inference',
                                transformer=loaders.XYSplit(train_mode=True, **config.xy_splitter[model_name]),
                                input_data=['callback_input'],
                                adapter=Adapter({'meta': E('callback_input', 'meta_valid')}),
                                experiment_directory=config.env.experiment_dir)
    else:
        raise NotImplementedError

    loader = Step(name='loader',
                  transformer=Loader(train_mode=True, **config.loader),
                  input_steps=[reader_train, reader_inference],
                  adapter=Adapter({'X': E(reader_train.name, 'X'),
                                   'y': E(reader_train.name, 'y'),
                                   'X_valid': E(reader_inference.name, 'X'),
                                   'y_valid': E(reader_inference.name, 'y'),
                                   }),
                  experiment_directory=config.env.experiment_dir)
    return loader


def preprocessing_inference(config, model_name='unet'):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.ImageSegmentationLoaderCropPad
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResize
    else:
        raise NotImplementedError

    if config.loader.dataset_params.image_source == 'memory':
        reader_inference = Step(name='reader_inference',
                                transformer=loaders.ImageReader(train_mode=False, **config.reader[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),

                                experiment_directory=config.env.experiment_dir)

    elif config.loader.dataset_params.image_source == 'disk':
        reader_inference = Step(name='xy_inference',
                                transformer=loaders.XYSplit(train_mode=False, **config.xy_splitter[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),
                                experiment_directory=config.env.experiment_dir)
    else:
        raise NotImplementedError

    loader = Step(name='loader',
                  transformer=Loader(train_mode=False, **config.loader),
                  input_steps=[reader_inference],
                  adapter=Adapter({'X': E(reader_inference.name, 'X'),
                                   'y': E(reader_inference.name, 'y'),
                                   }),
                  experiment_directory=config.env.experiment_dir,
                  cache_output=True)
    return loader


def preprocessing_inference_tta(config, model_name='unet'):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.ImageSegmentationLoaderCropPadTTA
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResizeTTA
    else:
        raise NotImplementedError

    if config.loader.dataset_params.image_source == 'memory':
        reader_inference = Step(name='reader_inference',
                                transformer=loaders.ImageReader(train_mode=False, **config.reader[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),
                                experiment_directory=config.env.experiment_dir)

        tta_generator = Step(name='tta_generator',
                             transformer=loaders.TestTimeAugmentationGenerator(**config.tta_generator),
                             input_steps=[reader_inference],
                             adapter=Adapter({'X': E('reader_inference', 'X')}),
                             experiment_directory=config.env.experiment_dir)

    elif config.loader.dataset_params.image_source == 'disk':
        reader_inference = Step(name='reader_inference',
                                transformer=loaders.XYSplit(train_mode=False, **config.xy_splitter[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),
                                experiment_directory=config.env.experiment_dir)

        tta_generator = Step(name='tta_generator',
                             transformer=loaders.MetaTestTimeAugmentationGenerator(**config.tta_generator),
                             input_steps=[reader_inference],
                             adapter=Adapter({'X': E('reader_inference', 'X')}),
                             experiment_directory=config.env.experiment_dir)
    else:
        raise NotImplementedError

    loader = Step(name='loader',
                  transformer=Loader(**config.loader),
                  input_steps=[tta_generator],
                  adapter=Adapter({'X': E(tta_generator.name, 'X_tta'),
                                   'tta_params': E(tta_generator.name, 'tta_params'),
                                   }),
                  experiment_directory=config.env.experiment_dir,
                  cache_output=True)
    return loader, tta_generator


def aggregator(name, model, tta_generator, experiment_directory, config):
    tta_aggregator = Step(name=name,
                          transformer=loaders.TestTimeAugmentationAggregator(**config),
                          input_steps=[model, tta_generator],
                          adapter=Adapter({'images': E(model.name, 'mask_prediction'),
                                           'tta_params': E(tta_generator.name, 'tta_params'),
                                           'img_ids': E(tta_generator.name, 'img_ids'),
                                           }),
                          experiment_directory=experiment_directory)
    return tta_aggregator


def mask_postprocessing(model, config):
    if config.execution.loader_mode == 'crop_and_pad':
        size_adjustment_function = partial(crop_image,
                                           target_size=(config.loader.dataset_params.h,
                                                        config.loader.dataset_params.w))
    elif config.execution.loader_mode == 'resize':
        size_adjustment_function = partial(resize_image,
                                           target_size=(config.loader.dataset_params.h,
                                                        config.loader.dataset_params.w))
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize',
                       transformer=make_apply_transformer(size_adjustment_function,
                                                          output_name='resized_images',
                                                          apply_on=['images']),
                       input_steps=[model],
                       adapter=Adapter({'images': E(model.name, 'mask_prediction'),
                                        }),
                       experiment_directory=config.env.experiment_dir)

    binarizer = Step(name='binarizer',
                     transformer=make_apply_transformer(partial(binarize, threshold=config.thresholder.threshold_masks),
                                                        output_name='binarized_images',
                                                        apply_on=['images']),
                     input_steps=[mask_resize],
                     adapter=Adapter({'images': E(mask_resize.name, 'resized_images'),
                                      }),
                     experiment_directory=config.env.experiment_dir)

    labeler = Step(name='labeler',
                   transformer=make_apply_transformer(binary_label,
                                                      output_name='labeled_images',
                                                      apply_on=['images']),
                   input_steps=[binarizer],
                   adapter=Adapter({'images': E(binarizer.name, 'binarized_images'),
                                    }),
                   experiment_directory=config.env.experiment_dir)

    return labeler


PIPELINES = {'unet': {'train': partial(unet, train_mode=True),
                      'inference': partial(unet, train_mode=False),
                      },
             'unet_tta': {'inference': unet_tta,
                          },
             }
