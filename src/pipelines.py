from functools import partial

from steppy.base import Step, IdentityOperation
from steppy.adapter import Adapter, E

from . import loaders
from .models import PyTorchUNet
from .utils import make_apply_transformer
from .postprocessing import crop_image, resize_image, binarize
from .pipeline_config import ORIGINAL_SIZE


def unet(config, train_mode, suffix=''):
    if train_mode:
        preprocessing = preprocessing_train(config, model_name='unet', suffix=suffix)
    else:
        preprocessing = preprocessing_inference(config, suffix=suffix)

    unet = Step(name='unet{}'.format(suffix),
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

    mask_postprocessed = mask_postprocessing(unet, config, suffix)

    output = Step(name='output{}'.format(suffix),
                  transformer=IdentityOperation(),
                  input_steps=[mask_postprocessed],
                  adapter=Adapter({'y_pred': E(mask_postprocessed.name, 'binarized_images'),
                                   }),
                  experiment_directory=config.env.experiment_dir)
    return output


def unet_tta(config, train_mode=False, suffix=''):
    preprocessing, tta_generator = preprocessing_inference_tta(config, model_name='unet')

    unet = Step(name='unet{}'.format(suffix),
                transformer=PyTorchUNet(**config.model['unet']),
                input_data=['callback_input'],
                input_steps=[preprocessing],
                is_trainable=True,
                experiment_directory=config.env.experiment_dir)

    tta_aggregator = aggregator('tta_aggregator{}'.format(suffix), unet,
                                tta_generator=tta_generator,
                                experiment_directory=config.env.experiment_dir,
                                config=config.tta_aggregator)

    prediction_renamed = Step(name='prediction_renamed{}'.format(suffix),
                              transformer=IdentityOperation(),
                              input_steps=[tta_aggregator],
                              adapter=Adapter({'mask_prediction': E(tta_aggregator.name, 'aggregated_prediction')
                                               }),
                              experiment_directory=config.env.experiment_dir)

    mask_postprocessed = mask_postprocessing(prediction_renamed, config, suffix)

    output = Step(name='output{}'.format(suffix),
                  transformer=IdentityOperation(),
                  input_steps=[mask_postprocessed],
                  adapter=Adapter({'y_pred': E(mask_postprocessed.name, 'binarized_images')}),
                  experiment_directory=config.env.experiment_dir)

    return output


def preprocessing_train(config, model_name='unet', suffix=''):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.ImageSegmentationLoaderCropPad
        loader_config = config.loaders.crop_and_pad
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResize
        loader_config = config.loaders.resize
    else:
        raise NotImplementedError

    if loader_config.dataset_params.image_source == 'memory':
        reader_train = Step(name='reader_train{}'.format(suffix),
                            transformer=loaders.ImageReader(train_mode=True, **config.reader[model_name]),
                            input_data=['input'],
                            adapter=Adapter({'meta': E('input', 'meta')}),
                            experiment_directory=config.env.experiment_dir)

        reader_inference = Step(name='reader_inference{}'.format(suffix),
                                transformer=loaders.ImageReader(train_mode=True, **config.reader[model_name]),
                                input_data=['callback_input'],
                                adapter=Adapter({'meta': E('callback_input', 'meta_valid')}),
                                experiment_directory=config.env.experiment_dir)

    elif loader_config.dataset_params.image_source == 'disk':
        reader_train = Step(name='xy_train{}'.format(suffix),
                            transformer=loaders.XYSplit(train_mode=True, **config.xy_splitter[model_name]),
                            input_data=['input'],
                            adapter=Adapter({'meta': E('input', 'meta')}),
                            experiment_directory=config.env.experiment_dir)

        reader_inference = Step(name='xy_inference{}'.format(suffix),
                                transformer=loaders.XYSplit(train_mode=True, **config.xy_splitter[model_name]),
                                input_data=['callback_input'],
                                adapter=Adapter({'meta': E('callback_input', 'meta_valid')}),
                                experiment_directory=config.env.experiment_dir)
    else:
        raise NotImplementedError

    loader = Step(name='loader{}'.format(suffix),
                  transformer=Loader(train_mode=True, **loader_config),
                  input_steps=[reader_train, reader_inference],
                  adapter=Adapter({'X': E(reader_train.name, 'X'),
                                   'y': E(reader_train.name, 'y'),
                                   'X_valid': E(reader_inference.name, 'X'),
                                   'y_valid': E(reader_inference.name, 'y'),
                                   }),
                  experiment_directory=config.env.experiment_dir)
    return loader


def preprocessing_inference(config, model_name='unet', suffix=''):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.ImageSegmentationLoaderCropPad
        loader_config = config.loaders.crop_and_pad
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResize
        loader_config = config.loaders.crop_and_pad
    else:
        raise NotImplementedError

    if loader_config.dataset_params.image_source == 'memory':
        reader_inference = Step(name='reader_inference{}'.format(suffix),
                                transformer=loaders.ImageReader(train_mode=False, **config.reader[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),

                                experiment_directory=config.env.experiment_dir)

    elif loader_config.dataset_params.image_source == 'disk':
        reader_inference = Step(name='xy_inference{}'.format(suffix),
                                transformer=loaders.XYSplit(train_mode=False, **config.xy_splitter[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),
                                experiment_directory=config.env.experiment_dir)
    else:
        raise NotImplementedError

    loader = Step(name='loader{}'.format(suffix),
                  transformer=Loader(train_mode=False, **loader_config),
                  input_steps=[reader_inference],
                  adapter=Adapter({'X': E(reader_inference.name, 'X'),
                                   'y': E(reader_inference.name, 'y'),
                                   }),
                  experiment_directory=config.env.experiment_dir,
                  cache_output=True)
    return loader


def preprocessing_inference_tta(config, model_name='unet', suffix=''):
    if config.loader.dataset_params.image_source == 'memory':
        reader_inference = Step(name='reader_inference{}'.format(suffix),
                                transformer=loaders.ImageReader(train_mode=False, **config.reader[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),
                                experiment_directory=config.env.experiment_dir)

        tta_generator = Step(name='tta_generator{}'.format(suffix),
                             transformer=loaders.TestTimeAugmentationGenerator(**config.tta_generator),
                             input_steps=[reader_inference],
                             adapter=Adapter({'X': E('reader_inference', 'X')}),
                             experiment_directory=config.env.experiment_dir)

    elif config.loader.dataset_params.image_source == 'disk':
        reader_inference = Step(name='reader_inference{}'.format(suffix),
                                transformer=loaders.XYSplit(train_mode=False, **config.xy_splitter[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),
                                experiment_directory=config.env.experiment_dir)

        tta_generator = Step(name='tta_generator{}'.format(suffix),
                             transformer=loaders.MetaTestTimeAugmentationGenerator(**config.tta_generator),
                             input_steps=[reader_inference],
                             adapter=Adapter({'X': E('reader_inference', 'X')}),
                             experiment_directory=config.env.experiment_dir)
    else:
        raise NotImplementedError

    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.ImageSegmentationLoaderCropPadTTA
        loader_config = config.loader.crop_and_pad_tta
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResizeTTA
        loader_config = config.loader.resize_tta
    else:
        raise NotImplementedError

    loader = Step(name='loader{}'.format(suffix),
                  transformer=Loader(**loader_config),
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


def mask_postprocessing(model, config, suffix=''):
    if config.execution.loader_mode == 'crop_and_pad':
        size_adjustment_function = partial(crop_image, target_size=ORIGINAL_SIZE)
    elif config.execution.loader_mode == 'resize':
        size_adjustment_function = partial(resize_image, target_size=ORIGINAL_SIZE)
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize{}'.format(suffix),
                       transformer=make_apply_transformer(size_adjustment_function,
                                                          output_name='resized_images',
                                                          apply_on=['images']),
                       input_steps=[model],
                       adapter=Adapter({'images': E(model.name, 'mask_prediction'),
                                        }),
                       experiment_directory=config.env.experiment_dir)

    binarizer = Step(name='binarizer{}'.format(suffix),
                     transformer=make_apply_transformer(partial(binarize, threshold=config.thresholder.threshold_masks),
                                                        output_name='binarized_images',
                                                        apply_on=['images']),
                     input_steps=[mask_resize],
                     adapter=Adapter({'images': E(mask_resize.name, 'resized_images'),
                                      }),
                     experiment_directory=config.env.experiment_dir)

    return binarizer


PIPELINES = {'unet': unet,
             'unet_tta': unet_tta
             }
