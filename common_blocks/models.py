import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from toolkit.pytorch_transformers.models import Model
from torch.autograd import Variable
from torch.nn import functional as F

from .architectures import unet, large_kernel_matters, misc, models_with_depth, pspnet
from . import callbacks as cbk
from .lovasz_losses import lovasz_hinge
from .utils import sigmoid, softmax, get_list_of_image_predictions, pytorch_where

ARCHITECTURES = {'UNetResNet': {'model': unet.UNetResNet,
                                'model_config': {'encoder_depth': 34, 'use_hypercolumn': True,
                                                 'dropout_2d': 0.0, 'pretrained': True, 'pool0': False
                                                 },
                                'init_weights': False},
                 'UNetSeResNet': {'model': unet.UNetSeResNet,
                                  'model_config': {'encoder_depth': 50, 'use_hypercolumn': True,
                                                   'dropout_2d': 0.0, 'pretrained': 'imagenet', 'pool0': False
                                                   },
                                  'init_weights': False},
                 'UNetSeResNetXt': {'model': unet.UNetSeResNetXt,
                                    'model_config': {'encoder_depth': 50, 'use_hypercolumn': True,
                                                     'dropout_2d': 0.0, 'pretrained': 'imagenet', 'pool0': False
                                                     },
                                    'init_weights': False},
                 'UNetDenseNet': {'model': unet.UNetDenseNet,
                                  'model_config': {'encoder_depth': 121, 'use_hypercolumn': True,
                                                   'dropout_2d': 0.0, 'pretrained': 'imagenet', 'pool0': False
                                                   },
                                  'init_weights': False},
                 'LargeKernelMatters': {'model': large_kernel_matters.LargeKernelMatters,
                                        'model_config': {'encoder_depth': 34, 'pretrained': True,
                                                         'kernel_size': 9, 'internal_channels': 21,
                                                         'dropout_2d': 0.0, 'use_relu': True, 'pool0': False
                                                         },
                                        'init_weights': False},
                 'PSPNet': {'model': pspnet.PSPNet,
                            'model_config': {'encoder_depth': 34, 'pretrained': True,
                                             'use_hypercolumn': True, 'pool0': False
                                             },
                            'init_weights': False},
                 'UNetResNetWithDepth': {'model': models_with_depth.UNetResNetWithDepth,
                                         'model_config': {'encoder_depth': 34, 'use_hypercolumn': True,
                                                          'dropout_2d': 0.0, 'pretrained': True, 'pool0': False
                                                          },
                                         'init_weights': False},
                 'StackingFCN': {'model': misc.StackingFCN,
                                 'model_config': {'input_model_nr': 32, 'filter_nr': 32, 'dropout_2d': 0.0
                                                  },
                                 'init_weights': True},
                 'StackingFCNWithDepth': {'model': misc.StackingFCNWithDepth,
                                          'model_config': {'input_model_nr': 32, 'filter_nr': 32, 'dropout_2d': 0.0
                                                           },
                                          'init_weights': True},
                 'EmptinessClassifier': {'model': misc.EmptinessClassifier,
                                         'model_config': {'encoder_depth': 18, 'pretrained': True,
                                                          },
                                         'init_weights': False},

                 }


class SegmentationModel(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.activation_func = self.architecture_config['model_params']['activation']
        self.set_model()
        self.set_loss()
        self.weight_regularization = weight_regularization
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.callbacks = callbacks_network(self.callbacks_config)

    def fit(self, datagen, validation_datagen=None, meta_valid=None):
        self._initialize_model_weights()

        if not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.callbacks.set_params(self, validation_datagen=validation_datagen, meta_valid=meta_valid)
        self.callbacks.on_train_begin()

        batch_gen, steps = datagen
        for epoch_id in range(self.training_config['epochs']):
            self.callbacks.on_epoch_begin()
            for batch_id, data in enumerate(batch_gen):
                self.callbacks.on_batch_begin()
                metrics = self._fit_loop(data)
                self.callbacks.on_batch_end(metrics=metrics)
                if batch_id == steps:
                    break
            self.callbacks.on_epoch_end()
            if self.callbacks.training_break():
                break
        self.callbacks.on_train_end()
        return self

    def _fit_loop(self, data):
        X = data[0]
        targets_tensors = data[1:]

        if torch.cuda.is_available():
            X = Variable(X).cuda()
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor).cuda())
        else:
            X = Variable(X)
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor))

        self.optimizer.zero_grad()
        outputs_batch = self.model(X)
        partial_batch_losses = {}

        if len(self.output_names) == 1:
            for (name, loss_function, weight), target in zip(self.loss_function, targets_var):
                batch_loss = loss_function(outputs_batch, target) * weight
        else:
            for (name, loss_function, weight), output, target in zip(self.loss_function, outputs_batch, targets_var):
                partial_batch_losses[name] = loss_function(output, target) * weight
            batch_loss = sum(partial_batch_losses.values())
        partial_batch_losses['sum'] = batch_loss

        batch_loss.backward()
        self.optimizer.step()

        return partial_batch_losses

    def transform(self, datagen, validation_datagen=None, *args, **kwargs):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            if self.activation_func == 'softmax':
                outputs[name] = [softmax(single_prediction, axis=0) for single_prediction in prediction]
            elif self.activation_func == 'sigmoid':
                outputs[name] = [sigmoid(np.squeeze(mask)) for mask in prediction]
            else:
                raise Exception('Only softmax and sigmoid activations are allowed')
        return outputs

    def _transform(self, datagen, validation_datagen=None, **kwargs):
        self.model.eval()

        batch_gen, steps = datagen
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            if isinstance(data, list):
                X = data[0]
            else:
                X = data

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
            else:
                X = Variable(X, volatile=True)
            outputs_batch = self.model(X)

            if len(self.output_names) == 1:
                outputs.setdefault(self.output_names[0], []).append(outputs_batch.data.cpu().numpy())
            else:
                for name, output in zip(self.output_names, outputs_batch):
                    output_ = output.data.cpu().numpy()
                    outputs.setdefault(name, []).append(output_)
            if batch_id == steps:
                break
        self.model.train()
        outputs = {'{}_prediction'.format(name): get_list_of_image_predictions(outputs_) for name, outputs_ in
                   outputs.items()}
        return outputs

    def set_model(self):
        architecture = self.architecture_config['model_params']['architecture']
        config = ARCHITECTURES[architecture]
        self.model = config['model'](num_classes=self.architecture_config['model_params']['out_channels'],
                                     **config['model_config'])
        self._initialize_model_weights = lambda: None

    def set_loss(self):
        if self.activation_func == 'softmax':
            raise NotImplementedError('No softmax loss defined')
        elif self.activation_func == 'sigmoid':
            loss_function = lovasz_loss
            # loss_function = nn.BCEWithLogitsLoss()
        else:
            raise Exception('Only softmax and sigmoid activations are allowed')
        self.loss_function = [('mask', loss_function, 1.0)]

    def load(self, filepath):
        self.model.eval()

        if not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available():
            self.model.cpu()
            self.model.load_state_dict(torch.load(filepath))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        return self


class SegmentationModelWithDepth(SegmentationModel):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.activation_func = self.architecture_config['model_params']['activation']
        self.set_model()
        self.set_loss()
        self.weight_regularization = weight_regularization
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.callbacks = callbacks_network(self.callbacks_config)

    def _fit_loop(self, data):
        X = data[0]
        D = data[1]
        targets_tensors = data[2:]

        if torch.cuda.is_available():
            X = Variable(X).cuda()
            D = Variable(D).cuda()
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor).cuda())
        else:
            X = Variable(X)
            D = Variable(D)
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor))

        self.optimizer.zero_grad()
        outputs_batch = self.model(X, D)
        partial_batch_losses = {}

        if len(self.output_names) == 1:
            for (name, loss_function, weight), target in zip(self.loss_function, targets_var):
                batch_loss = loss_function(outputs_batch, target) * weight
        else:
            for (name, loss_function, weight), output, target in zip(self.loss_function, outputs_batch, targets_var):
                partial_batch_losses[name] = loss_function(output, target) * weight
            batch_loss = sum(partial_batch_losses.values())
        partial_batch_losses['sum'] = batch_loss

        batch_loss.backward()
        self.optimizer.step()

        return partial_batch_losses

    def _transform(self, datagen, validation_datagen=None, **kwargs):
        self.model.eval()

        batch_gen, steps = datagen
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            X = data[0]
            D = data[1]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
                D = Variable(D, volatile=True).cuda()
            else:
                X = Variable(X, volatile=True)
                D = Variable(D, volatile=True)
            outputs_batch = self.model(X, D)

            if len(self.output_names) == 1:
                outputs.setdefault(self.output_names[0], []).append(outputs_batch.data.cpu().numpy())
            else:
                for name, output in zip(self.output_names, outputs_batch):
                    output_ = output.data.cpu().numpy()
                    outputs.setdefault(name, []).append(output_)
            if batch_id == steps:
                break
        self.model.train()
        outputs = {'{}_prediction'.format(name): get_list_of_image_predictions(outputs_) for name, outputs_ in
                   outputs.items()}
        return outputs


def weight_regularization(model, regularize, weight_decay_conv2d):
    if regularize:
        parameter_list = [
            {'params': filter(lambda p: p.requires_grad, model.parameters()),
             'weight_decay': weight_decay_conv2d},
        ]
    else:
        parameter_list = [filter(lambda p: p.requires_grad, model.parameters())]
    return parameter_list


def callbacks_network(callbacks_config):
    experiment_timing = cbk.ExperimentTiming(**callbacks_config['experiment_timing'])
    model_checkpoints = cbk.ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = cbk.ReduceLROnPlateauScheduler(**callbacks_config['reduce_lr_on_plateau_scheduler'])
    training_monitor = cbk.TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = cbk.ValidationMonitor(**callbacks_config['validation_monitor'])
    # validation_monitor = cbk.ValidationMonitorEmptiness(**callbacks_config['validation_monitor'])
    neptune_monitor = cbk.NeptuneMonitor(**callbacks_config['neptune_monitor'])
    early_stopping = cbk.EarlyStopping(**callbacks_config['early_stopping'])

    return cbk.CallbackList(
        callbacks=[experiment_timing, training_monitor, validation_monitor,
                   model_checkpoints, lr_scheduler, neptune_monitor, early_stopping])


class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (
                torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


def lovasz_loss(output, target):
    target = target.long()
    return lovasz_hinge(output, target)


def mixed_dice_bce_loss(output, target, dice_weight=0.2, dice_loss=None,
                        bce_weight=0.9, bce_loss=None,
                        smooth=0, dice_activation='sigmoid'):
    num_classes = output.size(1)
    target = target[:, :num_classes, :, :].long()
    if bce_loss is None:
        bce_loss = nn.BCEWithLogitsLoss()
    if dice_loss is None:
        dice_loss = multiclass_dice_loss
    return dice_weight * dice_loss(output, target, smooth, dice_activation) + bce_weight * bce_loss(output, target)


def mixed_dice_cross_entropy_loss(output, target, dice_weight=0.5, dice_loss=None,
                                  cross_entropy_weight=0.5, cross_entropy_loss=None, smooth=0,
                                  dice_activation='softmax'):
    num_classes_without_background = output.size(1) - 1
    dice_output = output[:, 1:, :, :]
    dice_target = target[:, :num_classes_without_background, :, :].long()
    cross_entropy_target = torch.zeros_like(target[:, 0, :, :]).long()
    for class_nr in range(num_classes_without_background):
        cross_entropy_target = where(target[:, class_nr, :, :], class_nr + 1, cross_entropy_target)
    if cross_entropy_loss is None:
        cross_entropy_loss = nn.CrossEntropyLoss()
    if dice_loss is None:
        dice_loss = multiclass_dice_loss
    return dice_weight * dice_loss(dice_output, dice_target, smooth,
                                   dice_activation) + cross_entropy_weight * cross_entropy_loss(output,
                                                                                                cross_entropy_target)


def multiclass_dice_loss(output, target, smooth=0, activation='softmax'):
    """Calculate Dice Loss for multiple class output.

    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x H x W).
        smooth (float, optional): Smoothing factor. Defaults to 0.
        activation (string, optional): Name of the activation function, softmax or sigmoid. Defaults to 'softmax'.

    Returns:
        torch.Tensor: Loss value.

    """
    if activation == 'softmax':
        activation_nn = torch.nn.Softmax2d()
    elif activation == 'sigmoid':
        activation_nn = torch.nn.Sigmoid()
    else:
        raise NotImplementedError('only sigmoid and softmax are implemented')

    loss = 0
    dice = DiceLoss(smooth=smooth)
    output = activation_nn(output)
    num_classes = output.size(1)
    target.data = target.data.float()
    for class_nr in range(num_classes):
        loss += dice(output[:, class_nr, :, :], target[:, class_nr, :, :])
    return loss / num_classes


def where(cond, x_1, x_2):
    cond = cond.long()
    return (cond * x_1) + ((1 - cond) * x_2)
