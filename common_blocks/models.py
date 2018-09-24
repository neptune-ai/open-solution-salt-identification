from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from toolkit.pytorch_transformers.models import Model

from .utils import sigmoid, softmax, get_list_of_image_predictions, pytorch_where
from . import callbacks as cbk
from .architectures import UNetResNet, LargeKernelMatters, UNetResNetWithDepth, StackingFCN, StackingFCNWithDepth, \
    EmptinessClassifier
from .lovasz_losses import lovasz_hinge

ARCHITECTURES = {'UNetResNet': {'model': UNetResNet,
                                'model_config': {'encoder_depth': 34, 'use_hypercolumn': True,
                                                 'dropout_2d': 0.0, 'pretrained': True,
                                                 },
                                'init_weights': False},

                 'UNetResNetWithDepth': {'model': UNetResNetWithDepth,
                                         'model_config': {'encoder_depth': 34, 'use_hypercolumn': True,
                                                          'dropout_2d': 0.0, 'pretrained': True,
                                                          },
                                         'init_weights': False},
                 'LargeKernelMatters': {'model': LargeKernelMatters,
                                        'model_config': {'encoder_depth': 34, 'pretrained': True,
                                                         'kernel_size': 9, 'internal_channels': 21,
                                                         'dropout_2d': 0.0, 'use_relu': True
                                                         },
                                        'init_weights': False},
                 'StackingFCN': {'model': StackingFCN,
                                 'model_config': {'input_model_nr': 18, 'filter_nr': 32, 'dropout_2d': 0.0
                                                  },
                                 'init_weights': True},
                 'StackingFCNWithDepth': {'model': StackingFCNWithDepth,
                                          'model_config': {'input_model_nr': 18, 'filter_nr': 32, 'dropout_2d': 0.0
                                                           },
                                          'init_weights': True},
                 'EmptinessClassifier': {'model': EmptinessClassifier,
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


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        logpt = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        pt = torch.exp(-logpt)
        at = self.alpha * target + (1 - target)
        loss = at * ((1 - pt).pow(self.gamma)) * logpt
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (
                torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


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


def weighted_lovash_focal_loss(output, target):
    focal = weighted_focal_loss(output, target)
    lovasz = lovasz_hinge(output, target)
    return 0.25 * focal + lovasz


def lovasz_loss(output, target):
    target = target.long()
    return lovasz_hinge(output, target)


def weighted_focal_loss(output, target,
                        alpha=1.0, gamma=5.0,
                        max_weight=100.0,
                        focus_threshold=0.1,
                        use_size_weight=True,
                        use_border_weight=True, border_size=24, border_weight=2.0
                        ):
    output = focus_output(output, focus_threshold=focus_threshold)
    loss_per_pixel = FocalWithLogitsLoss(alpha=alpha, gamma=gamma)(output, target)
    weights = get_weights(target,
                          max_weight=max_weight,
                          use_size_weight=use_size_weight,
                          use_border_weight=use_border_weight, border_size=border_size, border_weight=border_weight)
    loss = torch.mean(loss_per_pixel * weights)
    return loss


def focus_output(output, focus_threshold):
    if torch.cuda.is_available():
        output_numpy = F.sigmoid(output).data.cpu().numpy()
    else:
        output_numpy = F.sigmoid(output).data.numpy()
    focus_weights = np.where(output_numpy < focus_threshold, 0.0, 1.0)
    focus_weights = Variable(torch.Tensor(focus_weights), requires_grad=False)
    if torch.cuda.is_available():
        focus_weights = focus_weights.cuda()
    return torch.mul(focus_weights, output)


def get_weights(target, max_weight=5.0,
                use_size_weight=True,
                use_border_weight=True, border_size=10, border_weight=2.0):
    if torch.cuda.is_available():
        target_numpy = target.data.cpu().numpy()
    else:
        target_numpy = target.data.numpy()

    if use_size_weight:
        size_weights = _size_weights(target_numpy)
    else:
        size_weights = np.ones_like(target_numpy)

    if use_border_weight:
        border_weights = _border_weights(target_numpy, border_size=border_size, border_weight=border_weight)
    else:
        border_weights = np.ones_like(target_numpy)

    weights = border_weights * size_weights
    weights = np.where(weights > max_weight, max_weight, weights)
    weights = Variable(torch.Tensor(weights), requires_grad=False)

    if torch.cuda.is_available():
        weights = weights.cuda()
    return weights


def _size_weights(target):
    target_ = target[:, 1, :, :]
    size_per_image = np.mean(target_, axis=(1, 2))
    size_per_image = np.where(size_per_image == 0.0, 1.0, size_per_image)
    size_weights_per_image = 1.0 / size_per_image.reshape(-1, 1, 1, 1)
    size_weights = np.where(target, np.multiply(target, size_weights_per_image), 1.0)
    return size_weights


def _border_weights(target, border_size=10, border_weight=2.0):
    border_mask = border_weight * np.ones_like(target)
    border_mask[:, :, border_size:-border_size, border_size:-border_size] = 1.0
    return border_mask
