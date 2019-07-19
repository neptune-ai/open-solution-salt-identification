from functools import partial
import os
from datetime import datetime, timedelta

import numpy as np
import torch
from PIL import Image
import neptune
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tempfile import TemporaryDirectory

from sklearn.metrics import roc_auc_score
from steppy.base import Step, IdentityOperation
from steppy.adapter import Adapter, E
from toolkit.pytorch_transformers.utils import Averager, persist_torch_model
from toolkit.pytorch_transformers.validation import score_model

from .utils import get_logger, sigmoid, softmax, make_apply_transformer, read_masks, get_list_of_image_predictions
from .metrics import intersection_over_union, intersection_over_union_thresholds
from .postprocessing import crop_image, resize_image, binarize

logger = get_logger()

Y_COLUMN = 'file_path_mask'
ORIGINAL_SIZE = (101, 101)
THRESHOLD = 0.5


class Callback:
    def __init__(self):
        self.epoch_id = None
        self.batch_id = None

        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.output_names = None
        self.validation_datagen = None
        self.lr_scheduler = None

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.output_names = transformer.output_names
        self.validation_datagen = validation_datagen
        self.transformer = transformer

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0

    def on_train_end(self, *args, **kwargs):
        pass

    def on_epoch_begin(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        self.epoch_id += 1

    def training_break(self, *args, **kwargs):
        return False

    def on_batch_begin(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        self.batch_id += 1

    def get_validation_loss(self):
        if self.epoch_id not in self.transformer.validation_loss.keys():
            self.transformer.validation_loss[self.epoch_id] = score_model(self.model, self.loss_function,
                                                                          self.validation_datagen)
        return self.transformer.validation_loss[self.epoch_id]


class CallbackList:
    def __init__(self, callbacks=None):
        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, Callback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = callbacks

    def __len__(self):
        return len(self.callbacks)

    def set_params(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.set_params(*args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(*args, **kwargs)

    def on_epoch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def training_break(self, *args, **kwargs):
        callback_out = [callback.training_break(*args, **kwargs) for callback in self.callbacks]
        return any(callback_out)

    def on_batch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(*args, **kwargs)


class TrainingMonitor(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        self.epoch_loss_averagers = {}
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averagers = {}
        self.epoch_id = 0
        self.batch_id = 0

    def on_epoch_end(self, *args, **kwargs):
        for name, averager in self.epoch_loss_averagers.items():
            epoch_avg_loss = averager.value
            averager.reset()
            if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
                logger.info('epoch {0} {1}:     {2:.5f}'.format(self.epoch_id, name, epoch_avg_loss))
        self.epoch_id += 1

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]
            if name in self.epoch_loss_averagers.keys():
                self.epoch_loss_averagers[name].send(loss)
            else:
                self.epoch_loss_averagers[name] = Averager()
                self.epoch_loss_averagers[name].send(loss)

            if self.batch_every and ((self.batch_id % self.batch_every) == 0):
                logger.info('epoch {0} batch {1} {2}:     {3:.5f}'.format(self.epoch_id, self.batch_id, name, loss))
        self.batch_id += 1


class ExponentialLRScheduler(Callback):
    def __init__(self, gamma, epoch_every=1, batch_every=None):
        super().__init__()
        self.gamma = gamma
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        self.validation_datagen = validation_datagen
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.lr_scheduler = ExponentialLR(self.optimizer, self.gamma, last_epoch=-1)

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        logger.info('initial lr: {0}'.format(self.optimizer.state_dict()['param_groups'][0]['initial_lr']))

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and (((self.epoch_id + 1) % self.epoch_every) == 0):
            self.lr_scheduler.step()
            logger.info('epoch {0} current lr: {1}'.format(self.epoch_id + 1,
                                                           self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.epoch_id += 1

    def on_batch_end(self, *args, **kwargs):
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            self.lr_scheduler.step()
            logger.info('epoch {0} batch {1} current lr: {2}'.format(
                self.epoch_id + 1, self.batch_id + 1, self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.batch_id += 1


class ReduceLROnPlateauScheduler(Callback):
    def __init__(self, metric_name, minimize, reduce_factor, reduce_patience, min_lr):
        super().__init__()
        self.metric_name = metric_name
        self.minimize = minimize
        self.reduce_factor = reduce_factor
        self.reduce_patience = reduce_patience
        self.min_lr = min_lr

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        super().set_params(transformer, validation_datagen)
        self.validation_datagen = validation_datagen
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer,
                                              mode='min' if self.minimize else 'max',
                                              factor=self.reduce_factor,
                                              patience=self.reduce_patience,
                                              min_lr=self.min_lr)

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0

    def on_epoch_end(self, *args, **kwargs):
        self.model.eval()
        val_loss = self.get_validation_loss()
        metric = val_loss[self.metric_name]
        metric = metric.data.cpu().numpy()[0]
        self.model.train()

        self.lr_scheduler.step(metrics=metric, epoch=self.epoch_id)
        logger.info('epoch {0} current lr: {1}'.format(self.epoch_id + 1,
                                                       self.optimizer.state_dict()['param_groups'][0]['lr']))
        neptune.send_metric('Learning Rate', x=self.epoch_id, y=self.optimizer.state_dict()['param_groups'][0]['lr'])

        self.epoch_id += 1


class InitialLearningRateFinder(Callback):
    def __init__(self, min_lr=1e-8, multipy_factor=1.05, add_factor=0.0):
        super().__init__()
        self.min_lr = min_lr
        self.multipy_factor = multipy_factor
        self.add_factor = add_factor

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        super().set_params(transformer, validation_datagen)
        self.validation_datagen = validation_datagen
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]
        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        logger.info('Learning Rate {} Loss {})'.format(current_lr, loss))
        neptune.send_metric('Learning Rate Finder', x=self.batch_id, y=current_lr)
        neptune.send_metric('Loss', x=self.batch_id, y=loss)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr * self.multipy_factor + self.add_factor
        self.batch_id += 1


class ExperimentTiming(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every
        self.batch_start = None
        self.epoch_start = None
        self.current_sum = None
        self.current_mean = None

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        logger.info('starting training...')

    def on_train_end(self, *args, **kwargs):
        logger.info('training finished')

    def on_epoch_begin(self, *args, **kwargs):
        if self.epoch_id > 0:
            epoch_time = datetime.now() - self.epoch_start
            if self.epoch_every:
                if (self.epoch_id % self.epoch_every) == 0:
                    logger.info('epoch {0} time {1}'.format(self.epoch_id - 1, str(epoch_time)[:-7]))
        self.epoch_start = datetime.now()
        self.current_sum = timedelta()
        self.current_mean = timedelta()
        logger.info('epoch {0} ...'.format(self.epoch_id))

    def on_batch_begin(self, *args, **kwargs):
        if self.batch_id > 0:
            current_delta = datetime.now() - self.batch_start
            self.current_sum += current_delta
            self.current_mean = self.current_sum / self.batch_id
        if self.batch_every:
            if self.batch_id > 0 and (((self.batch_id - 1) % self.batch_every) == 0):
                logger.info('epoch {0} average batch time: {1}'.format(self.epoch_id, str(self.current_mean)[:-5]))
        if self.batch_every:
            if self.batch_id == 0 or self.batch_id % self.batch_every == 0:
                logger.info('epoch {0} batch {1} ...'.format(self.epoch_id, self.batch_id))
        self.batch_start = datetime.now()


class NeptuneMonitor(Callback):
    def __init__(self, image_nr, image_resize, image_every, model_name, use_depth):
        super().__init__()
        self.model_name = model_name
        self.epoch_loss_averager = Averager()
        self.image_resize = image_resize
        self.image_every = image_every
        self.image_nr = image_nr
        self.use_depth = use_depth

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averagers = {}
        self.epoch_id = 0
        self.batch_id = 0

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]

            if name in self.epoch_loss_averagers.keys():
                self.epoch_loss_averagers[name].send(loss)
            else:
                self.epoch_loss_averagers[name] = Averager()
                self.epoch_loss_averagers[name].send(loss)

            neptune.send_metric('{} batch {} loss'.format(self.model_name, name), x=self.batch_id, y=loss)

        self.batch_id += 1

    def on_epoch_end(self, *args, **kwargs):
        self._send_numeric_channels()
        if self.image_every is not None and self.epoch_id % self.image_every == 0:
            self._send_image_channels()
        self.epoch_id += 1

    def _send_numeric_channels(self, *args, **kwargs):
        for name, averager in self.epoch_loss_averagers.items():
            epoch_avg_loss = averager.value
            averager.reset()
            neptune.send_metric('{} epoch {} loss'.format(self.model_name, name), x=self.epoch_id,
                                             y=epoch_avg_loss)

        self.model.eval()
        val_loss = self.get_validation_loss()
        self.model.train()
        for name, loss in val_loss.items():
            loss = loss.data.cpu().numpy()[0]
            neptune.send_metric('{} epoch_val {} loss'.format(self.model_name, name), x=self.epoch_id,
                                             y=loss)

    def _send_image_channels(self):
        self.model.eval()
        image_triplets = self._get_image_triplets()
        if self.image_nr is not None:
            image_triplets = image_triplets[:self.image_nr]
        self.model.train()

        for i, image_triplet in enumerate(image_triplets):
            h, w = image_triplet.shape[1:]
            image_glued = np.zeros((h, 3 * w + 20))

            image_glued[:, :w] = image_triplet[0, :, :]
            image_glued[:, (w + 10):(2 * w + 10)] = image_triplet[1, :, :]
            image_glued[:, (2 * w + 20):] = image_triplet[2, :, :]

            pill_image = Image.fromarray((image_glued * 255.).astype(np.uint8))
            h_, w_ = image_glued.shape
            pill_image = pill_image.resize((int(self.image_resize * w_), int(self.image_resize * h_)),
                                           Image.ANTIALIAS)

            neptune.send_image('{} predictions'.format(self.model_name), pill_image)

    def _get_image_triplets(self):
        image_triplets = []
        batch_gen, steps = self.validation_datagen
        for batch_id, data in enumerate(batch_gen):
            predictions, targets_tensors = self._get_predictions_targets(data)

            raw_images = data[0].numpy()
            ground_truth_masks = targets_tensors[0].cpu().numpy()
            h, w = raw_images.shape[-2:]

            for image, prediction, target in zip(raw_images, predictions, ground_truth_masks):
                image_triplet = np.zeros((3, h, w))
                if image.shape[0] > 3:
                    image_ = image[0, :, :]
                else:
                    image_ = denormalize(image)[0, :, :]
                image_triplet[0, :, :] = image_
                image_triplet[1, :, :] = prediction[1, :, :]
                image_triplet[2, :, :] = target[1:, :, :]
                image_triplets.append(image_triplet)
            break
        return image_triplets

    def _get_predictions_targets(self, data):
        if self.use_depth:
            X = data[0]
            D = data[1]
            targets_tensors = data[2:]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
                D = Variable(D, volatile=True).cuda()
            else:
                X = Variable(X, volatile=True)
                D = Variable(D, volatile=True)

            predictions = sigmoid(self.model(X, D).data.cpu().numpy())
        else:
            X = data[0]
            targets_tensors = data[1:]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
            else:
                X = Variable(X, volatile=True)

            predictions = sigmoid(self.model(X).data.cpu().numpy())
        return predictions, targets_tensors


def denormalize(x):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    return x * std + mean


class ValidationMonitor(Callback):
    def __init__(self, data_dir, loader_mode, epoch_every=None, batch_every=None, use_depth=False):
        super().__init__()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

        self.use_depth = use_depth
        self.data_dir = data_dir
        self.validation_pipeline = postprocessing_pipeline_simplified
        self.loader_mode = loader_mode
        self.meta_valid = None
        self.y_true = None
        self.activation_func = None

    def set_params(self, transformer, validation_datagen, meta_valid=None, *args, **kwargs):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.output_names = transformer.output_names
        self.validation_datagen = validation_datagen
        self.meta_valid = meta_valid
        self.y_true = read_masks(self.meta_valid[Y_COLUMN].values)
        self.activation_func = transformer.activation_func
        self.transformer = transformer

    def get_validation_loss(self):
        return self._get_validation_loss()

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = self.get_validation_loss()
            self.model.train()
            for name, loss in val_loss.items():
                loss = loss.data.cpu().numpy()[0]
                logger.info('epoch {0} validation {1}:     {2:.5f}'.format(self.epoch_id, name, loss))
        self.epoch_id += 1

    def _get_validation_loss(self):
        output, epoch_loss = self._transform()
        logger.info('Selecting best threshold')

        iout_best, threshold_best = 0.0, 0.5
        for threshold in np.linspace(0.5, 0.3, 21):
            y_pred = self._generate_prediction(output, threshold)
            iout_score = intersection_over_union_thresholds(self.y_true, y_pred)
            logger.info('threshold {} IOUT {}'.format(threshold, iout_score))
            if iout_score > iout_best:
                iout_best = iout_score
                threshold_best = threshold
            else:
                break
        logger.info('Selected best threshold {} IOUT {}'.format(threshold_best, iout_best))

        logger.info('Calculating IOU and IOUT Scores')
        y_pred = self._generate_prediction(output, threshold_best)
        iout_score = intersection_over_union_thresholds(self.y_true, y_pred)
        iou_score = intersection_over_union(self.y_true, y_pred)
        logger.info('IOU score on validation is {}'.format(iou_score))
        logger.info('IOUT score on validation is {}'.format(iout_score))

        if not self.transformer.validation_loss:
            self.transformer.validation_loss = {}
        self.transformer.validation_loss.setdefault(self.epoch_id, {'sum': epoch_loss,
                                                                    'iou': Variable(torch.Tensor([iou_score])),
                                                                    'iout': Variable(torch.Tensor([iout_score]))})
        return self.transformer.validation_loss[self.epoch_id]

    def _transform(self):
        self.model.eval()
        batch_gen, steps = self.validation_datagen
        partial_batch_losses = []
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            targets_var, outputs_batch = self._get_targets_and_output(data)

            if len(self.output_names) == 1:
                for (name, loss_function_one, weight), target in zip(self.loss_function, targets_var):
                    loss_sum = loss_function_one(outputs_batch, target) * weight
                outputs.setdefault(self.output_names[0], []).append(outputs_batch.data.cpu().numpy())
            else:
                batch_losses = []
                for (name, loss_function_one, weight), output, target in zip(self.loss_function, outputs_batch,
                                                                             targets_var):
                    loss = loss_function_one(output, target) * weight
                    batch_losses.append(loss)
                    partial_batch_losses.setdefault(name, []).append(loss)
                    output_ = output.data.cpu().numpy()
                    outputs.setdefault(name, []).append(output_)
                loss_sum = sum(batch_losses)
            partial_batch_losses.append(loss_sum)
            if batch_id == steps:
                break
        self.model.train()
        average_losses = sum(partial_batch_losses) / steps
        outputs = {'{}_prediction'.format(name): get_list_of_image_predictions(outputs_) for name, outputs_ in
                   outputs.items()}
        for name, prediction in outputs.items():
            if self.activation_func == 'softmax':
                outputs[name] = [softmax(single_prediction, axis=0) for single_prediction in prediction]
            elif self.activation_func == 'sigmoid':
                outputs[name] = [sigmoid(np.squeeze(mask)) for mask in prediction]
            else:
                raise Exception('Only softmax and sigmoid activations are allowed')

        return outputs, average_losses

    def _get_targets_and_output(self, data):
        if self.use_depth:
            X = data[0]
            D = data[1]
            targets_tensors = data[2:]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
                D = Variable(D, volatile=True).cuda()
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True).cuda())
            else:
                X = Variable(X, volatile=True)
                D = Variable(D, volatile=True)
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True))
            outputs_batch = self.model(X, D)
        else:
            X = data[0]
            targets_tensors = data[1:]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True).cuda())
            else:
                X = Variable(X, volatile=True)
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True))
            outputs_batch = self.model(X)

        return targets_var, outputs_batch

    def _generate_prediction(self, outputs, threshold):
        data = {'callback_input': {'meta': self.meta_valid,
                                   'meta_valid': None,
                                   },
                'network_output': {**outputs}
                }
        with TemporaryDirectory() as cache_dirpath:
            pipeline = self.validation_pipeline(cache_dirpath, self.loader_mode, threshold)
            output = pipeline.transform(data)
        y_pred = output['y_pred']
        return y_pred


class ValidationMonitorEmptiness(Callback):
    def __init__(self, data_dir, loader_mode, epoch_every=None, batch_every=None, use_depth=False):
        super().__init__()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

        self.use_depth = use_depth
        self.data_dir = data_dir
        self.validation_pipeline = None
        self.loader_mode = loader_mode
        self.meta_valid = None
        self.y_true = None
        self.activation_func = None

    def set_params(self, transformer, validation_datagen, meta_valid=None, *args, **kwargs):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.output_names = transformer.output_names
        self.validation_datagen = validation_datagen
        self.meta_valid = meta_valid
        self.y_true = self.meta_valid[Y_COLUMN].values
        self.activation_func = transformer.activation_func
        self.transformer = transformer

    def get_validation_loss(self):
        return self._get_validation_loss()

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = self.get_validation_loss()
            self.model.train()
            for name, loss in val_loss.items():
                loss = loss.data.cpu().numpy()[0]
                logger.info('epoch {0} validation {1}:     {2:.5f}'.format(self.epoch_id, name, loss))
        self.epoch_id += 1

    def _get_validation_loss(self):
        output, epoch_loss = self._transform()
        y_pred = self._generate_prediction(output)

        auc_score = roc_auc_score(self.y_true, y_pred)
        logger.info('AUC score on validation is {}'.format(auc_score))

        if not self.transformer.validation_loss:
            self.transformer.validation_loss = {}
        self.transformer.validation_loss.setdefault(self.epoch_id, {'sum': epoch_loss,
                                                                    'auc': Variable(torch.Tensor([auc_score])),
                                                                    })
        return self.transformer.validation_loss[self.epoch_id]

    def _transform(self):
        self.model.eval()
        batch_gen, steps = self.validation_datagen
        partial_batch_losses = []
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            targets_var, outputs_batch = self._get_targets_and_output(data)

            if len(self.output_names) == 1:
                for (name, loss_function_one, weight), target in zip(self.loss_function, targets_var):
                    loss_sum = loss_function_one(outputs_batch, target) * weight
                outputs.setdefault(self.output_names[0], []).append(outputs_batch.data.cpu().numpy())
            else:
                batch_losses = []
                for (name, loss_function_one, weight), output, target in zip(self.loss_function, outputs_batch,
                                                                             targets_var):
                    loss = loss_function_one(output, target) * weight
                    batch_losses.append(loss)
                    partial_batch_losses.setdefault(name, []).append(loss)
                    output_ = output.data.cpu().numpy()
                    outputs.setdefault(name, []).append(output_)
                loss_sum = sum(batch_losses)
            partial_batch_losses.append(loss_sum)
            if batch_id == steps:
                break
        self.model.train()
        average_losses = sum(partial_batch_losses) / steps
        outputs = {'{}_prediction'.format(name): get_list_of_image_predictions(outputs_) for name, outputs_ in
                   outputs.items()}
        for name, prediction in outputs.items():
            if self.activation_func == 'softmax':
                outputs[name] = [softmax(single_prediction, axis=0) for single_prediction in prediction]
            elif self.activation_func == 'sigmoid':
                outputs[name] = [sigmoid(np.squeeze(mask)) for mask in prediction]
            else:
                raise Exception('Only softmax and sigmoid activations are allowed')

        return outputs, average_losses

    def _get_targets_and_output(self, data):
        if self.use_depth:
            X = data[0]
            D = data[1]
            targets_tensors = data[2:]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
                D = Variable(D, volatile=True).cuda()
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True).cuda())
            else:
                X = Variable(X, volatile=True)
                D = Variable(D, volatile=True)
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True))
            outputs_batch = self.model(X, D)
        else:
            X = data[0]
            targets_tensors = data[1:]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True).cuda())
            else:
                X = Variable(X, volatile=True)
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True))
            outputs_batch = self.model(X)

        return targets_var, outputs_batch

    def _generate_prediction(self, output):
        y_pred = output['mask_prediction']
        y_pred = np.stack(y_pred)[:, 1]
        return y_pred


class ModelCheckpoint(Callback):
    def __init__(self, filepath, metric_name='sum', epoch_every=1, minimize=True):
        self.filepath = filepath
        self.minimize = minimize
        self.best_score = None

        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every

        self.metric_name = metric_name

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = self.get_validation_loss()
            loss_sum = val_loss[self.metric_name]
            loss_sum = loss_sum.data.cpu().numpy()[0]

            self.model.train()

            if self.best_score is None:
                self.best_score = loss_sum

            if (self.minimize and loss_sum < self.best_score) or (not self.minimize and loss_sum > self.best_score) or (
                    self.epoch_id == 0):
                self.best_score = loss_sum
                persist_torch_model(self.model, self.filepath)
                logger.info('epoch {0} model saved to {1}'.format(self.epoch_id, self.filepath))

        self.epoch_id += 1


class EarlyStopping(Callback):
    def __init__(self, metric_name='sum', patience=1000, minimize=True):
        super().__init__()
        self.patience = patience
        self.minimize = minimize
        self.best_score = None
        self.epoch_since_best = 0
        self._training_break = False
        self.metric_name = metric_name

    def training_break(self, *args, **kwargs):
        return self._training_break

    def on_epoch_end(self, *args, **kwargs):
        self.model.eval()
        val_loss = self.get_validation_loss()
        loss_sum = val_loss[self.metric_name]
        loss_sum = loss_sum.data.cpu().numpy()[0]

        self.model.train()

        if not self.best_score:
            self.best_score = loss_sum

        if (self.minimize and loss_sum < self.best_score) or (not self.minimize and loss_sum > self.best_score):
            self.best_score = loss_sum
            self.epoch_since_best = 0
        else:
            self.epoch_since_best += 1

        if self.epoch_since_best > self.patience:
            self._training_break = True
        self.epoch_id += 1


def postprocessing_pipeline_simplified(cache_dirpath, loader_mode, threshold):
    if loader_mode == 'resize_and_pad':
        size_adjustment_function = partial(crop_image, target_size=ORIGINAL_SIZE)
    elif loader_mode == 'resize' or loader_mode == 'stacking':
        size_adjustment_function = partial(resize_image, target_size=ORIGINAL_SIZE)
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize',
                       transformer=make_apply_transformer(size_adjustment_function,
                                                          output_name='resized_images',
                                                          apply_on=['images']),
                       input_data=['network_output'],
                       adapter=Adapter({'images': E('network_output', 'mask_prediction'),
                                        }),
                       experiment_directory=cache_dirpath)

    binarizer = Step(name='binarizer',
                     transformer=make_apply_transformer(
                         partial(binarize, threshold=threshold),
                         output_name='binarized_images',
                         apply_on=['images']),
                     input_steps=[mask_resize],
                     adapter=Adapter({'images': E(mask_resize.name, 'resized_images'),
                                      }),
                     experiment_directory=cache_dirpath)

    output = Step(name='output',
                  transformer=IdentityOperation(),
                  input_steps=[binarizer],
                  adapter=Adapter({'y_pred': E(binarizer.name, 'binarized_images'),
                                   }),
                  experiment_directory=cache_dirpath)

    return output
