from typing import List, Tuple, Dict, Any, Union, Callable, Iterable, Optional
from loguru import logger
from fire import Fire
import pandas as pd
import uuid
import numpy as np

import models
import dataset
import utils
import torch
import sys
import datetime
from pathlib import Path
import ignite
from ignite.contrib.handlers import ProgressBar, create_lr_scheduler_with_warmup, CosineAnnealingScheduler
from ignite.engine import (Engine, Events)
from ignite.handlers import (Checkpoint, DiskSaver, global_step_from_engine,
                             EarlyStopping)

logger.configure(handlers=[{
    "sink": sys.stdout,
    "format": "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}",
    'level': 'DEBUG',
}])

DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')


def transfer_to_device(batch: Iterable, device=DEVICE):
    return (x.to(device, non_blocking=True)
            if isinstance(x, torch.Tensor) else x for x in batch)


def log_basic_info(params):
    config_parameters = params['params']
    import os
    if 'HOSTNAME' in os.environ:
        logger.info(f"Running on host {os.environ['HOSTNAME']}")

    logger.info(f"Running on device {DEVICE}")
    logger.info(f"Storing output in {params['outputdir']}")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        logger.info(f"- GPU Device: {torch.cuda.current_device()}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
    for k, v in config_parameters.items():
        logger.info(f"{k} : {v}")


def create_engine(engine_function: Callable,
                  evaluation_metrics: Optional[List[str]] = None):
    engine = Engine(engine_function)
    ProgressBar().attach(engine, output_transform=lambda x: x)

    if evaluation_metrics:
        eval_mets = utils.metrics(evaluation_metrics)
        for name, metric in eval_mets.items():
            metric.attach(engine, name)
    return engine


class Runner(object):

    def __init__(self, seed: int = 42, nthreads: int = 1):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(nthreads)

    def __setup(self,
                config: Union[Path, str],
                default_args: Dict[str, Any] = utils.DEFAULT_ARGS,
                **override_kwargs) -> Dict[str, Any]:
        config_parameters = utils.parse_config_or_kwargs(
            config, default_args=default_args, **override_kwargs)
        outputdir = Path(config_parameters['outputpath']) / Path(
            config).stem / f"{config_parameters['model']}" / "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'),
                uuid.uuid1().hex)
        outputdir.mkdir(exist_ok=True, parents=True)
        log_fname = config_parameters.get('logfile', 'train.log')
        output_log = outputdir / log_fname
        logger.add(
            output_log,
            enqueue=True,
            level='INFO',
            format=
            "[<red>{level}</red> <green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
        )
        return_params = {'outputdir': outputdir, 'params': config_parameters}
        log_basic_info(return_params)
        return return_params

    def train(self, config: Union[str, Path], **overwrite_kwargs: Dict[str,
                                                                       Any]):
        param_dict = self.__setup(config, **overwrite_kwargs)
        config_parameters: Dict[str, Any] = param_dict['params']
        outputdir: str = param_dict['outputdir']
        epochs: int = config_parameters['epochs']
        epoch_length: int = config_parameters['epoch_length']
        warmup_iters: int = config_parameters['warmup_iters']
        batch_size: int = config_parameters['batch_size']
        num_workers: int = config_parameters['num_workers']
        kws_batch_size: int = config_parameters.get('kws_batch_size',
                                                    batch_size // 2)
        as_batch_size: int = config_parameters.get('as_batch_size',
                                                   batch_size // 2)
        early_stop: int = config_parameters.get('early_stop', 10)
        mixup_alpha: float = config_parameters.get('mixup', None)
        use_scheduler: bool = config_parameters.get('use_scheduler', True)
        num_classes: int = config_parameters.get('num_classes', 527)
        as_sampler = config_parameters.get('as_sampler', None)
        kws_sampler = config_parameters.get('kws_sampler', None)
        use_mask = config_parameters.get('use_mask', True)
        pretrained_path = config_parameters.get('pretrained', None)
        spectransforms = utils.parse_spectransforms(
            config_parameters.get('spectransforms', []))
        wavtransforms = utils.parse_wavtransforms(
            config_parameters.get('wavtransforms', []))
        chunk_length: float = config_parameters.get('chunk_length', None)
        max_grad_norm: bool = config_parameters.get('max_grad_norm', None)
        psl_model_params = config_parameters.get('psl')
        basename = config_parameters.get('basename', True)

        model = getattr(models, config_parameters['model'])(
            spectransforms=spectransforms,
            wavtransforms=wavtransforms,
            outputdim=num_classes,
            **config_parameters['model_args'])
        logger.info(model)

        if pretrained_path is not None:
            if 'http' in pretrained_path:
                pretrained_dump = torch.hub.load_state_dict_from_url(
                    pretrained_path
                )
                utils.load_pretrained(model,
                                      trained_model=pretrained_dump)
            else:
                utils.load_pretrained(model,
                                      trained_model=torch.load(
                                          pretrained_path, map_location='cpu'))

        model = model.to(DEVICE).train()

        if config_parameters['optimizer'] == 'Adam8bit':
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                model.parameters(),
                **config_parameters['optimizer_args'])  # add bnb optimizer
        else:
            optimizer = getattr(torch.optim, config_parameters['optimizer'])(
                model.parameters(), **config_parameters['optimizer_args'])

        criterion = getattr(torch.nn, config_parameters['loss'])(
            **config_parameters.get('loss_args', {}))

        psl_model = None
        if psl_model_params is not None:
            logger.info(f"Using PSL model {psl_model_params['model']}")
            psl_model = getattr(models,
                                psl_model_params['model'])(outputdim=527)
            if 'http' in psl_model_params['pretrained']:
                psl_model_dump = torch.hub.load_state_dict_from_url(
                    psl_model_params['pretrained']
                )
            else:
                psl_model_dump = torch.load(psl_model_params['pretrained'],
                                            map_location='cpu')
            psl_model = utils.load_pretrained(psl_model, psl_model_dump)
            psl_model = psl_model.to(DEVICE).eval()

        def _forward(x, y, lengths=None) -> torch.Tensor:
            if mixup_alpha is not None and mixup_alpha > 0.0:
                mixup_lamb = torch.tensor(np.random.beta(mixup_alpha,
                                                         mixup_alpha,
                                                         size=len(x)),
                                          device=DEVICE,
                                          dtype=torch.float32)
                # calucate the maximum over the two original lengths
                if lengths is not None:
                    lengths = utils.mixup_lengths(lengths)
                model_pred = model(x, mixup_lamb)
                y = utils.mixup_single(y, mixup_lamb)
            else:
                model_pred = model(x)
            return criterion(model_pred, y)

        def _train_with_psl(engine, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad(set_to_none=True)
                as_x, as_y, as_lengths, *_ = transfer_to_device(
                    batch['audioset'])
                kws_x, kws_y, kws_lengths, *_ = transfer_to_device(
                    batch['kws'])
                with torch.no_grad():
                    y_teacher = psl_model(as_x).detach()
                #Copy over the new labels from PSL
                as_y[:, :527] = y_teacher[:, :527]
                x = torch.cat((as_x, kws_x), dim=0)
                y = torch.cat((as_y, kws_y), dim=0)
                lengths = torch.cat((as_lengths, kws_lengths), dim=0)
                # Only use PSL for the audioset dataset
                if not use_mask:
                    lengths = None
                loss = _forward(x, y, lengths)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_grad_norm)
                optimizer.step()
                return {
                    'total_loss': loss.item(),
                    'lr': optimizer.param_groups[0]['lr'],
                }

        def _default_train_batch(engine, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad(set_to_none=True)
                x, y, lengths, *_ = transfer_to_device(batch)
                # Only use PSL for the audioset dataset
                if not use_mask:
                    lengths = None
                loss = _forward(x, y)
                loss.backward()
                optimizer.step()
                return {
                    'total_loss': loss.item(),
                    'lr': optimizer.param_groups[0]['lr'],
                }

        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                data, targets, lengths, *_ = transfer_to_device(batch)
                if not use_mask:
                    lengths = None
                clip_out = model(data)
                return clip_out, targets

        def run_validation(engine, title=None):
            results = engine.state.metrics
            output_str_list = [
                f"{title:<10} Results - Epoch : {train_engine.state.epoch:<4}"
            ]
            for metric in results:
                if isinstance(results[metric], np.ndarray):
                    pass
                else:
                    output_str_list += [f"{metric} {results[metric]:<5.4f}"]
            output_str_list += [f"LR: {optimizer.param_groups[0]['lr']:.2e}"]
            logger.info(" ".join(output_str_list))

        train_engine = create_engine(
            _default_train_batch if psl_model is None else _train_with_psl)
        inference_engine = create_engine(
            _inference,
            evaluation_metrics=['mAP'])  # Common mAP between all datasets

        audioset_train_df = utils.read_tsv_data(
            config_parameters['audioset_train_data'], basename=True)
        audioset_eval_df = utils.read_tsv_data(
            config_parameters['audioset_eval_data'], basename=True)

        test_df = audioset_eval_df

        info_message = f"#Lengths: Audioset Train - {len(audioset_train_df)} Audioset Eval - {len(audioset_eval_df)}"
        logger.info(info_message)

        logger.info(
            f"Mixing with KWS data Train: {config_parameters['kws_train_data']} Test: {config_parameters['kws_test_data']}"
        )
        kws_train_df = utils.read_tsv_data(config_parameters['kws_train_data'],
                                           basename=basename)

        if psl_model is None and chunk_length is None:
            kws_ds = dataset.WeakHDF5Dataset(kws_train_df,
                                             num_classes=num_classes)
            audioset_ds = dataset.WeakHDF5Dataset(audioset_train_df,
                                                  num_classes=num_classes)
        else:
            kws_ds = dataset.WeakRandomCropHDF5Dataset(
                kws_train_df,
                chunk_length=chunk_length,
                num_classes=num_classes)
            audioset_ds = dataset.WeakRandomCropHDF5Dataset(
                audioset_train_df,
                chunk_length=chunk_length,
                num_classes=num_classes)

        kws_eval_df = utils.read_tsv_data(config_parameters['kws_test_data'],
                                          basename=basename)
        test_df = pd.concat((audioset_eval_df, kws_eval_df))
        mAPAudioset = utils.ALL_EVAL_METRICS['AP']()[:527].mean()
        mAPAudioset.attach(inference_engine, 'mAPAudioset')
        mAPKWS = utils.ALL_EVAL_METRICS['AP']()[527:].mean()
        mAPKWS.attach(inference_engine, 'mAPKWS')

        as_sampeler_kwargs = {'shuffle': True}
        kws_sampeler_kwargs = {'shuffle': True}
        if as_sampler is not None and as_sampler == 'balanced':
            as_sampeler_kwargs = {
                'sampler': dataset.BalancedSampler(audioset_train_df['labels'])
            }
        if kws_sampler is not None and kws_sampler == 'balanced':
            kws_sampeler_kwargs = {
                'sampler': dataset.BalancedSampler(kws_train_df['labels'])
            }

        train_dataloader = dataset.MultiDataLoader(
            kws=torch.utils.data.DataLoader(
                kws_ds,
                batch_size=kws_batch_size,
                num_workers=num_workers,
                collate_fn=dataset.sequential_pad,
                **kws_sampeler_kwargs,
            ),
            audioset=torch.utils.data.DataLoader(
                audioset_ds,
                batch_size=as_batch_size,
                num_workers=num_workers,
                collate_fn=dataset.sequential_pad,
                **as_sampeler_kwargs))

        test_dataloader = torch.utils.data.DataLoader(
            dataset.WeakHDF5Dataset(test_df, num_classes=num_classes),
            batch_size=config_parameters.get('eval_batch_size',
                                             config_parameters['batch_size']),
            num_workers=num_workers,
            shuffle=False,
            collate_fn=dataset.sequential_pad,
        )

        score_function = Checkpoint.get_default_score_fn(
            *config_parameters.get('score_function', ['mAP', 1.0]))
        checkpoint_saver = Checkpoint(
            {
                'model': model,
                'config': utils.DictWrapper(config_parameters),
            },
            DiskSaver(outputdir),
            n_saved=config_parameters.get('n_saved', 4),
            global_step_transform=global_step_from_engine(train_engine),
            filename_prefix='best',
            score_function=score_function)
        decay_steps = epochs * len(
            train_dataloader
        ) if epoch_length == None else epochs * epoch_length
        if use_scheduler:
            scheduler = CosineAnnealingScheduler(
                optimizer, 'lr', optimizer.param_groups[0]['lr'],
                optimizer.param_groups[0]['lr'] * 0.01, decay_steps)
            logger.info(f"Using scheduler {scheduler.__class__.__name__}")

            if warmup_iters is not None:
                logger.info(
                    f"Warmup with {warmup_iters}, if you want to disable warmup pass warmup_iters = None"
                )
                scheduler = create_lr_scheduler_with_warmup(
                    scheduler,
                    warmup_start_value=0.0,
                    warmup_duration=warmup_iters)
            train_engine.add_event_handler(Events.ITERATION_STARTED, scheduler)
        earlystop_handler = EarlyStopping(patience=early_stop,
                                          score_function=score_function,
                                          trainer=train_engine)
        # Stop on Wensheng no improvement
        inference_engine.add_event_handler(Events.COMPLETED, earlystop_handler)

        inference_engine.add_event_handler(Events.COMPLETED, checkpoint_saver)

        @train_engine.on(
            Events.EPOCH_COMPLETED(
                every=config_parameters.get('valid_every', 1)))
        def valid_eval(train_engine):
            with inference_engine.add_event_handler(Events.COMPLETED,
                                                    run_validation,
                                                    "Validation"):
                inference_engine.run(test_dataloader)

        @train_engine.on(Events.COMPLETED)
        def average_models_and_eval(engine):
            output_model = outputdir / checkpoint_saver.last_checkpoint
            if config_parameters.get('average', True):
                logger.info("Averaging best models ...")
                output_model = outputdir / 'averaged.pt'

                averaged_state_dict = utils.average_models(
                    [outputdir / f.filename for f in checkpoint_saver._saved])
                torch.save(averaged_state_dict, output_model)

                model.load_state_dict(averaged_state_dict['model'],
                                      strict=True)
            else:
                logger.info(f"Loading best model {output_model}")
                model.load_state_dict(torch.load(output_model)['model'],
                                      strict=True)
            #Final evaluation
            valid_eval(engine)
            engine.state.output_model = output_model
            logger.info(f"Results can be found at {outputdir}")
            logger.info(f"Final model is at {engine.state.output_model}")

        train_engine.run(
            train_dataloader,
            max_epochs=epochs,
            epoch_length=epoch_length,
        )
        return train_engine.state.output_model

    def run(self, config, **kwargs):
        output_dir = self.train(config, **kwargs)
        from evaluate import Evaluator
        eval = Evaluator()
        eval.gsc(output_dir)
        eval.audioset(output_dir)


if __name__ == "__main__":
    Fire(Runner)
