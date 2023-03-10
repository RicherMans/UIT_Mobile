from typing import Tuple
from ignite.metrics import Accuracy
import numpy as np
import torch
import models
import pandas as pd
import dataset
import utils
from pathlib import Path
from fire import Fire
from ignite.engine import Events
from run import DEVICE, transfer_to_device, create_engine, logger
import sys

class Evaluator(object):

    def __setup_eval(self, experiment_path):
        if not hasattr(self,'model') and Path(experiment_path).exists():
            # Do not reintialize the model of one is already present
            # Just for the case that one uses the _all function
            experiment_path = Path(experiment_path)
            model_dump_path = None
            if experiment_path.is_file():
                # Is the file itself
                model_dump_path = experiment_path
                self.experiment_path = experiment_path.parent
            else:
                # Is a directory,need to find file
                model_dump_path = next(experiment_path.glob('*pt'))
                self.experiment_path = experiment_path
            model_dump = torch.load(model_dump_path, map_location='cpu')
            config_parameters = model_dump['config']
            self.num_classes = config_parameters.get('num_classes', 527)
            self.model = getattr(models, config_parameters['model'])(
                outputdim=self.num_classes, **config_parameters['model_args'])
            self.model = self.model.to(DEVICE).eval()
            self.model = utils.load_pretrained(self.model, model_dump['model'])
            self.config = config_parameters
        elif experiment_path in models.PRETRAINED_CHECKPOINTS:
            # For pretrained model evaluation
            self.config = {}
            self.num_classes = 537
            model_params = models.PRETRAINED_CHECKPOINTS[experiment_path]
            dump = torch.hub.load_state_dict_from_url(model_params['chkpt'],
                                                      map_location='cpu')
            self.experiment_path = Path('/tmp/')
            self.model = model_params['model'](**model_params['model_kwargs'])
            self.model.load_state_dict(dump, strict=True)
            self.model.to(DEVICE).eval()

        return self

    def _inference(self, engine, batch, pad=False) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            data, targets, lengths, filenames = transfer_to_device(batch)
            if pad:
                if hasattr(self.model, 'target_length'):
                    t_len = self.model.target_length - 1
                    input_nframes = data.shape[-1] / self.model.hop_size
                    if input_nframes < t_len:
                        diff = int(
                            (t_len - input_nframes) * self.model.hop_size)
                        data = torch.nn.functional.pad(data, (0, diff),
                                                       mode='constant')
            clip_out = self.model(data)
            return clip_out, targets

    def audioset(self,
                 experiment_path: str,
                 audioset_eval_data: str = 'datasets/audioset/data/labels/eval.csv',
                 batch_size=32):
        """audioset.

        :param self:
        :param experiment_path: Finished Exp Path or Path to the final model.pt 
        :type experiment_path: str
        :param audioset_eval_data: Evaluation dataset
        :type audioset_eval_data: str
        :param batch_size: Batch size during evaluation
        """
        self.__setup_eval(experiment_path)
        df = utils.read_tsv_data(audioset_eval_data)
        dataloader = torch.utils.data.DataLoader(
            dataset.WeakHDF5Dataset(df, num_classes=527),
            batch_size=batch_size,
            num_workers=3,
            collate_fn=dataset.sequential_pad)

        def inference_audioset(clip_out, targets):
            clip_out, targets = self._inference(clip_out, targets)
            # Assert that in any way, we only use the 527 labels
            return clip_out[..., :527], targets[..., :527]

        engine = create_engine(inference_audioset,
                               evaluation_metrics=[
                                   'Precision', 'Recall',
                                   'Macro_Precision', 'Macro_Recall',
                                   'Macro_F1', 'Micro_Precision',
                                   'Micro_Recall', 'Micro_F1', 'AP',
                                   'PositiveMultiClass_Accuracy', 'mAP'
                               ])

        class_labels = Path('datasets/merged_class_label_indices.csv')
        if class_labels.exists():
            label_map_df = pd.read_csv(class_labels)
            label_map_df['display_name'] = label_map_df['display_name'].str.lower()
            label_maps = label_map_df.set_index('index')['display_name'].to_dict()
            engine.state.label_maps = label_maps
        else:
            engine.state.label_maps = None
        self.__run_eval(engine, dataloader, target='Audioset')

    def __run_eval(self,
                   engine,
                   dataloader,
                   target='Audioset',
                   label=""):
        # If we rerun the function, .remove() will only output to the newest filelog
        # and does not also update previous initilizations
        logger.remove()
        logger.configure(handlers=[{
            "sink": sys.stderr,
            "format": "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}",
            'level': 'DEBUG',
        }])

        logger.add(Path(self.experiment_path) /
                   f'evaluation_{target}{label}.txt',
                   format='{message}',
                   level='INFO',
                   mode='w')

        def log_metrics(engine, title, scale: float = 100):
            results = engine.state.metrics
            log = [f"{title:}"]
            for metric in results.keys():
                # Returned dict means that its for each class some result metric
                if isinstance(results[metric], np.ndarray):
                    if engine.state.label_maps is None:
                        engine.state.label_maps = {idx:idx for idx in range(len(results[metric]))}
                    sorted_idxs = np.argsort(results[metric])[::-1]

                    for i, cl in enumerate(sorted_idxs):
                        log.append(
                            f"{metric} Class {engine.state.label_maps[cl]} : {results[metric][cl]*scale:<4.2f}"
                        )
                else:
                    log.append(f"{metric} : {results[metric]*scale:<4.2f}")
            logger.info("\n".join(log))

        engine.add_event_handler(Events.COMPLETED, log_metrics,
                                 f"{target} Results")
        engine.run(dataloader)

    def _evaluate(self, inference_engine, test_dataloader):
        #Label Maps for audioset
        class_labels = 'data/class_labels_indices.csv'
        label_map_df = pd.read_csv(class_labels)
        label_map_df['display_name'] = label_map_df['display_name'].str.lower()
        label_maps = label_map_df.set_index('index')['display_name'].to_dict()

        def log_metrics(engine, title, scale: float = 100):
            results = engine.state.metrics
            log = [f"{title:}"]
            for metric in results.keys():
                # Returned dict means that its for each class some result metric
                if isinstance(results[metric], np.ndarray):
                    if engine.label_maps is None:
                        engine.label_maps = {
                            idx: idx
                            for idx in range(len(results[metric]))
                        }
                    sorted_idxs = np.argsort(results[metric])[::-1]

                    for i, cl in enumerate(sorted_idxs):
                        log.append(
                            f"{metric} Class {engine.label_maps[cl]} : {results[metric][cl]*scale:<4.2f}"
                        )
                else:
                    log.append(f"{metric} : {results[metric]*scale:<4.2f}")
            logger.info("\n".join(log))

        inference_engine.label_maps = label_maps
        with inference_engine.add_event_handler(Events.COMPLETED, log_metrics,
                                                "Audioset Eval"):
            inference_engine.run(test_dataloader)

    def _kws(
            self,
            experiment_path: str,
            eval_data: str,
            threshold: float = 0.2,
            batch_size=32,
            label_name='GSC',
            pad: bool = False,  # Padding to target length
    ):
        self.__setup_eval(experiment_path)
        if pad:
            logger.info("Using Padding")
        df = utils.read_tsv_data(eval_data,
                                 basename=self.config.get('basename', False))
        dataloader = torch.utils.data.DataLoader(
            dataset.WeakHDF5Dataset(df, num_classes=self.num_classes),
            batch_size=batch_size,
            num_workers=4,
            collate_fn=dataset.sequential_pad)

        def inference_xiaoai(clip_out, targets):
            clip_out, targets = self._inference(clip_out, targets, pad=pad)
            return clip_out, targets

        def _output_transform_xiaoai(output):
            y_pred, y = output
            mask = torch.ones(y_pred.shape[0],
                              y_pred.shape[1],
                              device=y_pred.device)
            mask[:, :527] = (y_pred[:, :527] == y_pred[:, :527].max(
                dim=1, keepdim=True)[0]).to(dtype=torch.int32)
            y_pred = y_pred * mask
            _, y = y.max(dim=-1)
            for sample_idx, scores in enumerate(y_pred):
                max_filer_score_idx = scores[0:527].max(dim=-1)[1]
                # ????????????id???????????????y[sample_idx] < 527, ??????filer???????????????????????????527???????????????
                if y[sample_idx] < 527:
                    y[sample_idx] = max_filer_score_idx
                for score in scores[527:]:
                    if score >= threshold:
                        y_pred[sample_idx][max_filer_score_idx] = 0.0
            return y_pred, y

        engine = create_engine(inference_xiaoai)
        Accuracy(output_transform=_output_transform_xiaoai).attach(
            engine, f"Accuracy@{threshold}")
        self.__run_eval(engine, dataloader, target=label_name)

    def gsc(self,
            experiment_path: str,
            eval_data: str = 'datasets/gsc/data/labels/test_gsc_aslabels.tsv',
            **kwargs):
        self._kws(
            experiment_path=experiment_path,
            eval_data=eval_data,
            label_name='GSC',
            **kwargs,
        )

    def test_sample(
        self,
        experiment_path: str,
        sample: str,
    ):
        import torchaudio
        self.__setup_eval(experiment_path)
        wav, sr = torchaudio.load(sample)
        if hasattr(self.model, 'target_length'):
            t_len = self.model.target_length - 1
            input_nframes = wav.shape[-1] / self.model.hop_size
            if input_nframes < t_len:
                diff = int((t_len - input_nframes) * self.model.hop_size)
                wav = torch.nn.functional.pad(wav, (0, diff), mode='constant')
        pred = self.model(wav.to(DEVICE))[0].squeeze(0).cpu()
        for val, idx in zip(*pred.topk(5)):
            print(f"[{idx:=3}] : {val*100:.2f}")


if __name__ == "__main__":
    Fire(Evaluator)
