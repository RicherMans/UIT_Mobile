import torch
import pandas as pd
import torchaudio
from pathlib import Path
import argparse
import models
from models import PRETRAINED_CHECKPOINTS



def main():
    label_maps = pd.read_csv(
        Path(__file__).parent /
        'datasets/merged_class_label_indices.csv').set_index(
            'index')['display_name'].to_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument('input_wav', type=Path, nargs="+")
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        metavar=
        f"Public Checkpoint [{','.join(PRETRAINED_CHECKPOINTS.keys())}] or Experiement Path",
        nargs='?',
        default='uit_xs')
    parser.add_argument(
        '-k',
        '--topk',
        type=int,
        help="Print top-k results",
        default=3,
    )
    args = parser.parse_args()

    if args.model in PRETRAINED_CHECKPOINTS.keys():
        model_params = PRETRAINED_CHECKPOINTS[args.model]
        dump = torch.hub.load_state_dict_from_url(model_params['chkpt'],
                                                  map_location='cpu')
        model = model_params['model'](**model_params['model_kwargs'])
        model.load_state_dict(dump, strict=True)
    else:
        trained_dump = torch.load(args.model, map_location='cpu')
        model_name = trained_dump['config']['model']
        num_classes = trained_dump['config'].get('num_classes', 537)
        model_kwargs = trained_dump['config']['model_args']
        model = getattr(models, model_name)(outputdim=num_classes,
                                            **model_kwargs)
        model.load_state_dict(trained_dump['model'], strict=True)
    model.eval()

    for wavpath in args.input_wav:
        wave, sr = torchaudio.load(wavpath)
        assert sr == 16000, "Models are trained on 16khz, please sample your input to 16khz"
        with torch.no_grad():
            output = model(wave).squeeze(0)
            print(f"===== {str(wavpath):^20} =====")
            for prob, label in zip(*output.topk(args.topk)):
                lab_idx = label.item()
                label_name = label_maps[lab_idx]
                if lab_idx > 526:
                    label_name = f"Keyword: {label_name}"
                print(f"{label_name:<30} {prob:.4f}")




if __name__ == "__main__":
    main()
