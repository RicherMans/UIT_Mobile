import torch
import pandas as pd
import torchaudio
from pathlib import Path
import argparse
import models

PRETRAINED_CHECKPOINTS = {
        'uit_xs': {'model': models.uit.uit_xs,
                   'model_kwargs': dict(outputdim=537, target_length=102),
                   'chkpt':'checkpoints/uit_xs_mAP3409.pt'},
        'uit_xxs': {'model': models.uit.uit_xxs,
                   'model_kwargs': dict(outputdim=537, target_length=102),
                   'chkpt':'checkpoints/uit_xxs_mAP3221.pt'},
        'uit_xxxs': {'model': models.uit.uit_xxxs,
                   'model_kwargs': dict(outputdim=537, target_length=102),
                   'chkpt':'checkpoints/uit_xxxs_mAP3097.pt'},
}



def main():
    label_maps = pd.read_csv(
        Path(__file__).parent /
        'datasets/merged_class_label_indices.csv').set_index(
            'index')['display_name'].to_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument('input_wav', type=Path, nargs = "+")
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
        dump = torch.load(model_params['chkpt'], map_location='cpu')
        model = model_params['model'](**model_params['model_kwargs'])
        model.load_state_dict(dump, strict=True)
    else:
        trained_dump = torch.load(args.model, map_location='cpu')
        model_name = trained_dump['config']['model']
        model_kwargs = trained_dump['config']['model_args']
        model = trained_dump['config']['model']().load_state_dict(dump, strict=True)
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
