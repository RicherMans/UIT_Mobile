# Unified Keyword Spotting and Audio Tagging on Mobile Devices with Transformers


This repository contains the source code for the ICASSP 2023 Paper "Unified Keyword Spotting and Audio Tagging on Mobile Devices with Transformers".
The aim is to deploy transformer models on mobile devices, which are both capable as a keyword spotter and a audio tagger.


Notable Features:

* We propose three pico-sized models, based on ViT, which are optimized for speed and real-world deployment.
* Performance of 97.76 on GSCV1 and an mAP of 34.1 on Audioset while only using the balanced training dataset. This is a joint model that can do both keyword spotting and Audio tagging.
* A model delay of 1s, which provides a reasonable user-experience (compared to most transformers in the field).
* All models are fast on mobile devices and can be used for online audio tagging.

We evaluated all three proposed UiT models on different mobile hardware in regards to speed (ms), where TC-Resnet8 is a KWS "speed topline" and MobileNetV2 is an an Audio tagging baseline:


| Model      | SD865 | SD888 | G90T | MT700 |
|------------|-------|-------|------|-------|
| TC-ResNet8 | 0.4   | 0.4   | 1.1  | 1.1   |
| MBv2       | 8.0   | 6.2   | 13.1 | 11.6  |
| Uit-XS     | 3.4   | 3.4   | 7.3  | 7.1   |
| UiT-2XS    | 1.7   | 1.5   | 2.8  | 3.2   |
| UiT-3XS    | 1.2   | 1.1   | 2.2  | 2.2   |


While the Audioset performance appears to be not at SOTA-levels, its important to note that UiT is evaluated on *1s crops*, which heavily degrades Audiosets performance.
To give an idea, if we evaluate [AST](https://github.com/YuanGongND/ast) with the same 1s crops, we achieve an mAP of 36.56 and [CNN14](https://github.com/qiuqiangkong/audioset_tagging_cnn) achieves 38.55.
Both of the mentioned models were trained on the *full* Audioset dataset.


## Preparation

```bash
git clone https://github.com/RicherMans/UIT_Mobile
pip3 install -r requirements.txt
```

## Dataset acquisition


### Audioset

We propose simple preprocessing scripts in `datasets/` for `Audioset` and `GSCV1`.


For getting the (balanced) Audioset data you need [gnu-parallel](https://www.gnu.org/s/parallel) and [yt-dlp](https://github.com/yt-dlp/yt-dlp).
Downloading the balanced audioset is quick ( for me takes about 30 minutes):

```bash
cd datasets/audioset/
./1_download_audioset.sh
# After having downloaded the dataset, dump the .wav to .h5
./2_prepare_data.sh
```


### GSCV1

For preparing Google Speech Commands:

```bash
cd datasets/gsc/
./1_download_gscv1.sh
# After having downloaded the dataset, dump the .wav to .h5
python3 2_prepare_data.py
```



## Inference


We prepare a simple script to run inference for all three proposed UiT-XS/XXS/XXXS models.
The checkpoints are hosted on [zenodo](https://zenodo.org/record/7690036).

Running inference is simple:

```bash
python3 inference.py samples/water*
```

outputs:

```
===== samples/water_000.wav =====
Water                          0.4467
Trickle, dribble               0.3263
Gush                           0.1718
Stream                         0.1509
Speech                         0.1239
===== samples/water_001.wav =====
Trickle, dribble               0.4133
Water                          0.3864
Stream                         0.3351
Speech                         0.1716
Gush                           0.1512
===== samples/water_002.wav =====
Water                          0.4017
Trickle, dribble               0.3091
Speech                         0.2379
Gush                           0.2190
Stream                         0.1722
===== samples/water_003.wav =====                                                               
Trickle, dribble               0.5570                                                           
Water                          0.4017
Pour                           0.2454
Stream                         0.2454
Liquid                         0.1546
===== samples/water_004.wav =====
Trickle, dribble               0.3870
Stream                         0.3867
Water                          0.3668
Speech                         0.1630
Bathtub (filling or washing)   0.1135
```

An example for KWS:

```bash
python3 inference.py samples/85b877b5_nohash_0.wav

### Prints:
#===== samples/85b877b5_nohash_0.wav =====
#Speech                         1.0000
#Keyword: on                    0.9999
#Inside, small room             0.0001
```

One can change the models (`uit_xxs`, `uit_xxxs`):

```bash
python3 inference.py -m uit_xxs samples/85b877b5_nohash_0.wav

### Prints:
#===== samples/85b877b5_nohash_0.wav =====
#Speech                         0.9999
#Keyword: on                    0.9885
#Clicking                       0.0196
```

## Training


After having prepared the data, to train a model just run:

```bash
# For UiT-XS
python3 run.py run config/train_uit_xs.yaml
# For UiT-XXS
python3 run.py run config/train_uit_xxs.yaml
# For UiT-XXXS
python3 run.py run config/train_uit_xxxs.yaml
```


After having trained the model you can use it for inference as:

```bash
python3 inference.py -m $PATH_TO_YOUR_CHECKPOINT samples/85b877b5_nohash_0.wav
```

### Evaluation

There is a separate evaluation script:

```bash
python3 evaluate.py audioset YOUR_CHECKPOINT
python3 evaluate.py gsc YOUR_CHECKPOINT
```

We also provide the results for our own checkpoints:

```bash
python3 evaluate.py gsc uit_xs
# Gives
# [2023-03-02 09:56:36] GSC Results                                                    
# Accuracy@0.2 : 97.76
python3 evaluate.py gsc uit_xxs
# Gives
# [2023-03-02 09:56:36] GSC Results                                                    
# Accuracy@0.2 : 97.29
python3 evaluate.py gsc uit_xxxs
# Gives
# [2023-03-02 09:56:36] GSC Results                                                    
# Accuracy@0.2 : 97.18
```
