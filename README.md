# Unified Keyword Spotting and Audio Tagging on Mobile Devices with Transformers


This repository contains the source code for the ICASSP 2023 Paper "Unified Keyword Spotting and Audio Tagging on Mobile Devices with Transformers".
The aim is to deploy transformer models on mobile devices, which are both capable as a keyword spotter and a audio tagger.


Notable Features:

* Performance of 97.76 on GSCV1 and an mAP of 34.1 on Audioset while only using the balanced training dataset.
* A model delay of 1s, which provides a reasonable user-experience (compared to most transformers in the field).
* All models are fast on mobile devices and can be used for online audio tagging.

While this Audioset performance seems to be "low" for some people, I'd like to point out that our model is evaluated on *1s crops*, which heavily degrades Audiosets performance.
To give an idea, if we evaluate [AST](https://github.com/YuanGongND/ast) with the same setting (1s crops), we achieve an mAP of 36.56, about 10 points mAP lower compared to evaluating on 10s.


## Dataset acquisition

We propose simple preprocessing scripts in `datasets/` for `Audioset` and `GSCV1`.
For getting the (balanced) Audioset data please run:

```
bash
cd datasets/audioset/
./1_download_audioset.sh
# After having downloaded the dataset, dump the .wav to .h5
./2_prepare_data.sh
```



## Inference


We prepare a simple script to run inference for all three proposed UiT-XS/XXS/XXXS models:


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
