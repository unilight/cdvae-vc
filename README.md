# CDVAE VC
- Official TensorFlow implementation of CDVAE-VC for reproducing the results in various papers.
- This repository uses the the Voice Conversion Challenge 2018 (VCC 2018) dataset.
- This repository provides the following models to reproduce the results for our TETCI paper:
  - VAE with FCN structure
  - CDVAE with FCN structure (please refer to our [Interspeech 2019 paper](https://arxiv.org/pdf/1905.00615.pdf).)
  - CDVAE-CLS-GAN with FCN structure (refer to our TETCI journal paper; will release when accepted.)

## Requirements

The following requirement is not a minimum set but should cover packages needed to run everything.

- tensorflow-gpu 1.4
- h5py
- scipy
- [sprocket-vc](https://github.com/k2kobayashi/sprocket)
- librosa

## Basic Usage

### Architectures & Hyperparameters

The model architectures and hyperparameters of the models are stored in the JSON format. Paths to feature, configuration and statistics files are also in the architecture files. They are in the `architectures` directory and names as `architecture-{model}.json`. Feel free to modify them for your own purposes.

### Downloading the dataset

First we download the VCC2018 dataset. It will be saved in `data/vcc2018/wav/`.

```
cd data
sh download.sh
cd ..
```

### preprocessing

Then, we perform feature extraction and calculate statistics including:

- min/max, mean/std of spectral features
- mean/std of f0
- global variances of spectral features

The extracted features will be stored in binary format for fast TensorFlow data loading, and the statistics will be stored in h5 format. You can decide where to store these files. We provide a default execution script as follows:

```
python preprocessing/vcc2018/feature_extract.py \
  --waveforms data/vcc2018/wav/ \
  --bindir data/vcc2018/bin/ \
  --confdir data/vcc2018/conf
python preprocessing/vcc2018/calc_stats.py \
  --bindir data/vcc2018/bin/VAD/ \
  --stats data/vcc2018/stats/stats.h5 \
  --spklist data/vcc2018/conf/spk.list
```

### Training

Now we are ready to train a model.

``` 
CUDA_VISIBLE_DEVICES={GPU-index} python train.py \
  --architecture architectures/architecture-{model}.json \
  --note {note-to-dsitinguish-different-runs}
```

It is adviced to specify `CUDA_VISIBLE_DEVICES` since TensorFlow tends to occupy all available GPUs. After a training run starts, a directory will be created in `logdir/{training-timestamp}-{note}`. It is suggested to use the `--note` argument to distinguish different training runs. Each training directory contains a copy of the architecture file, a training log file and the saved model weight files.

### Conversion, Synthesizing and MCD calculation

After we train a model, we can perform conversion of spectral features, synthesize waveforms from the converted feautures and calculate the mel-cepstrum distortion (MCD) of the converted features using the trained model.

``` 
CUDA_VISIBLE_DEVICES={GPU-index} python convert.py \
  --logdir logdir/{training-timestamp}-{note}/ \
  --src {src} --trg {trg} \
  --input_feat {sp or mcc} --output_feat {sp or mcc} \
  --mcd --syn
```

Any of the two speakers in `data/vcc2018/conf/spk.list` can form a conversion pair. You can also use the same speaker as source and target to perform autoencoding (reconstruction). Note that only sp and mcc are available for the VAE and CDVAE-CLS-GAN models, respectively.

A directory will be created for each conversion run in `logdir/{training-timestamp}-{note}/{testing-timestamp}-{src}-{trg}`. A experiment log file, the latent code files, the converted feature files and converted waveforms can be found in it.

It is suggested to specify the `--syn` and `--mcd` flags so that the three procedures can be performed within one command (which is what we usually desire). However, if you wish, you can still execute them seperately:

```
python mcd_calculate.py \
  --logdir logdir/{training-timestamp}-{note}/{testing-timestamp}-{src}-{trg} \
  --input_feat {sp or mcc} --output_feat {sp or mcc}
python synthesize.py \
  --logdir logdir/{training-timestamp}-{note}/{testing-timestamp}-{src}-{trg} \
  --input_feat {sp or mcc} --output_feat {sp or mcc}
```

### Note

There are many command line arguments and architecture parameters that can be modified. Feel free to play with them and ask about the purpose and usage.

## TODO

- [ ] validation script
- [ ] pretrained model weights

## Papers and Audio samples
- ISCSLP 2018 conference paper: [[arXiv]](https://arxiv.org/pdf/1808.09634.pdf) [[Demo]](https://unilight.github.io/CDVAE-Demo/)
- Interspeech 2019 conference paper: [[arXiv]](https://arxiv.org/pdf/1905.00615.pdf) [[Demo]](https://unilight.github.io/Publication-Demos/publications/f0-fcn-cdvae/)
- TETCI journal paper (under review): [under review] [[Demo]](https://unilight.github.io/CDVAE-GAN-CLS-Demo/)


## References

Please cite the following articles.

```
@inproceedings{huang2018voice,
  title={Voice conversion based on cross-domain features using variational auto encoders},
  author={Huang, Wen-Chin and Hwang, Hsin-Te and Peng, Yu-Huai and Tsao, Yu and Wang, Hsin-Min},
  booktitle={2018 11th International Symposium on Chinese Spoken Language Processing (ISCSLP)},
  pages={51--55},
  year={2018},
  organization={IEEE}
}
@inproceedings{huang2019f0fcncdvae,
  author={Huang, Wen-Chin and Wu, Yi-Chiao and Lo, Chen-Chou and
         Lumban Tobing, Patrick and Hayashi, Tomoki and
         Kobayashi, Kazuhiro and Toda, Tomoki and Tsao, Yu and Wang, Hsin-Min},
  title={Investigation of F0 conditioning and Fully Convolutional Networks in Variational Autoencoder based Voice Conversion},
  year=2019,
  booktitle={Proc. Interspeech 2019},
}
@article{huang19, 
  author={Huang, Wen-Chin and Luo, Hao and Hwang, Hsin-Te and Lo, Chen-Chou and
         Peng, Yu-Huai and Tsao, Yu and Wang, Hsin-Min},
  journal={Submited to IEEE Transactions on Emerging Topics in Computational Intelligence},
  title={Unsupervised Representation Disentanglement using Cross Domain Features and Adversarial Learning in Variational Autoencnder based Voice Conversion}, 
  year={2019},
}

```

## Acknowledgements

This repository is greatly inspired from the [official implmenetation](https://github.com/JeremyCCHsu/vae-npvc) of the original VAE-VC, and used lots of codes snippets from the [sprocket](https://github.com/k2kobayashi/sprocket) voice conversion software and the [PyTorchWaveNetVocoder](https://github.com/kan-bayashi/PytorchWaveNetVocoder) repository.

## Authors

- Wen-Chin Huang, Graduate School of Informatics, Nagoya University, Japan. [[github]](https://github.com/unilight/)
- Hao Luo, Institute of Information Science, Academia Sinica, Taiwan.
