# CDVAE VC
- Official Tensorflow implementation of CDVAE-VC.
- This repository provides two kinds of models:
  - CDVAE with FCN structure (please refer to our [Interspeech 2019 paper](https://arxiv.org/pdf/1905.00615.pdf).)
  - CDVAE-CLS-GAN with FCN structure (will wait until our TETCI journal paper gets accepted.)

## Requirements

## Basic Usage

### Training

``` 
python main.py
```
### Conversion

``` 
python convert.py --src [src] --trg [trg] --logdir [training logdir]
```
### Synthesizing

``` 
python synthesize.py --logdir [conversion logdir]
```

## Notes


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

## Authors

- Wen-Chin Huang, Graduate School of Informatics, Nagoya University, Japan. [[github]](https://github.com/unilight/)
- Hao Luo, Institute of Information Science, Academia Sinica, Taiwan.
