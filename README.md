# CDVAE+
- Official Tensorflow implementation of CDVAE and CDVAE+ for VC.

## Paper

*Wen-Chin Huang, Hsin-Te Hwang, Yu-Huai Peng, Yu Tsao, Hsin-Min Wang*, "Voice Conversion Based on Cross-Domain Features Using Variational Auto Encoders", ISCSLP 2018. [[paper]](https://arxiv.org/pdf/1808.09634.pdf)

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

## Authors

- Wen-Chin Huang, Institute of Information Science, Academia Sinica. [[github]](https://github.com/unilight/)
- Hao Luo, Institute of Information Science, Academia Sinica.
