## Installation

**Requirements**

* Python 3.8
* Pytorch 1.12.0
* torchvision 0.13.0
* yacs 0.1.8

**Dataset Preparation**
* [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [ImageNet](http://image-net.org/index)

Change the `data_path` in `config/*/*.yaml` accordingly.

## Training

```
python train.py --cfg ./config/DATASETNAME/DATASETNAME_SUFFIX.yaml
```

`DATASETNAME` can be selected from `cifar10`,  `cifar100`, and `imagenet`.

`SUFFIX` can be `imb001/002/01` for `cifar10/100`, `resnet10/50` for `imagenet`, respectively.

The saved folder (including logs and checkpoints) is organized as follows.
```
CSA
├── saved
│   ├── modelname_date
│   │   ├── ckps
│   │   │   ├── current.pth.tar
│   │   │   └── model_best.pth.tar
│   │   └── logs
│   │       └── modelname.txt
│   ...   
```

## Acknowledgement

We thank the authors for the following repositories for code reference:
[LDAM-DRW](https://github.com/kaidic/LDAM-DRW), 
[MiSLAS](https://github.com/dvlab-research/MiSLAS)