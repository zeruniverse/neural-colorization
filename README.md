# neural-colorization

[![Build Status](https://www.travis-ci.org/zeruniverse/neural-colorization.svg?branch=pytorch)](https://www.travis-ci.org/zeruniverse/neural-colorization)
![Environment](https://img.shields.io/badge/python-3.6-blue.svg)
![License](https://img.shields.io/github/license/zeruniverse/QQRobot.svg)

GAN for image colorization based on [Johnson's network structure](https://github.com/jcjohnson/fast-neural-style).

![Result](https://cloud.githubusercontent.com/assets/4648756/20504440/4067e0f6-affc-11e6-88e7-26de6f5c1cce.jpg)

## Setup

Install the following Python libraries:
+ numpy
+ scipy
+ Pytorch
+ scikit-image
+ Pillow
+ opencv-python


## Colorize images

```bash
#Download pre-trained model
wget -O model.pth "https://github.com/zeruniverse/neural-colorization/releases/download/1.1/G.pth"

#Colorize an image with CPU
python colorize.py -m model.pth -i input.jpg -o output.jpg --gpu -1

# If you want to colorize all images in a folder with GPU
python colorize.py -m model.pth -i input -o output --gpu 0
```

## Train your own model

Note: Training is only supported with GPU (CUDA).

### Prepare dataset

+ Download some datasets and unzip them into a same folder (saying `train_raw_dataset`). If the images are not in `.jpg` format, you should convert them all in `.jpg`s.
+ run `python build_dataset_directory.py -i train_raw_dataset -o train` (you can skip this if all your images are **directly** under the `train_raw_dataset`, in which case, just rename the folder as `train`)
+ run `python resize_all_imgs.py -d train` to resize all training images into `256*256` (you can skip this if your images are already in `256*256`)

### Optional preparation

It's highly recommended to train from my pretrained models. You can get both generator model and discriminator model from the GitHub Release:

```bash
wget "https://github.com/zeruniverse/neural-colorization/releases/download/1.1/G.pth"
wget "https://github.com/zeruniverse/neural-colorization/releases/download/1.1/D.pth"
```

It's also recommended to have a test image (the script will generate a colorization for the test image you give at every checkpoint so you can see how the model works during training).


### Training

The required arguments are training image directory (e.g. `train`) and path to save checkpoints (e.g. `checkpoints`)

```bash
python train.py -d train -c chekpoints
```

To add initial weights and test images:

```bash
python train.py -d train -c chekpoints --d_init D.pth --g_init G.pth -t test.jpg
```

More options are available and you can run `python train.py --help` to print all options.

For torch equivalent (no GAN), you can set option `-p 1e9` (set a very large weight for pixel loss). 

## Reference
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://github.com/jcjohnson/fast-neural-style)

## License

GNU GPL 3.0 for personal or research use. COMMERCIAL USE PROHIBITED.

Model weights are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
