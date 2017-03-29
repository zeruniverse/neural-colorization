# neural-colorization
Feed-forward neural network for image colorization based on [Johnson's network structure](https://github.com/jcjohnson/fast-neural-style).

![Result](https://cloud.githubusercontent.com/assets/4648756/20504440/4067e0f6-affc-11e6-88e7-26de6f5c1cce.jpg)
  
## Setup  
```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson

#GPU acceleration
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
```
  
## Colorize images  
Assume you want to colorize image `input.jpg` and save resulting image as `output.png`  
  
```bash
#Download pre-trained model
wget -O model.t7 "https://github.com/zeruniverse/neural-colorization/releases/download/1.0/places2.t7"
#Colorize an image
th colorize.lua -model model.t7 -input_image input.jpg -output_image output.png -gpu 0
  
#If you want to colorize all images in a folder
mkdir -p output
th colorize.lua -model model.t7 -input_dir input -output_dir output -gpu 0
```

## Train your own model  
Suppose all your training data is in folder `train` and validation data is in folder `validation`.   
The python script recursively checks all image files and ignore all gray ones.  

```bash
python make_dataset.py --train_dir train --val_dir validation --output_file dataset.h5
th train.lua -h5_file dataset.h5 -checkpoint_name model -gpu 0
```
  
To compute the prediction error of your model in validation dataset, use `validation.lua`.  
```bash
th validation.lua -h5_file dataset.h5 -model model.t7 -gpu 0
```
  
## Reference  
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://github.com/jcjohnson/fast-neural-style)  
  
## License  
GNU GPL 3.0 for personal or research use. COMMERCIAL USE PROHIBITED.
