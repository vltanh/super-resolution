Super Resolution for the internship at Coordinated Science Laboratory of UIUC

# Requirements

```
torch
torchvision
tqdm
Pillow
matplotlib

```

# Usage

## Data

For this project, we will be using the DIV2K dataset. You can download it [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and put it in the ```data``` folder following this structure.

```
data/
  DIV2K/
    DIV2K_HR/
      train/
        0001.png
        ...
      val/
        0801.png
        ...
    DIV2K_LR-8x/
      train/
        0001.png
        ...
      val/
        0801.png
        ...
```

In addition, make sure there are two folders ```output``` (to store the output images) and ```models``` (to store the pretrained models).

## Training

To train, run

```
python train.py
```

## Testing

To test, run

```
python eval.py
```

# References

[ONNX Sub-pixel Super Resolution](https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016)
