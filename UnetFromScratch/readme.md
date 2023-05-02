# U-net from scratch

This is a complete implementation of U-net architecture from the original [paper](https://arxiv.org/pdf/1505.04597.pdf). The complete architecture is written in `model_configs.json` file. Unet class is impletemented in `unet.py` which takes model configs along with initial `input_channels` in the image and `num_classes` as arguments and outputs a convolution layer like array. It is then compared with the available mask to optimize the parameters and minimize loss. `Binary Cross entroy with logit loss` loss is used to evaluate the model. Note: This loss works effectively for multi-class segmentations as well. 

`train.py` is the final implementation of unet model. It takes the data and process images and masks into dataloader object for pytorch. Model configs provide the Unet architecture. For a batch of images and masks are passed to the model and are evaluated to minimize the loss. 


An interactive plot shows the loss for each batch as well as each epoch. 

Currently, the model only takes 2D images. Data loader for 3D images is in making and will be available soon. 