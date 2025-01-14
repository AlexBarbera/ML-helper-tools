# ML-helper-tools

This repo contains helper functions and skeletons for training ML models in an attempt to avoid repeating code search.

The idea is to have a single repo to refer to for implementation and editing instead of writting from scratch, this is not supposed to be a script repo.

## Losses
 - **WassersteinLoss**
   - Earth-mover distance is a better metric for small permutations in image data compared to `MSELoss`.
   - [Original repo](https://github.com/jeanfeydy/geomloss/)


 - **PerceptualLoss**
   - Uses an encoder network to compare extracted features between target and generated, useful for SuperResolution. [paper](https://arxiv.org/pdf/1609.04802v5)


 - **TotalVariationLoss**
   - Jitters the image and calculates `MSE`.
   - Used in SR paper


 - **DeepEnergyLoss**
   - Maximizes difference between labels rather than set to `1` or `0`.
   - [Docs](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html)


 - **CycleGANLoss**
   - Extracted Cyclegan loss pipeline in case we want to abstract it to somthing different.

## Models
 - **CycleGAN**
   - Unaligned datasets for domain tranformation.
   - [Original repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)


 - **Siamese Network**
   - Classifies differences in latent space of encoder.
   - This is mine :)


 - **ResNet Generator**
   - Standard ResNet
   - [Original repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)



 - **MultiLayer Perceptron** (Linear and Convolutional)
 - ¯\\_(ツ)_/¯

## Utils
 - LightningWrapper for training
   - Allows to easily use distributed training and GPU/TPU usage.
   - [Pytorch Lightining](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)


 - **TripletDataset**
   - Handles tuples of 3 (anchor [reference], positive [simmilar], negative [different]).


 - **ABDataset**
   - Matches 2 classes of data in pairs.


 - **MulticlassDataset**
   - Matches N classes of data in pairs.


 - **ZCA Whitening**
   - Normalizes images so that covariance $`\sum`$ is the Identity matrix leading to decorrelated features.
   - According to the [paper](https://arxiv.org/pdf/1804.08450v1), it should be applied batch-wise.
