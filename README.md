# ML-helper-tools

This repo contains helper functions and skeletons for training ML models in an attempt to avoid repeating code search.

The idea is to have a single repo to refer to for implementation and editing instead of writting from scratch, or utillities that are not easily available.

**This is not supposed to be a script repo.**

Also installable as a pip module:

`pip install git+ssh://git@github.com:AlexBarbera/ML-helper-tools.git`
&nbsp;

## Losses
 ### **WassersteinLoss**
   - Earth-mover distance is a better metric for small permutations in image data compared to `MSELoss`.
   - [Original repo](https://github.com/jeanfeydy/geomloss/)


 ### **PerceptualLoss**
   - Uses an encoder network to compare extracted features between target and generated, useful for SuperResolution. [paper](https://arxiv.org/pdf/1609.04802v5)


 ### **TotalVariationLoss**
   - Jitters the image and calculates `MSE`.
   - Used in SR paper


 ### **DeepEnergyLoss**
   - Maximizes difference between labels rather than set to `1` or `0`.
   - [Docs](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html)


 ### **CycleGANLoss**
   - Extracted Cyclegan loss pipeline in case we want to abstract it to somthing different.


 ### **WordTreeLoss**
   - Based on [YOLO9000 paper](https://arxiv.org/pdf/1612.08242) for hierarchical classification.
   - **Also utility class to use label hierarchy outside of training.**

 ### **BinaryLabelSmoothing Wrapper**
   - Smoothes a one-hot vector by taking a bit of the top label and spreading it over the rest of the classes.
   - Based on [Improved Techniques for training GANs](https://arxiv.org/pdf/1606.03498).
   - Technically the smoothing would be $0 \rightarrow \frac{\epsilon}{k}$ and $1 \rightarrow \frac{k-1}{k}$
   - While not a loss itself it is a loss wrapper.

&nbsp;

## Models
 ### **CycleGAN**
   - Unaligned datasets for domain tranformation.
   - [Original repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)


 ### **Siamese Network**
   - Classifies differences in latent space of encoder.
   - This is mine :)


 ### **ResNet Generator**
   - Standard ResNet
   - [Original repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)



 ### **MultiLayer Perceptron** (Linear and Convolutional)
   - ¯\\_(ツ)_/¯

&nbsp;

## Utils
 ### LightningWrapper for training
   - Allows to easily use distributed training and GPU/TPU usage.
   - [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)
   - Scaling functions
     - MinMax
     - L2-norm
     - Tensor to 8bit


 ### **TripletDataset**
   - Handles tuples of 3 (anchor [reference], positive [simmilar], negative [different]).


 ### **ABDataset**
   - Matches 2 classes of data in pairs.


 ### **MulticlassDataset**
   - Matches N classes of data in pairs.


 ### **ZCA Whitening**
   - Normalizes images so that covariance $`\sum`$ is the Identity matrix leading to decorrelated features.
   - According to the [paper](https://arxiv.org/pdf/1804.08450v1), it should be applied batch-wise.
