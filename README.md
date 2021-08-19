
# PaDiM
###### _A Patch Distribution Modeling Framework for Anomaly Detection and Localization_

This is an **unofficial** re-implementation of the paper *PaDiM:  a Patch Distribution Modeling Framework for Anomaly Detection and Localization* available on [arxiv](http://arxiv.org/abs/2011.08785). 

### Features

The key features of this implementation are: 

- Constant memory footprint - training on more images does not result in more memory required
- Resumable learning - the training step can be stopped and then resumed with inference in-between
- Limited dependencies - apart from PyTorch, Torchvision and Numpy 

### Variants

This repository also contains variants on the original PaDiM model:

- [PaDiMSVDD](https://github.com/Pangoraw/PaDiM/blob/release/padim/padim_svdd.py) uses a [Deep-SVDD](http://proceedings.mlr.press/v80/ruff18a.html) model instead of a multi-variate Gaussian distribution for the normal patch representation.
- [PaDiMShared](https://github.com/Pangoraw/PaDiM/blob/release/padim/padim_shared.py) shares the multi-variate Gaussian distribution between all patches instead of learning it only for specific coordinates.
- [PaDiMNVP](https://github.com/Pangoraw/PaDiM/blob/release/padim/panf.py) uses a normalizing flow to transform the embedding vectors in a Gaussian distribution.

### Installation

```
git clone https://github.com/Pangoraw/PaDiM.git padim
```

### Getting started

#### Training

```python
from torch.utils.data import DataLoader
from padim import PaDiM

# i) Initialize
padim = PaDiM(num_embeddings=100, device="cpu", backbone="resnet18") 

# ii) Create a dataloader producing image tensors
dataloader = DataLoader(...)

# iii) Consume the data to learn the normal distribution
# Use PaDiM.train(...)
padim.train(dataloader)

# Or PaDiM.train_one_batch(...)
for imgs in dataloader:
	padim.train_one_batch(imgs)
```
#### Testing

With the same `PaDiM` instance as in the [Training](#training) section:

```python
for new_imgs in test_dataloader:
	distances = padim.predict(new_imgs) 
	# distances is a (n * c) matrix of the mahalanobis distances
	# Compute metrics...
```

### Acknowledgements

This implementation was built on the work of:

- [The original *PaDiM* paper](http://arxiv.org/abs/2011.08785)
- [taikiinoue45/PaDiM](https://github.com/taikiinoue45/PaDiM)'s implementation - see section [Features](#features) for the main differences.
