
# PaDiM
###### _A Patch Distribution Modeling Framework for Anomaly Detection and Localization_

This is an **unofficial** implementation of the paper *PaDiM:  a Patch Distribution Modeling Framework for Anomaly Detection and Localization* available on [arxiv](http://arxiv.org/abs/2011.08785). 

### Features

The key features of this implementation are: 

- Constant memory footprint - training on more images does not result in more memory required
- Resumable learning - the training step can be stopped and then resumed with inference in-between
- Limited dependencies - apart from PyTorch, Torchvision and Numpy 

### Installation

```
git clone https://github.com/Pangoraw/PaDiM.git
```

### Getting started

#### Training

#### Testing

### Acknowledgements

This implementation was built on the work of:

- [The original *PaDiM* paper](http://arxiv.org/abs/2011.08785)
- [taikiinoue45/PaDiM](https://github.com/taikiinoue45/PaDiM)'s implementation - see section [Features](#features) for the main differences.