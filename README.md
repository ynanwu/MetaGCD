# Learning to Continually Learn in Generalized Category Discovery
This repository is the official implementation of [MetaGCD: Learning to Continually Learn in Generalized Category Discovery](https://arxiv.org/pdf/2308.11063.pdf).

<div align="center">
<img width="80%" alt="Method Overview" src="assets/overview.jpg">
</div>

## Requirements

The code was tested on python3.7 and CUDA10.1.

We recommend using conda environment to setup all required dependencies:

```setup
conda env create -f environment.yml
conda activate MetaGCD
```

If you have any problem with the above command, you can also install them by `pip install -r requirements.txt`.

Either of these commands will automatically install all required dependecies **except for the `torch-scatter` and `torch-geometric` packages** , which require a [quick manual install](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-binaries).

## Training
