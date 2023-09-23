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
We provide the training script for the following 3 datasets from the NCD benchmark: [CIFAR10](https://pytorch.org/vision/stable/datasets.html)
, [CIFAR100](https://pytorch.org/vision/stable/datasets.html) and `Tiny-ImageNet`. To train the models in the paper, run the following commands:

```Training
python run.py --dataset <dataset>
```

The data will be automatically downloaded to the data folder.

## Evaluation

To evaluate trained models, run:

```eval
python run.py --dataset <dataset> --data_dir <path to data folder> --test
```

## Citation
If you find this codebase useful in your research, consider citing:
```
@inproceedings{
    wu2023metagcd,
    title={MetaGCD: Learning to Continually Learn in Generalized Category Discovery},
    author={Yanan Wu and Zhixiang Chi and Yang Wang and and Songhe Feng},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2023}
}
```
