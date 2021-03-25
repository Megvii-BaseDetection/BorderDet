# BorderDet

This project provides an implementation for "BorderDet: Border Feature for Dense Object Detection" (*ECCV2020 Oral*) on PyTorch.

For the reason that experiments in the paper were conducted using internal framework, this project reimplements them on cvpods and reports detailed comparisons below.

<center><img src="./playground/detection/coco/borderdet/intro/borderdet.png" width="700" align="middle"/></center>

## Requirements
* [cvpods](https://github.com/Megvii-BaseDetection/cvpods)


## Get Started

* install cvpods locally (requires cuda to compile)
```shell

python3 -m pip install 'git+https://github.com/Megvii-BaseDetection/cvpods.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/Megvii-BaseDetection/cvpods.git
python3 -m pip install -e cvpods

# Or,
pip install -r requirements.txt
python3 setup.py build develop
```

* prepare datasets
```shell
cd /path/to/cvpods
cd datasets
ln -s /path/to/your/coco/dataset coco
```

* Train & Test
```shell
git clone https://github.com/Megvii-BaseDetection/BorderDet.git
cd BorderDet/playground/detection/coco/borderdet/borderdet.res50.fpn.coco.800size.1x  # for example

# Train
pods_train --num-gpus 8

# Test
pods_test --num-gpus 8 \
    MODEL.WEIGHTS /path/to/your/save_dir/ckpt.pth # optional
    OUTPUT_DIR /path/to/your/save_dir # optional

# Multi node training
## sudo apt install net-tools ifconfig
pods_train --num-gpus 8 --num-machines N --machine-rank 0/1/.../N-1 --dist-url "tcp://MASTER_IP:port"



## Results on COCO
For your convenience, we provide the performance of the following trained models. All models are trained with 16 images in a mini-batch and frozen batch normalization. All model including X_101/DCN_X_101 will be released soon.

| Model | Multi-scale training | Multi-scale testing | Testing time / im | AP (minival) | Link |
|:--- |:--------------------:|:--------------------:|:-----------------:|:-------:|:---:|
| [FCOS_R_50_FPN_1x](https://github.com/Megvii-BaseDetection/BorderDet/blob/master/playground/detection/coco/fcos/fcos.res50.fpn.coco.800size.1x) | No | No | 54ms | 38.7 | [download](https://drive.google.com/file/d/1hcDobxvqolMwqj20BEAPikSMcz4NYZRx/view?usp=sharing)
| [BD_R_50_FPN_1x](https://github.com/Megvii-BaseDetection/BorderDet/blob/master/playground/detection/coco/borderdet/borderdet.res50.fpn.coco.800size.1x) | No | No | 60ms | 41.4 | [download](https://drive.google.com/file/d/1nhGA0TYtwGp_RMwPoZDAPbZ_TNL8-XCj/view?usp=sharing)
| [BD_R_101_FPN_1x](https://github.com/Megvii-BaseDetection/BorderDet/blob/master/playground/detection/coco/borderdet/borderdet.res101.fpn.coco.800size.2x) | Yes | No | 76ms | 45.0 | [download](https://drive.google.com/file/d/1LEbLZwP_9eKbpZXC52D5B_V85A4pr9eE/view?usp=sharing)
| [BD_X_101_32x8d_FPN_1x](https://github.com/Megvii-BaseDetection/BorderDet/blob/master/playground/detection/coco/borderdet/borderdet.x101.32x8d.fpn.coco.800size.2x) | Yes | No | 124ms | 45.6 | [download](https://drive.google.com/file/d/1Cd5xJCVdb1RPE1VAFAzCBXyLxcH315-f/view?usp=sharing)
| [BD_X_101_64x4d_FPN_1x](https://github.com/Megvii-BaseDetection/BorderDet/blob/master/playground/detection/coco/borderdet/borderdet.x101.64x4d.fpn.coco.800size.2x) | Yes | No | 123ms | 46.2 | [download](https://drive.google.com/file/d/15UH3PPQONv4nhHIDQGll0iHnuhmqwbAp/view?usp=sharing)
| [BD_DCNV2_X_101_32x8d_FPN_1x](https://github.com/Megvii-BaseDetection/BorderDet/blob/master/playground/detection/coco/borderdet/borderdet.dcnv2.x101.32x8d.fpn.coco.800size.2x) | Yes | No | 150ms | 47.9 | [download](https://drive.google.com/file/d/1xGnomS2rn2rayMrPxE_hpzbUQxMJ-eCN/view?usp=sharing)
| [BD_DCNV2_X_101_64x4d_FPN_1x](https://github.com/Megvii-BaseDetection/BorderDet/blob/master/playground/detection/coco/borderdet/borderdet.dcnv2.x101.64x4d.fpn.coco.800size.2x) | Yes | No | 156ms | 47.5 | [download](https://drive.google.com/file/d/1R6a7CzwHu8iXSENZXNrWXVwaAaV-oB5_/view?usp=sharing)



## Acknowledgement
cvpods is developed based on Detectron2. For more details about official detectron2, please check [DETECTRON2](https://github.com/facebookresearch/detectron2/blob/master/README.md).


## Contributing to the project
Any pull requests or issues are welcome.