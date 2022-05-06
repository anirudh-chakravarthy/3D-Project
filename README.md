# Constrained Humanification: Improving Multi-Person Reconstruction Using Temporal Constraints
Code repository for [CMU 16-889: Learning for 3D Vision](https://learning3d.github.io/) Course Project.


![teaser](assets/teaser.png)

## Contents

Our repository includes training/testing/demo code for our paper. Additionally, you might find useful some parts of the code that can also be used in a standalone manner. More specifically:

[Neural Mesh Renderer](./neural_renderer):
Fast implementation of the original [NMR](https://hiroharu-kato.com/projects_en/neural_renderer.html).

[SDF](./sdf):
CUDA implementation of the SDF computation and our SDF-based collision loss.

[SMPLify 3D fitting](./misc/smplify-x):
Extension of [SMPLify](http://files.is.tue.mpg.de/black/papers/BogoECCV2016.pdf) that offers the functionality of fitting the SMPL model to 3D keypoints.

## Installation instructions
This codebase was adopted from early version of mmdetection and mmcv. Users of this repo are highly recommended to
read the readme of [mmcv](./mmcv/README.rst) and [mmdetection](./mmdetection/README.md) before using this code.

To install mmcv and mmdetection:
```bash
conda env create -f environment.yml
cd neural_renderer/
python3 setup.py install
cd ../mmcv
python3 setup.py install
cd ../mmdetection
./compile.sh
python setup.py develop
cd ../sdf
python3 setup.py install
```

## Fetch data
Download [our model data](https://drive.google.com/file/d/1y5aKzW9WL42wTfQnv-JJ0YSIgsdb_mJn/view?usp=sharing) and place them under `mmdetection/data`.
This includes the model checkpoint and joint regressors.
You also need to download the mean SMPL parameters from [here](https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/neutral_smpl_mean_params.h5).
Besides these files, you also need to download the SMPL model. You will need the [neutral model](http://smplify.is.tue.mpg.de) for training, evaluation and running the demo code.
Please go to the websites for the corresponding projects and register to get access to the downloads section. In case you need to convert the models to be compatible with python3, please follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools).

After finishing with the installation and downloading the necessary data, you can continue with running the demo/evaluation/training code.

## Run demo code

We provide code to evaluate our pretrained model on a folder of images by running:

```bash
python3 tools/demo.py --config=configs/smpl/tune.py --image_folder=demo_images/ --output_folder=results/ --ckpt data/checkpoint.pt
```

We also provide code to run our model on a video sequence with pre-computed optical flow:
```bash
python3 tools/demo_video.py --config configs/smpl/flow.py --image_folder data/posetrack2018/images/val/002374_mpii_test --flow_folder data/posetrack2018/optical_flow/val/002374_mpii_test --output_folder posetrack_add_flow/002374_mpii_test --ckpt work_dirs/add_flow/latest.pth
```

## Prepare datasets
Please refer to [DATASETS.md](./DATASETS.md) for the preparation of the dataset files. We only train on PoseTrack 2018 and evaluate on MuPoTS-3D Dataset. 

## Generating optical flow
We provide code to generate optical flow on a dataset consisting of several image sequences:

```bash
cd optical_flow
python optical_flow_prediction.py --images_folder_path ../mmdetection/data/posetrack2018/images/val/ --optical_flow_save_path ../mmdetection/data/posetrack2018/optical_flow/val/
```

## Run evaluation code
Besides the demo code, we also provide code to evaluate our models on the datasets we employ for our quantitative evaluation. Before continuing, please make sure that you follow the [preparation of test sets](DATASETS.md).

You could use either our pretrained checkpoint or the model trained by yourself to evaluate on Panoptic, MuPoTS-3D, Human3.6M and PoseTrack.

Example usage:
```bash
cd mmdetection
python3 tools/full_eval.py configs/smpl/flow.py mupots --ckpt ./work_dirs/add_flow/latest.pth
```

Running the above command will generate the 3D poses on the MuPoTS-3D Dataset. 
The ```mupots``` option can be replaced with other dataset or sequences based on the type of evaluation you want to perform:
- `haggling`: haggling sequence of Panoptic
- `mafia`: mafia sequence of Panoptic
- `ultimatum`: ultimatum sequence of Panoptic
- `haggling`: haggling sequence of Panoptic
- `mupots`: MuPoTS-3D dataset
- `posetrack`: PoseTrack dataset

Regarding the evaluation:
- For Panoptic, the command will compute the MPJPE for each sequence.
- For MuPoTS-3D, the command will save the results to the `work_dirs/add_flow/mupots.mat` which can be taken as input for official MuPoTS-3D test script.
- For H36M, the command will compute P1 and P2 for test set.

To evaluate on MuPoTS-3D, we use a [Python reimplementation](https://github.com/ddddwee1/MuPoTS3D-Evaluation. Please follow the instructions in the `MuPoTS3D-Evaluation` directory.

## Run training code

Please make sure you have prepared all [datasets](./DATASETS.md) before running our training script.
The training of our model would take four phases, pretrain -> baseline -> fine tuning -> flow fine-tuning. We prepared four configuration files under `mmdetection/configs/smpl/`. In order to experiment with fusion methods for optical flow, change [L20-21](https://github.com/anirudh-chakravarthy/3D-Project/blob/optical-flow-3d/mmdetection/configs/smpl/flow.py#L20-L21) to the appropriate setting.
To train our model from scratch:

```bash
cd mmdetection
# Phase 1: pretraining
python3 tools/train.py configs/smpl/pretrain.py --create_dummy
while true:
do
    python3 tools/train.py configs/smpl/pretrain.py
done
# We could move to next phase after training for 240k iterations

# Phase 2: baseline
python3 tools/train.py configs/smpl/baseline.py --load_pretrain ./work_dirs/pretrain/latest.pth
while true:
do
    python3 tools/train.py configs/smpl/baseline.py 
done
# We could move to next phase after training for 180k iterations

# Phase 3: Fine-tuning
python3 tools/train.py configs/smpl/tune.py --load_pretrain ./work_dirs/baseline/latest.pth
while true:
do
    python3 tools/train.py configs/smpl/tune.py 
done
# It could be done after 100k iterations of training

# Phase 4: Optical Flow Fine-tuning
python3 tools/train.py configs/smpl/flow.py --load_pretrain ./work_dirs/tune/latest.pth
while true:
do
    python3 tools/train.py configs/smpl/flow.py 
done
# It could be done for 5-20 epochs
```

All the checkpoints, evaluation results and logs would be saved to `./mmdetection/work_dirs/` + `pretrain|baseline|tune` respectively. For optical flow, the logs are stored to `./mmdetection/work_dirs/` in a directory named based on the fusion method and fusion input. Refer to [L313](https://github.com/anirudh-chakravarthy/3D-Project/blob/optical-flow-3d/mmdetection/configs/smpl/flow.py#L313) for more details.
Our training program will save the checkpoints and restart every 50 mins. You could change the `time_limit` in the configurations files to something more convenient

<!-- ## Citing
If you find this code useful for your research or the use data generated by our method, please consider citing the following paper:

	@Inproceedings{jiang2020mpshape,
	  Title          = {Coherent Reconstruction of Multiple Humans from a Single Image},
	  Author         = {Jiang, Wen and Kolotouros, Nikos and Pavlakos, Georgios and Zhou, Xiaowei and Daniilidis, Kostas},
	  Booktitle      = {CVPR},
	  Year           = {2020}
	} -->

## Acknowledgements

This code uses [SMPL R-CNN](https://github.com/JiangWenPL/multiperson), ([mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection)) as backbone.
We gratefully appreciate the impact these libraries had on our work.
