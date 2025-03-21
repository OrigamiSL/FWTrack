# FWTrack

**Hierarchical Spatial-Temporal UAV Tracking with Three-Dimensional Wavelets**

This work is submitted to IEEE T-ITS.

GitHub maintainer: [Li Shen](https://github.com/OrigamiSL)

E-mail: shenli@buaa.edu.cn

Model Architecture
---------
![](./figure/FWTrack_overview.jpg)


Abstract
------------------

Visual tracking plays a vital role in modern intelligent transportation systems (ITSs) to sense traffic environments
and trace targets, wherein unmanned aerial vehicles (UAVs)
are commonly used for data collection. Currently, many existing trackers leverage spatial-temporal features to increase
their tracking capabilities. However, the utilization of spatial-temporal features normally involves considerable extra network
modules and time-consuming recursive deduction processes,
making these trackers impractical in ITSs. To address the
issue of low efficiency, we propose FWTrack, a novel tracker
that constructs hierarchical spatial-temporal features via three-dimensional wavelets and thus achieves efficient spatial-temporal
visual tracking. FWTrack employs the spatial maximal-overlap
discrete wavelet transform (MODWT) smooths to reinforce its
feature extraction ability in an approximately parameter-free
manner and applies the wavelet coefficients of the temporal
MODWT to adaptively separate static backgrounds from the
attention modules of FWTrack, thereby speeding up the model.
Moreover, the window-wise attention technique, which prevails
in many computer vision tasks but is rarely applied in visual
tracking, is adopted and enhanced in FWTrack to further
increase its efficiency. Extensive experiments conducted on seven
benchmarks demonstrate that FWTrack achieves state-of-the-art
tracking accuracy and efficiency.

Performance
----------------------

|            Benchmarks              |       FWTrack-288     |  
|:-------------------------------:|:-----------------------:|
| GOT-10k (AO / SR 0.5 / SR 0.75) |   76.1	/ 85.5	/ 73.3  |  
|    LaSOT (AUC / Norm P / P)     |   71.7	/ 81.6	/ 78.9 
|  LaSOT_ext (AUC / Norm P / P)   |   52.0 / 62.5 / 58.9   |     
|          UAV123 (AUC)           |          72.6         |     
|          NFS (AUC)           |          68.4        | 
|          TNL2K (AUC)           |          59.8        | 


## Install the environment

We use the Anaconda (**CUDA 11.3**) to build the environment (**Linux**).
Before you run our project, we suggest that you install and activate the environment by running the commands below. 
```
conda env create -f FWTrack_env_cuda113.yaml

conda activate FWTrack
```
Partial paramount site-packages requirements are listed below:
- `python == 3.9.7` 
- `pytorch == 1.11.0`
- `torchvision == 0.12.0`
- `matplotlib == 3.5.1`
- `numpy == 1.21.2`
- `pandas == 1.4.1`
- `pyyaml == 6.0`
- `scipy == 1.7.3`
- `scikit-learn == 1.0.2`
- `tqdm == 4.63.0`
- `yaml == 0.2.5`
- `opencv-python == 4.5.5.64`
- `pywavelets == 1.3.0`

## Set project paths and dataset paths

Before you start the data preparation, you are advised to modify project paths and dataset paths by editing these two files.
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```
The train path should look like this:
```
# paths about training
import os
class EnvironmentSettings:
    def __init__(self):
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        # project path
        self.workspace_dir = project_path 
        self.tensorboard_dir = os.path.join(project_path, 'tensorboard')  
        self.pretrained_networks = os.path.join(project_path, 'pretrained_networks')  

        # dataset path
        self.lasot_dir = ${LASOT_TRAIN_PATH}
        self.got10k_dir = ${GOT10K_TRAIN_PATH}
        self.trackingnet_dir = ${TRACKINGNET_TRAIN_PATH}
        self.coco_dir = ${COCO_TRAIN_PATH}
```
The test path should look like this:
```
# paths about testing
from lib.test.evaluation.environment import EnvSettings
import os
# Set your local paths here.
def local_env_settings():
    settings = EnvSettings()

    # root path
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # dataset path
    settings.got10k_path = ${GOT10K_TEST_PATH} 
    settings.lasot_extension_subset_path = ${LASOTEXT_PATH}
    settings.lasot_path = ${LASOT_TEST_PATH}
    settings.trackingnet_path = ${TRACKINGNET_TEST_PATH}
    settings.uav_path = ${UAV_PATH}
    settings.tnl2k_path = ${NFS_PATH}
    settings.nfs_path = ${TNL2K_PATH}
    settings.network_path = os.path.join(project_path,'output/test/networks' )   # Where tracking networks are stored.
    
    # save path
    settings.prj_dir = project_path
    settings.result_plot_path = os.path.join(project_path,'output/test/result_plots') 
    settings.results_path = os.path.join(project_path,'output/test/tracking_results') 
    settings.save_dir = os.path.join(project_path,'output')  
    
    return settings
```
## Data
You can acquire the raw data of all datasets from the links listed below:
GOT-10k dataset can be acquired at: [GOT-10k](http://got-10k.aitestunion.com/downloads).
LaSOT dataset can be acquired at: [LaSOT](https://onedrive.live.com/?authkey=%21AMZfYsa%2DWN%5Fd6lg&id=83EEFE32EECC7F4B%2133324&cid=83EEFE32EECC7F4B&parId=root&parQt=sharedby&o=OneUp).
TrackingNet dataset can be acquired at: [TrackingNet](https://drive.google.com/drive/u/0/folders/1gJOR-r-jPFFFCzKKlMOW80WFtuaMiaf6).
LaSOTEXT dataset can be acquired at: [LaSOTEXT](https://onedrive.live.com/?authkey=%21AL6OYePAAteZeuw&id=83EEFE32EECC7F4B%2133323&cid=83EEFE32EECC7F4B&parId=root&parQt=sharedby&o=OneUp).
UAV123 dataset can be acquired at: [UAV123](https://drive.google.com/file/d/0B6sQMCU1i4NbNGxWQzRVak5yLWs/view?resourcekey=0-IjwQcWEzP2x3ec8kXtLBpA).
NFS dataset can be acquired at: [NFS](https://ci2cv.net/nfs/index.html).
TNL2K dataset can be acquired at: [TNL2K](https://sites.google.com/view/langtrackbenchmark).

## Data Preparation
After you acquire and unzip the raw data of all datasets, please separately place them in corresponding folders, e.g., putting the train split of LaSOT under `${LASOT_TRAIN_PATH}`. Note that the annotations of the 30FPS version in the NFS dataset can be obtained by running `./lib/test/evaluation/process_nfs.py`, then you can find the preprocessed annotations in `${NFS_PATH}/NAME/NAME/30/NAME_sampled.txt`.

The file tree shall look like this:
```
    #Training Split:
    -- ${LASOT_TRAIN_PATH}
        |-- airplane
        |-- basketball
        |-- bear
        ...
    -- ${GOT10K_TRAIN_PATH}
        |-- train
    -- ${COCO_TRAIN_PATH}
        |-- annotations
        |-- images
    -- ${TRACKINGNET_TRAIN_PATH}
        |-- TRAIN_0
        |-- TRAIN_1
        ...
        |-- TRAIN_11
```
```
    #Testing Split:
    -- ${LASOT_TEST_PATH}
        |-- airplane-1
        |-- airplane-9
        |-- airplane-13
        ...
    -- ${GOT10K_TEST_PATH}
        |-- test
    -- ${TRACKINGNET_TEST_PATH}
        |-- TEST
    -- ${LASOTEXT_PATH}
        |-- atv
        |-- badminton
        |-- cosplay
        ...
    -- ${UAV_PATH}
        |-- anno
        |-- data_seq
    -- ${NFS_PATH}
        |-- airboard_1
        |-- |-- airboard_1
        |-- |-- |-- 240
        |-- |-- |-- 30
        |-- |-- |-- |-- airboard_1
        |-- |-- |-- |-- airboard_1_sampled.txt
        ...
    -- ${TNL2K_PATH}
        |-- advSamp_Baseball_game_002-Done
        ...
```


## Weight source (FWTrack_288_full):

You can download the model weights from [MEGA](https://mega.nz/file/iMIFXBgA#rAIxOK5VOTV2n_hs9IVfrCRJRM51TDMUrRCas7HJxBM).

Put the model weight you download in `./test_checkpoint.` The file tree shall look like this:
```
   ${PROJECT_ROOT}
    |-- test_checkpoint
    |   |-- FWTrack_best.pth.tar
```
## Raw results
You can download the raw results in `$PROJECT_ROOT$/raw_result`. The file tree shall look like this:
```
   ${PROJECT_ROOT}
    |-- raw_result
    |   |-- lasot.zip
    |   |-- lasotext.zip
    |   |-- uav.zip
    |   |-- nfs.zip
    |   |-- tnl2k.zip
```

## Download the pre-trained weight
Download pre-trained [MAE ViT-Base weight](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` 

## Train OTETrack

For full dataset training (GOT10K, LaSOT, TrackingNet, COCO)
```
python tracking/train.py --script fwtrack --config fwtrack_256_full --save_dir ./output --mode single --nproc_per_node 4 --use_wandb 0

```

For GOT10K training (GOT10K)
```
python tracking/train.py --script fwtrack --config fwtrack_256_got --save_dir ./output --mode single --nproc_per_node 4 --use_wandb 0

```

## Test and evaluate on benchmarks

You can test and evaluate each benchmark respectively by running the commands below (Notice that GOT10K is evaluated on [GOT10K](http://got-10k.aitestunion.com/) and TrackingNet is evaluated on [TrackingNet](https://eval.ai/web/challenges/challenge-page/1805/overview)). You can find all the commands in `./script/train_test.sh`.

- GOT10K-test
```
python tracking/test.py fwtrack fwtrack_256_got --dataset got10k_test --test_checkpoint 'THE PATH OF YOUR TRAINED CHECKPOINT' --threads 0 --num_gpus 1
python lib/test/utils/transform_got10k.py --tracker_name fwtrack --cfg_name fwtrack_256_got
```
- LaSOT
```
python tracking/test.py fwtrack fwtrack_256_full --dataset lasot --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1
```
- TrackingNet
```
python tracking/test.py fwtrack fwtrack_256_full --dataset trackingnet --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1
python lib/test/utils/transform_trackingnet.py --tracker_name fwtrack --cfg_name fwtrack_256_full
```
- LaSOText
```
python tracking/test.py fwtrack fwtrack_256_full --dataset lasot_extension_subset --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1
```

- UAV123
```
python tracking/test.py fwtrack fwtrack_256_full --dataset uav --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1
```

- NFS
```
python tracking/test.py fwtrack fwtrack_256_full --dataset nfs --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1
```

- TNL2K
```
python tracking/test.py fwtrack fwtrack_256_full --dataset tnl2k --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1
```

## Others

If you would like to train GOT-10k by yourself, you may find that 200 epochs are not enough for the convergence, which can be manifested by the validation result. Then, you can try 400 epochs and decays the weight at the 300-th epoch.

## Acknowledgement

This codebase is implemented on the following projects. We really appreciate their authors for the open-source works!

- [ARTrack](https://github.com/miv-xjtu/artrack) [[related paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_Autoregressive_Visual_Tracking_CVPR_2023_paper.pdf)]
- [SeqTrack](https://github.com/microsoft/VideoX/tree/master/SeqTrack) [[related paper](https://arxiv.org/abs/2304.14394)]

**This project is not for commercial use. For commercial use, please contact the author.**

## Citation

If any part of our work helps your research, please consider citing us and giving a star to our repository.

## Contact

If you have any questions or concerns, feel free to open issues or directly contact me through the ways on my GitHub homepage.
