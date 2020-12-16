# CSE599-fianl-project

## Introduction
This project is aiming for provide ground truth for [CRUW dataset](https://www.cruwdataset.org/introduction) built by Information Processing Lab @UWECE. The CRUW dataset is a autonomous driving senerio dataset with dual-camera and dual radar sensor setup. Which aming for sensor fusion reseach and object detection by radar. And also this dataset will be publish soon and a challenge for radar-camera sensor fusion will be held. My works is to preprocess both the radar and image and then provide image base object detection groud truth result as benchmark for radar process algorism. The object detection base on Detectron is the main part for this final project. I use a new dataset called [nuImages](https://www.nuscenes.org/nuimages) to pretrain my benchmark model.


## Related Work

[Detectron2](https://github.com/facebookresearch/detectron2) 

Detectron 2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms. It is a ground-up rewrite of the previous version, Detectron, and it originates from maskrcnn-benchmark. This object detection algorithms is powered by the PyTorch deep learning framework,and also includes more features such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend, DeepLab, etc. I use mask R-CNN part of this project, and modify the source code to custom dataset.

To install Dectron2:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

[nuImages](https://www.nuscenes.org/nuimages) and [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) 

Instead of using KITTI, nuImages is the latest dataset for autonomous driving. This dataset is extracted and re-organize from the orginal nuScense dataset. The annotating the 1.4 million images in nuScenes with 2d annotations would have led to highly redundant annotations, which are not ideal for training an object detector. So the nuImages dataset set out to label a more varied large-scale image dataset from nearly 500 logs (compared to 83 in nuScenes) of driving data. 
The seniero and quaility are similar to the CRUW we collected. And also, the metadata and annotation is extract from [nuScenes](https://www.nuscenes.org/nuscenes) dataset, which have more information in the relational database.

This devkit is used in this project to load nuImage dataset. And also, when we desgined our dataset format, we took some schema built idea from this dataset.
To use nuScenes devkit:
```
pip install nuscenes-devkit
```


[CRUW dataset](https://www.cruwdataset.org/introduction)

CRUW is a camera-radar dataset for autonomous vehicle applications collected eariler this year by IPL@UWECE, we collect these data from E18 parking lot, road in side UW, city scenes and I-5 highway. It is a good resource for researchers to study FMCW radar data, that has high potential in the future autonomous driving. We will publish this dataset soon.Our dataset contains a pair of stereo cameras and 2 77GHz FMCW radar antenna array, both the camera and radar are calibrated and sychronized. During the last summer, I made the preprocess of this dataset, time-syschronize the camera and radar,and transfer the FMCW radar data to Range-Angle-Doppler(RAD) format



## Approach
**1. CRUW dataset annotation format**

  My first attribute to this project is to develope a annotation tool kit for our own dataset annotation format. The annotation writer and loader parts of project is done in this summer 2020. The development kit is in this repositories [CRUW devkit](https://github.com/yizhou-wang/cruw-devkit). Also, I made a annotation visualization tool for the inference. This part of the work is in [anno.py](https://github.com/yizhou-wang/cruw-devkit/blob/461d91b695b44bc8d6139942c60d584947e40886/scripts/anno.py) and [anno_loader.py](https://github.com/yizhou-wang/cruw-devkit/blob/master/scripts/anno_loader.py) in cruw devkit.
  The following picture shows result of the anno_loader tool. This tool can read the config file for different dataset configuration and show both detection result in image and radar domain.
  ![anno_loader](/pix/IMG_2052.PNG)
To use cruw devkit:
  
Create a new conda environment.
```
conda create -n cruw-devkit python=3.6
```

Run setup tool for this devkit.
```
conda activate cruw-devkit
pip install -e .
```
The annotations provided by CRUW dataset include the following:
```
{
  "dataset": "CRUW",
  "date_collect": "2019_09_29",
  "seq_name": "2019_09_29_onrd000",
  "n_frames": 1694,
  "fps": 30,
  "sensors": "C2R2",                  // <str>: "C1R1", "C2R1", "C2R2"
  "view": "front",                    // <str>: "front", "right-side"
  "setup": "vehicle",                 // <str>: "cart", "vehicle"
  "metadata": [
    {  // metadata for each frame
      "frame_id": 0,
      "cam_0": {
        "folder_name": "images_0",
        "frame_name": "0000000000.jpg",
        "width": 1440,
        "height": 864,
        "n_objects": 5,
        "obj_info": {
          "anno_source": "human",     // <str>: "human", "mrcnn", etc.
          "categories": [],           // <str> [n_objects]: category names
          "bboxes": [],               // <int> [n_objects, 4]: xywh
          "scores": [],               // <float> [n_objects]: confidence scores [0, 1]
          "masks": [],                // <rle_code> [n_objects]: instance masks
          "visibilities": [],         // <float> [n_objects]: [0, 1]
          "truncations": [],          // <float> [n_objects]: [0, 1]
          "translations": []          // <float> [n_objects, 3]: xyz(m)
        }
      },
      "radar_h": {
        "folder_name": "radar_chirps_win_RISEP_h",
        "frame_name": "000000.npy",
        "range": 128,
        "azimuth": 128,
        "n_chirps": 255,
        "n_objects": 3,
        "obj_info": {
          "anno_source": "human",     // <str>: "human", "co", "crf", etc.
          "categories": [],           // <str> [n_objects]: category names
          "centers": [],              // <float> [n_objects, 2]: range(m), azimuth(rad)
          "center_ids": [],           // <int> [n_objects, 2]: range indices, azimuth indices
          "scores": []                // <float> [n_objects]: confidence scores [0, 1]
        }
      },
    {...}
  ]
}
```
**2. Convert nuImages dataset to CRUW dataset format**

The nuImages dataset have more attributes than our dataset, and also the nuImages's categories is in detail, which is meaning less for our Camera-Radar Fusion (CRF) annotation(The information extracted from radar can only provide accuracy location and velocity information, the feature of objects are compressed).This part of the code can be access from [nuimages.py](https://github.com/TedSongjh/CSE599-fianl-project/blob/main/nuimages.py).

I use the nuScence build in devkit to load nuImages dataset and convert the categories use a mapping function, read all the relational data and transfer the metadata as a dict. For the segmantation part, the orignal segmantation format is a single map with category IDs for each instance, I convert the segmantation to each map per object, which can help me with futher fusion in objects.

And also, in nuImages, the cyclist are not seperated into different kind of pedestrain, but we want to merge the cyclist and the vehicle.cycle, so I read the attribution annotation and the bicycle with rider will be train as different category. After the training, I can relate the human.pedestrian with the vehicle.cycle.withrider by identify the corss part in bounding box and segmantation, and merge these bounding box and segmantation together. Same idea will be implement on vehicle.trailer and vehicle.car. But we don't have enough trailer object to train for now, this part will be added in future.

The categories mapping from nuImages to CRUW is:
nuImages Category | CRUW Category
------------ | -------------
animal	|	-
human.pedestrian.adult	|	human.pedestrian
human.pedestrian.child	|	human.pedestrian
human.pedestrian.construction_worker	|	human.pedestrian
human.pedestrian.personal_mobility	|	human.pedestrian
human.pedestrian.police_officer	|	human.pedestrian
human.pedestrian.stroller	|	human.pedestrian
human.pedestrian.wheelchair	|	human.pedestrian
movable_object.barrier	|	-
movable_object.debris	|	-
movable_object.pushable_pullable	|	-
movable_object.trafficcone	|	-
static_object.bicycle_rack	|	-
vehicle.bicycle(without attribute: without_rider)	|	vehicle.cycle
vehicle.bicycle(without attribute: with_rider)	|	vehicle.cycle.withrider
vehicle.bus.bendy	|	vehicle.bus
vehicle.bus.rigid	|	vehicle.bus
vehicle.car	|	vehicle.car
vehicle.construction	|	vehicle.car
vehicle.emergency.ambulance	|	vehicle.car
vehicle.emergency.police	|	vehicle.car
vehicle.motorcycle(without attribute: without_rider)	|	vehicle.cycle
vehicle.motorcycle(with attribute: with_rider)	|	vehicle.cycle.withrider
vehicle.trailer	|	vehicle.truck
vehicle.truck	|	vehicle.truck
flat.drivable_surface	|	-
flat.ego	|	-

**3. Use Custom datasets on Detectron2**
After made the dataset reader, I register the nuimages_test and nuimages_train dataset and metadata in [builtin.py](https://github.com/TedSongjh/CSE599-fianl-project/blob/main/builtin.py). Because the nuImages don't have built in evaluator. I choose to use COCO InstanceSegmentation evaluator in the following part, so I load these two dataset by COCO format. So I have to convert the CRUW dataset format to COCO format, by changing object information schema, segmantation map to bitmask and bounding box format. Also, because CRUW dataset sensor setup only have dual camera facing front, I filter out all the samples facing other direction in nuImages. This part is also in [nuimages.py](https://github.com/TedSongjh/CSE599-fianl-project/blob/main/nuimages.py).The instances detail information is in the chart below.

|   category    | #instances   |   category    | #instances   |   category    | #instances   |
|:-------------:|:-------------|:-------------:|:-------------|:-------------:|:-------------|
| human.pedes.. | 2983         |  vehicle.car  | 4530         |  vehicle.bus  | 189          |
| vehicle.truck | 722          | vehicle.cycle | 483          | vehicle.cyc.. | 0            |
|               |              |               |              |               |              |
|     total     | 8907         |               |              |               |              |

**4.Train nuImages use Mask R-CNN**

```
./detectron2/tools/train_net.py   --config-file ../configs/NuImages-RCNN-FPN.yaml   --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

To train the dataset on Detectron2 I modify the base mask-RCNN architeture config, which use ResNet FPN backbone. 

The detail of this archetecuture can be found in [NuImages-RCNN-FPN.yaml](https://github.com/TedSongjh/CSE599-fianl-project/blob/main/configs/NuImages-RCNN-FPN.yaml)
```
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"
  BACKBONE:
    NAME: "build_resnet_fpn_backbone"
  RESNETS:
    OUT_FEATURES: ["res2", "res3", "res4", "res5"]
  FPN:
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
  ANCHOR_GENERATOR:
    SIZES: [[32], [64], [128], [256], [512]]  # One size for each in feature map
    ASPECT_RATIOS: [[0.5, 1.0, 2.0]]  # Three aspect ratios (same for all in feature maps)
  RPN:
    IN_FEATURES: ["p2", "p3", "p4", "p5", "p6"]
    PRE_NMS_TOPK_TRAIN: 2000  # Per FPN level
    PRE_NMS_TOPK_TEST: 1000  # Per FPN level
    POST_NMS_TOPK_TRAIN: 1000
    POST_NMS_TOPK_TEST: 1000
  ROI_HEADS:
    NAME: "StandardROIHeads"
    IN_FEATURES: ["p2", "p3", "p4", "p5"]
  ROI_BOX_HEAD:
    NAME: "FastRCNNConvFCHead"
    NUM_FC: 2
    POOLER_RESOLUTION: 7
  ROI_MASK_HEAD:
    NAME: "MaskRCNNConvUpsampleHead"
    NUM_CONV: 4
    POOLER_RESOLUTION: 14
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
  MASK_ON: True
  RESNETS:
    DEPTH: 50
DATASETS:
  TRAIN: ("nuimages_train",)
  TEST: ("nuimages_test",)
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.02
  STEPS: (210000, 250000)
  MAX_ITER: 270000
INPUT:
  MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)
  MASK_FORMAT: "bitmask"
VERSION: 2
```

**5.Use nuImgaes and CRUW to make inference**

The inference part of this project can be found in [nuimages_inference.py](https://github.com/TedSongjh/CSE599-fianl-project/blob/main/nuimages_inference.py)
I use my final model to make inference to both dataset, and visualize the result. The transform_instance_to_dict function read the instance in each sample, because in the first version, I didn't modify the nuImages, so this function will filter out all the instance to be ignored and reture the dictionary and new instance. In the meanwhile, the write_det_txt function will write the result to a single annotation file for each sample sequence.

## Qualitative Results
I firstly use nuImages-mini dataset to evaluate the model. And then run full inference on CRUW dataset. Here I can show both success case and failure case in complex senerio for both dataset. The results are as follow:

**nuImage Inference Result**
For full nuImages inference result in txt dict and images, please check [here](https://github.com/TedSongjh/CSE599-fianl-project/tree/main/nuimage_mini_result).
* Successfull Result
![nuimage_mini_1](/nuimage_mini_result/vis/CAM_FRONT/nuimage_coco/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915374762465.jpg)In this result, the trucks and trailer with low contrast and low saturation are successfully seprated from the building with similar geo features.![nuimage_mini_2](/nuimage_mini_result/vis/CAM_FRONT/nuimage_coco/n009-2018-09-12-09-59-51-0400__CAM_FRONT__1536761961012656.jpg)In this result, all the cars and pedestrian is identified in this complex city scenes.

* Partially Failure Result
![nuimage_mini_3](/nuimage_mini_result/vis/CAM_FRONT_LEFT/nuimage_coco/n013-2018-09-04-13-30-50+0800__CAM_FRONT_LEFT__1536039168104825.jpg)All the motorcycle are failed to detection, also the pedestrian segmantation is not accuracy.![nuimage_mini_4](/nuimage_mini_result/vis/CAM_BACK/nuimage_coco/n003-2018-01-08-11-30-34+0800__CAM_BACK__1515382745757583.jpg)Although all the cars and truck are identified, the pink bounding box and segmantation mask is overlapped for the ISUZU truck, can the score of vehicle.car is even higher. Also, the pedestrian in the right is detected as 2 instances for barly the same bounding box and segmantation. I will implement a function to merge instances result when IoU arrive at certain thread hold.

**CRUW Inference Result**
Because the CRUW dataset has not been published yet. I can only show some representative result.
For more result please check [here](https://github.com/TedSongjh/CSE599-fianl-project/tree/main/pix)
* Successfull Result
![CRUW_1](/pix/0000000004.jpg)In this complex trafic scence, all the instances are identified, even part of the pedestrain is block by the car in front, the bounding box is still accuracy.![CRUW_2](/pix/0000000063.jpg)This night scence result can show that this model successfully detect objects in low light and high image noise samples. ![CRUW_3](/pix/0000001264.jpg)
This is a result from highway scence, which will be disscuss in the compare with the following failure result.

* Partially Failure Result
![CRUW_3](/pix/0000001048.jpg)
For the truck on the left, this model fail to identify object class, bounding box and segmantation. But after few frames, the picture above show the correct result for the truck. 


## Quantitative Results
Because nuScense dataset only have a evaluator for nuScense, I didn't find a proper evaluator for nuImages, so I use built in COCO evaluator in Dectron2 and register the nuImages dataset as COCO format. The bounding box evaluation result can reach a average precision of 50.304%

Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 50.304 | 68.203 | 58.817 | 51.841 | 58.214 | 44.379 |

Evaluation results for segm: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 25.658 | 52.114 | 21.579 | 14.535 | 40.472 | 34.619 |


Per-category bbox AP: 
| category                           | AP     | category                        | AP     | category                             | AP     |
|:-----------------------------------|:-------|:--------------------------------|:-------|:-------------------------------------|:-------|
| human.pedestrian.adult             | 70.000 | human.pedestrian.child          | nan    | human.pedestrian.stroller            | 11.777 |
| human.pedestrian.personal_mobility | 61.262 | human.pedestrian.police_officer | 34.234 | human.pedestrian.construction_worker | 86.386 |
| vehicle.car                        | 64.267 | vehicle.bus.bendy               | 69.901 | vehicle.bus.rigid                    | 60.198 |
| vehicle.truck                      | 12.354 | vehicle.trailer                 | 32.661 |                                      |        |


Per-category segm AP: 
| category                           | AP     | category                        | AP     | category                             | AP     |
|:-----------------------------------|:-------|:--------------------------------|:-------|:-------------------------------------|:-------|
| human.pedestrian.adult             | 40.000 | human.pedestrian.child          | nan    | human.pedestrian.stroller            | 7.303  |
| human.pedestrian.personal_mobility | 23.305 | human.pedestrian.police_officer | 18.604 | human.pedestrian.construction_worker | 44.653 |
| vehicle.car                        | 43.410 | vehicle.bus.bendy               | 21.460 | vehicle.bus.rigid                    | 35.347 |
| vehicle.truck                      | 8.803  | vehicle.trailer                 | 13.693 |                                      |        |




Intersection over Union(IoU) Evaluation AP and AR:
|Evaluation|  IoU   | Range  | MaxDets|Result|
|:--------:|:------:|:------:|:------:|:------:|
|Average Precision  (AP)| @IoU=0.50:0.95 | area=   all | maxDets=100 |0.257|
|Average Precision  (AP) |0.50      | area=   all | maxDets=100 |0.521|
|Average Precision  (AP) |0.75      | area=   all | maxDets=100 |0.216|
|Average Precision  (AP) |0.50:0.95 | area= small | maxDets=100 |0.145|
|Average Precision  (AP) |IoU=0.50:0.95 | area=medium | maxDets=100 |0.405|
|Average Precision  (AP) |IoU=0.50:0.95 | area= large | maxDets=100 |0.346|
|Average Recall     (AR) |IoU=0.50:0.95 | area=   all | maxDets=  1 |0.262|
|Average Recall     (AR) |IoU=0.50:0.95 | area=   all | maxDets= 10 |0.423|
|Average Recall     (AR) |IoU=0.50:0.95 | area=   all | maxDets=100 |0.437|
|Average Recall     (AR) |IoU=0.50:0.95 | area= small | maxDets=100 |0.295|
|Average Recall     (AR) |IoU=0.50:0.95 | area=medium | maxDets=100 |0.615|
|Average Recall     (AR) |IoU=0.50:0.95 | area= large | maxDets=100 |0.661


## Discussion
Since this is a long term project for both my lab and myself, I will continue to fix the problem above to make our dataset perfect. Here are my idea to fix current defects:
* For the identify of bicycle and motorcycle, the cyclist on the cycle affect the feature extracting part a lot, which makes the result of cycle not accuracy. Also this is one of the problem that there weren't a lot cycle and cyclist input in both datasets. For this problem, I have two feasible solutions. Fist I will seprately train the cycle with cyclist and the ones without, and concatinate the cycle and cyclist instances by the overlap area and location between two bounding box and segmantation. The other solution is to  use the same idea to concatinate the cycle and cyclist before training, and then train the empty cycle and cycle with cyclist sepratlly.
* For some failure case when the truck is identify for both car and truck, I will use the bounding box and segmantationv to identify the overlap and use overlap area as weight times the score to give a single result from multiple redundent instances result.

In the end, many thanks to Joseph and all the TAs for all the help this quater, I will keep leaning to make a better progress.
