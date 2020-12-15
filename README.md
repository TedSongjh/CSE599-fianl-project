# CSE599-fianl-project

## Introduction
This project is aiming for provide ground truth for [CRUW dataset](https://www.cruwdataset.org/introduction) built by Information Processing Lab @UWECE. The CRUW dataset is a autonomous driving senerio dataset with dual-camera and dual radar sensor setup. Which aming for sensor fusion reseach and object detection by radar. And also this dataset will be publish soon and a challenge for radar-camera sensor fusion will be held. My works is to preprocess both the radar and image and then provide image base object detection groud truth result as benchmark for radar process algorism. The object detection base on Detectron is the main part for this final project. I use a new dataset called [nuImages](https://www.nuscenes.org/nuimages) to pretrain my benchmark model instead of KITTI because the seniero and quaility are similar to the CRUW we collected. And also, the metadata and annotation is extract from [nuScenes](https://www.nuscenes.org/nuscenes) dataset, which have more information in the relational database.



## Related Work
[Detectron2](https://github.com/facebookresearch/detectron2) 
Detectron 2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms. It is a ground-up rewrite of the previous version, Detectron, and it originates from maskrcnn-benchmark. This object detection algorithms is powered by the PyTorch deep learning framework,and also includes more features such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend, DeepLab, etc. We use mask R-CNN part of this project, and modify the source code to custom dataset.
To install Dectron2:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

[nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) 
This devkit is used in this project to load nuImage dataset. And also, when we desgined our dataset format, we took some schema built idea from this dataset.
To use nuScenes devkit:
```
pip install nuscenes-devkit
```
[CRUW dataset](https://www.cruwdataset.org/introduction)
CRUW is a public camera-radar dataset for autonomous vehicle applications. It is a good resource for researchers to study FMCW radar data, that has high potential in the future autonomous driving. We will publish this dataset soon.


Other people are out there doing things. What did they do? Was it good? Was it bad? Talk about it here.

## Approach
**CRUW dataset annotation format**
My first attribute to this project is to develope a annotation tool kit for our own dataset annotation format. The annotation writer and loader parts of project is done in this summer 2020. The development kit is in this repositories [CRUW devkit](https://github.com/yizhou-wang/cruw-devkit). Also, I made a annotation visualization tool for the inference
* To use cruw devkit:
** 1.Create a new conda environment.
```
conda create -n cruw-devkit python=3.6
```
** 2.Run setup tool for this devkit.
```
conda activate cruw-devkit
pip install -e .
```
* The annotations provided by CRUW dataset include the following:
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
**Convert nuImages dataset to CRUW dataset format**
The nuImages dataset have more attributes than our dataset, and also the nuImages's categories is in detail, which is meaning less for our Camera-Radar Fusion (CRF) annotation(The information extracted from radar can only provide accuracy location and velocity information, the feature of objects are compressed).
The categories mapping from nuImages to CRUW is:
nuImages Category | CRUW Category
------------ | -------------
animal	|	
human.pedestrian.adult	|	human.pedestrian
human.pedestrian.child	|	human.pedestrian
human.pedestrian.construction_worker	|	human.pedestrian
human.pedestrian.personal_mobility	|	human.pedestrian
human.pedestrian.police_officer	|	human.pedestrian
human.pedestrian.stroller	|	human.pedestrian
human.pedestrian.wheelchair	|	human.pedestrian
movable_object.barrier	|	
movable_object.debris	|	
movable_object.pushable_pullable	|	
movable_object.trafficcone	|	
static_object.bicycle_rack	|	
vehicle.bicycle	|	vehicle.cycle
vehicle.bus.bendy	|	vehicle.bus
vehicle.bus.rigid	|	vehicle.bus
vehicle.car	|	vehicle.car
vehicle.construction	|	vehicle.car
vehicle.emergency.ambulance	|	vehicle.car
vehicle.emergency.police	|	vehicle.car
vehicle.motorcycle	|	vehicle.cycle
vehicle.trailer	|	vehicle.truck
vehicle.truck	|	vehicle.truck
flat.drivable_surface	|	
flat.ego	|	

**Use Custom datasets on Detectron2**



How did you decide to solve the problem? What network architecture did you use? What data? Lots of details here about all the things you did. This section describes almost your whole project.

Figures are good here. Maybe you present your network architecture or show some example data points?


```
./train_net.py   --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml   --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```
## Results

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

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.257
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.521
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.262
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.295
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.661




How did you evaluate your approach? How well did you do? What are you comparing to? Maybe you want ablation studies or comparisons of different methods.

You may want some qualitative results and quantitative results. Example images/text/whatever are good. Charts are also good. Maybe loss curves or AUC charts. Whatever makes sense for your evaluation.

## Discussion
