# CSE599-fianl-project

## Introduction
This project is aiming for provide ground truth for futher CRUW dataset made by Information Processing Lab @UWECE. The CRUW dataset is a autonomous driving senerio dataset with dual-camera and dual radar sensor setup. Which aming for sensor fusion reseach and object detection by radar. And also this dataset will be publish soon and a challenge for radar-camera sensor fusion will be held. My works is to preprocess both the radar and image and then provide image base object detection groud truth result as benchmark for radar process algorism. The object detection base on Detectron is the main part for this final project. I use a new dataset called nuImages instead of KITTI

## Related Work

Other people are out there doing things. What did they do? Was it good? Was it bad? Talk about it here.

## Approach

How did you decide to solve the problem? What network architecture did you use? What data? Lots of details here about all the things you did. This section describes almost your whole project.

Figures are good here. Maybe you present your network architecture or show some example data points?


./train_net.py   --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml   --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
## Results

Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 50.304 | 68.203 | 58.817 | 51.841 | 58.214 | 44.379 |
[12/08 08:17:54 d2.evaluation.coco_evaluation]: Per-category bbox AP: 
| category                           | AP     | category                        | AP     | category                             | AP     |
|:-----------------------------------|:-------|:--------------------------------|:-------|:-------------------------------------|:-------|
| human.pedestrian.adult             | 70.000 | human.pedestrian.child          | nan    | human.pedestrian.stroller            | 11.777 |
| human.pedestrian.personal_mobility | 61.262 | human.pedestrian.police_officer | 34.234 | human.pedestrian.construction_worker | 86.386 |
| vehicle.car                        | 64.267 | vehicle.bus.bendy               | 69.901 | vehicle.bus.rigid                    | 60.198 |
| vehicle.truck                      | 12.354 | vehicle.trailer                 | 32.661 |                                      |        |
Loading and preparing results...
DONE (t=1.50s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
COCOeval_opt.evaluate() finished in 2.15 seconds.
Accumulating evaluation results...
COCOeval_opt.accumulate() finished in 0.30 seconds.
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
[12/08 08:18:01 d2.evaluation.coco_evaluation]: Evaluation results for segm: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 25.658 | 52.114 | 21.579 | 14.535 | 40.472 | 34.619 |
[12/08 08:18:01 d2.evaluation.coco_evaluation]: Per-category segm AP: 
| category                           | AP     | category                        | AP     | category                             | AP     |
|:-----------------------------------|:-------|:--------------------------------|:-------|:-------------------------------------|:-------|
| human.pedestrian.adult             | 40.000 | human.pedestrian.child          | nan    | human.pedestrian.stroller            | 7.303  |
| human.pedestrian.personal_mobility | 23.305 | human.pedestrian.police_officer | 18.604 | human.pedestrian.construction_worker | 44.653 |
| vehicle.car                        | 43.410 | vehicle.bus.bendy               | 21.460 | vehicle.bus.rigid                    | 35.347 |
| vehicle.truck                      | 8.803  | vehicle.trailer                 | 13.693 |                                      |        |
[12/08 08:18:02 d2.engine.defaults]: Evaluation results for nuimages_test in csv format:
[12/08 08:18:02 d2.evaluation.testing]: copypaste: Task: bbox
[12/08 08:18:02 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[12/08 08:18:02 d2.evaluation.testing]: copypaste: 50.3040,68.2026,58.8173,51.8409,58.2141,44.3791
[12/08 08:18:02 d2.evaluation.testing]: copypaste: Task: segm
[12/08 08:18:02 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[12/08 08:18:02 d2.evaluation.testing]: copypaste: 25.6578,52.1140,21.5786,14.5351,40.4724,34.6188

How did you evaluate your approach? How well did you do? What are you comparing to? Maybe you want ablation studies or comparisons of different methods.

You may want some qualitative results and quantitative results. Example images/text/whatever are good. Charts are also good. Maybe loss curves or AUC charts. Whatever makes sense for your evaluation.

## Discussion
