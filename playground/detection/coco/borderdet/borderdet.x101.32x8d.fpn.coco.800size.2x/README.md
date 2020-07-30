# borderdet.x101.32x8d.fpn.coco.800size.2x  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.456
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.646
 Average Precision  (AP) @[ IoU=0.60      | area=   all | maxDets=100 ] = 0.607
 Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=100 ] = 0.543
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.494
 Average Precision  (AP) @[ IoU=0.80      | area=   all | maxDets=100 ] = 0.429
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.280
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.492
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.619
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.661
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.774
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 45.641 | 64.558 | 60.676 | 54.325 | 49.416 | 42.896 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 58.956 | bicycle      | 37.279 | car            | 48.317 |  
| motorcycle    | 48.111 | airplane     | 71.809 | bus            | 70.894 |  
| train         | 67.042 | truck        | 40.555 | boat           | 30.627 |  
| traffic light | 30.664 | fire hydrant | 68.832 | stop sign      | 69.594 |  
| parking meter | 48.734 | bench        | 26.863 | bird           | 40.347 |  
| cat           | 74.244 | dog          | 68.222 | horse          | 64.364 |  
| sheep         | 59.283 | cow          | 62.006 | elephant       | 68.723 |  
| bear          | 73.879 | zebra        | 69.744 | giraffe        | 72.562 |  
| backpack      | 19.847 | umbrella     | 44.134 | handbag        | 19.156 |  
| tie           | 38.491 | suitcase     | 44.565 | frisbee        | 69.201 |  
| skis          | 27.633 | snowboard    | 41.766 | sports ball    | 49.451 |  
| kite          | 44.985 | baseball bat | 33.652 | baseball glove | 40.148 |  
| skateboard    | 59.199 | surfboard    | 40.173 | tennis racket  | 55.932 |  
| bottle        | 41.381 | wine glass   | 40.691 | cup            | 47.185 |  
| fork          | 41.139 | knife        | 22.577 | spoon          | 24.743 |  
| bowl          | 42.592 | banana       | 25.143 | apple          | 22.151 |  
| sandwich      | 37.810 | orange       | 32.263 | broccoli       | 23.524 |  
| carrot        | 24.997 | hot dog      | 39.631 | pizza          | 54.956 |  
| donut         | 52.294 | cake         | 38.828 | chair          | 32.478 |  
| couch         | 48.102 | potted plant | 29.524 | bed            | 45.428 |  
| dining table  | 29.528 | toilet       | 63.965 | tv             | 58.264 |  
| laptop        | 63.910 | mouse        | 64.017 | remote         | 37.700 |  
| keyboard      | 53.124 | cell phone   | 40.218 | microwave      | 63.154 |  
| oven          | 35.101 | toaster      | 38.519 | sink           | 39.538 |  
| refrigerator  | 59.270 | book         | 15.531 | clock          | 51.200 |  
| vase          | 40.140 | scissors     | 34.375 | teddy bear     | 51.316 |  
| hair drier    | 13.497 | toothbrush   | 31.505 |                |        |
