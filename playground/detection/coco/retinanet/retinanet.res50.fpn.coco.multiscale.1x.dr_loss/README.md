# retinanet.res50.fpn.coco.800size.1x.dr_loss  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.560
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.585
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.708
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 37.353 | 56.008 | 40.058 | 21.421 | 41.257 | 50.294 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 51.226 | bicycle      | 28.259 | car            | 41.051 |  
| motorcycle    | 40.522 | airplane     | 61.670 | bus            | 63.804 |  
| train         | 59.230 | truck        | 33.637 | boat           | 22.437 |  
| traffic light | 23.812 | fire hydrant | 63.307 | stop sign      | 63.117 |  
| parking meter | 43.824 | bench        | 21.553 | bird           | 33.718 |  
| cat           | 65.062 | dog          | 59.897 | horse          | 50.959 |  
| sheep         | 46.765 | cow          | 49.910 | elephant       | 57.967 |  
| bear          | 66.841 | zebra        | 65.187 | giraffe        | 64.257 |  
| backpack      | 12.948 | umbrella     | 33.987 | handbag        | 12.395 |  
| tie           | 27.163 | suitcase     | 32.594 | frisbee        | 61.789 |  
| skis          | 18.194 | snowboard    | 19.006 | sports ball    | 43.529 |  
| kite          | 37.304 | baseball bat | 21.454 | baseball glove | 31.040 |  
| skateboard    | 48.668 | surfboard    | 29.566 | tennis racket  | 44.222 |  
| bottle        | 32.986 | wine glass   | 33.557 | cup            | 38.439 |  
| fork          | 23.384 | knife        | 10.605 | spoon          | 11.826 |  
| bowl          | 38.466 | banana       | 21.741 | apple          | 17.490 |  
| sandwich      | 32.029 | orange       | 28.217 | broccoli       | 20.240 |  
| carrot        | 18.842 | hot dog      | 28.313 | pizza          | 47.620 |  
| donut         | 40.253 | cake         | 29.825 | chair          | 23.526 |  
| couch         | 39.873 | potted plant | 23.474 | bed            | 40.251 |  
| dining table  | 24.787 | toilet       | 57.574 | tv             | 52.879 |  
| laptop        | 55.358 | mouse        | 58.304 | remote         | 24.762 |  
| keyboard      | 43.610 | cell phone   | 32.612 | microwave      | 57.191 |  
| oven          | 31.706 | toaster      | 33.621 | sink           | 31.389 |  
| refrigerator  | 49.167 | book         | 11.491 | clock          | 48.782 |  
| vase          | 34.223 | scissors     | 22.478 | teddy bear     | 43.896 |  
| hair drier    | 5.499  | toothbrush   | 16.101 |                |        |
