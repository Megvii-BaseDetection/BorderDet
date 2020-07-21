# fcos.res50.fpn.coco.800size.1x.fix_bug  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.576
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.534
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.616
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.735
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 38.805 | 57.601 | 42.041 | 22.400 | 42.578 | 50.262 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 53.260 | bicycle      | 29.385 | car            | 41.813 |  
| motorcycle    | 39.798 | airplane     | 64.624 | bus            | 62.616 |  
| train         | 59.045 | truck        | 32.574 | boat           | 23.324 |  
| traffic light | 25.466 | fire hydrant | 64.778 | stop sign      | 62.586 |  
| parking meter | 42.110 | bench        | 20.482 | bird           | 33.208 |  
| cat           | 64.192 | dog          | 60.999 | horse          | 51.798 |  
| sheep         | 50.941 | cow          | 55.895 | elephant       | 62.747 |  
| bear          | 70.969 | zebra        | 67.077 | giraffe        | 64.299 |  
| backpack      | 13.320 | umbrella     | 37.708 | handbag        | 13.033 |  
| tie           | 28.672 | suitcase     | 34.798 | frisbee        | 65.062 |  
| skis          | 17.534 | snowboard    | 27.383 | sports ball    | 45.471 |  
| kite          | 41.312 | baseball bat | 23.984 | baseball glove | 33.830 |  
| skateboard    | 50.089 | surfboard    | 32.528 | tennis racket  | 44.606 |  
| bottle        | 35.022 | wine glass   | 34.308 | cup            | 41.791 |  
| fork          | 26.318 | knife        | 14.142 | spoon          | 12.493 |  
| bowl          | 38.979 | banana       | 22.695 | apple          | 19.476 |  
| sandwich      | 33.567 | orange       | 33.654 | broccoli       | 22.437 |  
| carrot        | 19.591 | hot dog      | 29.484 | pizza          | 49.417 |  
| donut         | 45.853 | cake         | 34.259 | chair          | 25.902 |  
| couch         | 39.947 | potted plant | 26.663 | bed            | 40.067 |  
| dining table  | 24.568 | toilet       | 57.637 | tv             | 51.173 |  
| laptop        | 55.931 | mouse        | 58.387 | remote         | 27.944 |  
| keyboard      | 46.272 | cell phone   | 34.371 | microwave      | 56.653 |  
| oven          | 29.280 | toaster      | 29.388 | sink           | 33.247 |  
| refrigerator  | 51.578 | book         | 12.728 | clock          | 49.556 |  
| vase          | 36.491 | scissors     | 22.074 | teddy bear     | 44.277 |  
| hair drier    | 9.562  | toothbrush   | 13.890 |                |        |
