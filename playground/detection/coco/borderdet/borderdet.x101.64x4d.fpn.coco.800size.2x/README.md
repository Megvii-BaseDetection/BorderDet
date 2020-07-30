# borderdet.x101.64x4d.fpn.coco.800size.2x  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.462
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.653
 Average Precision  (AP) @[ IoU=0.60      | area=   all | maxDets=100 ] = 0.612
 Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=100 ] = 0.547
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.498
 Average Precision  (AP) @[ IoU=0.80      | area=   all | maxDets=100 ] = 0.437
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.288
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.496
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.600
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.362
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.585
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.623
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.659
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.786
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 46.207 | 65.279 | 61.202 | 54.691 | 49.823 | 43.712 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 59.241 | bicycle      | 36.330 | car            | 48.669 |  
| motorcycle    | 48.335 | airplane     | 71.538 | bus            | 69.738 |  
| train         | 68.671 | truck        | 42.970 | boat           | 31.435 |  
| traffic light | 31.347 | fire hydrant | 71.772 | stop sign      | 68.089 |  
| parking meter | 51.214 | bench        | 28.897 | bird           | 40.709 |  
| cat           | 75.600 | dog          | 70.672 | horse          | 64.999 |  
| sheep         | 57.434 | cow          | 63.126 | elephant       | 70.437 |  
| bear          | 75.683 | zebra        | 72.273 | giraffe        | 72.346 |  
| backpack      | 20.303 | umbrella     | 44.795 | handbag        | 20.356 |  
| tie           | 39.245 | suitcase     | 45.557 | frisbee        | 70.649 |  
| skis          | 28.437 | snowboard    | 40.447 | sports ball    | 49.142 |  
| kite          | 46.711 | baseball bat | 34.244 | baseball glove | 41.546 |  
| skateboard    | 57.384 | surfboard    | 39.518 | tennis racket  | 55.668 |  
| bottle        | 42.347 | wine glass   | 40.849 | cup            | 47.464 |  
| fork          | 41.168 | knife        | 24.289 | spoon          | 22.837 |  
| bowl          | 43.935 | banana       | 27.690 | apple          | 23.232 |  
| sandwich      | 41.520 | orange       | 32.638 | broccoli       | 23.099 |  
| carrot        | 24.157 | hot dog      | 38.263 | pizza          | 55.776 |  
| donut         | 51.929 | cake         | 40.061 | chair          | 33.287 |  
| couch         | 48.307 | potted plant | 31.849 | bed            | 47.714 |  
| dining table  | 31.118 | toilet       | 62.766 | tv             | 58.878 |  
| laptop        | 64.823 | mouse        | 63.921 | remote         | 39.885 |  
| keyboard      | 54.652 | cell phone   | 40.964 | microwave      | 62.241 |  
| oven          | 38.151 | toaster      | 42.455 | sink           | 40.054 |  
| refrigerator  | 55.307 | book         | 16.685 | clock          | 53.163 |  
| vase          | 38.922 | scissors     | 36.620 | teddy bear     | 51.836 |  
| hair drier    | 9.814  | toothbrush   | 26.395 |                |        |
