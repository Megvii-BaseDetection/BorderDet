# borderdet.res50.fpn.coco.800size.1x

## Evaluation results for bbox:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.414
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.594
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.445
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.236
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.546
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.592
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.634
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.765
```
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 41.363 | 59.434 | 44.513 | 23.609 | 45.107 | 54.633 |

### Per-category bbox AP:

| category      | AP     | category     | AP     | category       | AP     |
|:--------------|:-------|:-------------|:-------|:---------------|:-------|
| person        | 55.215 | bicycle      | 30.361 | car            | 43.764 |
| motorcycle    | 42.495 | airplane     | 68.315 | bus            | 66.038 |
| train         | 64.576 | truck        | 37.915 | boat           | 25.074 |
| traffic light | 27.415 | fire hydrant | 65.933 | stop sign      | 64.989 |
| parking meter | 45.616 | bench        | 22.465 | bird           | 35.470 |
| cat           | 70.755 | dog          | 65.977 | horse          | 57.478 |
| sheep         | 52.916 | cow          | 57.816 | elephant       | 65.628 |
| bear          | 67.673 | zebra        | 69.101 | giraffe        | 67.730 |
| backpack      | 14.406 | umbrella     | 40.337 | handbag        | 14.295 |
| tie           | 31.805 | suitcase     | 37.788 | frisbee        | 67.562 |
| skis          | 21.066 | snowboard    | 30.056 | sports ball    | 45.495 |
| kite          | 42.875 | baseball bat | 27.789 | baseball glove | 37.333 |
| skateboard    | 53.046 | surfboard    | 33.703 | tennis racket  | 47.475 |
| bottle        | 37.817 | wine glass   | 37.240 | cup            | 42.693 |
| fork          | 31.639 | knife        | 15.265 | spoon          | 13.950 |
| bowl          | 41.319 | banana       | 25.722 | apple          | 20.206 |
| sandwich      | 35.238 | orange       | 32.397 | broccoli       | 24.431 |
| carrot        | 21.509 | hot dog      | 32.201 | pizza          | 50.616 |
| donut         | 47.213 | cake         | 37.497 | chair          | 27.931 |
| couch         | 44.569 | potted plant | 27.574 | bed            | 42.770 |
| dining table  | 27.596 | toilet       | 60.420 | tv             | 54.948 |
| laptop        | 59.768 | mouse        | 62.668 | remote         | 28.422 |
| keyboard      | 49.233 | cell phone   | 36.512 | microwave      | 60.062 |
| oven          | 33.500 | toaster      | 35.623 | sink           | 35.969 |
| refrigerator  | 54.427 | book         | 13.803 | clock          | 50.479 |
| vase          | 36.694 | scissors     | 29.152 | teddy bear     | 48.276 |
| hair drier    | 6.760  | toothbrush   | 19.140 |                |        |
