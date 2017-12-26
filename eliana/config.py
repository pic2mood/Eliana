
import os

annotator_params = {
    'model': 'ssd_mobilenet_v1_coco_11_06_2017',
    'ckpt': os.path.join(
        os.getcwd(),
        'training',
        'models',
        'ssd_mobilenet_v1_coco_11_06_2017',
        'frozen_inference_graph.pb'
    ),
    'labels': os.path.join(
        os.getcwd(),
        'training',
        'data',
        'mscoco_label_map.pbtxt'
    ),
    'classes': 90
}
