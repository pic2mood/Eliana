
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '\n[%(asctime)s] %(name)s: %(levelname)s: %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


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


trainer = {
    'dataset': os.path.join(
        os.getcwd(),
        'training',
        'data',
        'eliana_ann_overall_dataset.pkl'
    ),
    'test_images': os.path.join(
        os.getcwd(),
        'training',
        'data',
        'test_images'
    ),
    'model': os.path.join(
        os.getcwd(),
        'training',
        'models',
        'eliana_ann_overall',
        'eliana_ann_overall.pkl'
    )
}
