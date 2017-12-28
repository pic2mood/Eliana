
import os
import logging
import traceback

logger_ = logging.getLogger(__name__)
logger_.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '\n[%(asctime)s] %(name)s: %(levelname)s: %(message)s'
)
handler.setFormatter(formatter)
logger_.addHandler(handler)


def log(message, verbose=True):
    def decorator(func):

        def wrapped(*args, **kwargs):
            logger_.info(message)
            try:
                func(*args, **kwargs)
            except Exception:
                if verbose:
                    logger_.error(traceback.format_exc())
            finally:
                logger_.info('DONE.')

        return wrapped
    return decorator


emotions_map = {
    'happiness': 1,
    'anger': 2,
    'surprise': 3,
    'disgust': 4,
    'sadness': 5,
    'fear': 6
}

emotions_list = [
    '',
    'happiness',
    'anger',
    'surprise',
    'disgust',
    'sadness',
    'fear'
]

verbose = True

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
        'eliana_ann_overall_3_class_3_feature_dataset.pkl'
    ),
    'test_images': os.path.join(
        os.getcwd(),
        'training',
        'data'
    ),
    'model': os.path.join(
        os.getcwd(),
        'training',
        'models',
        'eliana_ann_overall',
        'eliana_ann_overall_3_class_3_feature.pkl'
    )
}
