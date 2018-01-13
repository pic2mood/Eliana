
from eliana.imports import *

import logging
import traceback

from eliana.lib.palette import Palette
from eliana.lib.mlp import MLP
from eliana.lib.annotator import Annotator
from eliana.lib.color import Color
from eliana.lib.texture import Texture
# from eliana.utils import *


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

annotator = Annotator(
    model=annotator_params['model'],
    ckpt=annotator_params['ckpt'],
    labels=annotator_params['labels'],
    classes=annotator_params['classes']
)


trainer_oea_less = {
    'dataset': os.path.join(
        os.getcwd(),
        'training',
        'data',
        'oea_less_dataset.pkl'
    ),
    'model': os.path.join(
        os.getcwd(),
        'training',
        'models',
        'oea_less_model.pkl'
    ),
    'raw_images_root': os.path.join(
        os.getcwd(),
        'training',
        'data'
    ),
    'features': {
        'top_colors': Palette.dominant_colors,
        'colorfulness': Color.scaled_colorfulness,
        'texture': Texture.texture
    },
    'columns': [
        'Image Path',
        'Top Color 1st',
        'Top Color 2nd',
        'Top Color 3rd',
        'Colorfulness',
        'Texture',
        'Emotion Tag',
        'Emotion Value'
    ]
}

trainer_oea = {
    'dataset': os.path.join(
        os.getcwd(),
        'training',
        'data',
        'oea_dataset.pkl'
    ),
    'model': os.path.join(
        os.getcwd(),
        'training',
        'models',
        'oea_model.pkl'
    ),
    'raw_images_root': os.path.join(
        os.getcwd(),
        'training',
        'data'
    ),
    'features': {
        'top_colors': Palette.dominant_colors,
        'colorfulness': Color.scaled_colorfulness,
        'texture': Texture.texture,
        'top_object': annotator.annotate
    },
    'columns': [
        'Image Path',
        'Top Color 1st',
        'Top Color 2nd',
        'Top Color 3rd',
        'Colorfulness',
        'Texture',
        'Top Object',
        'Emotion Tag',
        'Emotion Value'
    ]
}
