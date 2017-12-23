"""
.. module:: trainer
    :synopsis: trainer module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 23, 2017
"""
import os
import pandas as pd
import tensorflow as tf
from eliana.lib.annotator import Annotator
from eliana.lib.color import Color
from eliana.lib.texture import Texture
from eliana.lib.image_batch_loader import ImageBatchLoader
from eliana.lib.mlp import MLP


class Trainer:

    def train(self, model, dataset):

        df = pd.read_pickle(dataset)

        mlp = MLP()
        mlp.load_model(path=None)

        inputs = df[['Color', 'Texture']]
        outputs = df[['Emotion Value']].as_matrix().ravel()

        print('Trainer:', mlp.train(inputs, outputs))

        mlp.save_model(path=model)


class TrainingData:

    def interpolate(self, value, place=0.01):
        return float(format(value * place, '.3f'))

    def build(self, dir_images, dataset):

        # manual emotion tags
        emotions = []

        dir_et = os.path.join(dir_images, 'manual_annotator.txt')

        with open(dir_et, 'r') as et:
            for entry in et:

                entry = entry.strip()

                if entry != '':
                    name, emotion = tuple(entry.split('|'))
                    name = name.split('/')[-1]

                    emotions.append(emotion)

        emotions_map = {
            'happiness': 1,
            'anger': 2,
            'surprise': 3,
            'disgust': 4,
            'sadness': 5,
            'fear': 6
        }

        # prepare object annotator
        # model = 'ssd_mobilenet_v1_coco_11_06_2017'
        # graph = 'frozen_inference_graph.pb'
        # file_ckpt = os.path.join(
        #     os.getcwd(),
        #     'training',
        #     'models',
        #     model,
        #     graph
        # )
        # label = 'mscoco_label_map.pbtxt'
        # file_label = os.path.join(
        #     os.getcwd(),
        #     'training',
        #     'data',
        #     label
        # )
        # num_classes = 90
        # detection_graph = tf.Graph()

        # with detection_graph.as_default():

        #     graph_def = tf.GraphDef()
        #     fid = tf.gfile.GFile(file_ckpt, 'rb')
        #     graph_serialized = fid.read()
        #     graph_def.ParseFromString(graph_serialized)
        #     tf.import_graph_def(graph_def, name='')

        #     sess = tf.Session(graph=detection_graph)

        # raw images
        # raw_images = ImageBatchLoader(dir_images, limit=None).images

        # data building
        data = []
        for i, (img, img_path) in enumerate(
            ImageBatchLoader.images(dir_images, limit=None)
        ):

            # objects = Annotator(
            #     img,
            #     model,
            #     file_ckpt,
            #     file_label,
            #     num_classes,
            #     sess,
            #     detection_graph
            # ).annotate()

            color = Color.colorfulness(img)
            color = Color.scaled_colorfulness(color)
            color = self.interpolate(color)

            texture = Texture.texture(img)
            texture = self.interpolate(texture, place=0.1)

            data.append(
                [
                    img_path.split('/')[-1],
                    color,
                    texture,
                    emotions[i],
                    emotions_map[emotions[i]]
                ]
            )

        df = pd.DataFrame(
            data,
            columns=[
                'Image',
                'Color',
                'Texture',
                'Emotion',
                'Emotion Value'
            ]
        )
        print('Dataset:\n', df)
        df.to_pickle(dataset)

        # df = pd.read_pickle(dataset)
        # print(df)


dataset = os.path.join(
    os.getcwd(),
    'training',
    'data',
    'eliana_ann_overall_dataset.pkl'
)
dir_images = os.path.join(
    os.getcwd(),
    'training',
    'data',
    'test_images'
)
TrainingData().build(
    dir_images=dir_images,
    dataset=dataset
)

model_path = os.path.join(
    os.getcwd(),
    'training',
    'models',
    'eliana_ann_overall',
    'eliana_ann_overall.pkl'
)
Trainer().train(
    model=model_path,
    dataset=dataset
)

mlp = MLP()
mlp.load_model(path=model_path)

df = pd.read_pickle(dataset)
print('Dataset:\n', df)
df = df[['Color', 'Texture']].as_matrix()
for data in df:
    print('Run:', mlp.run(input_=data))
