#!/usr/bin/python

"""
.. module:: trainer
    :synopsis: trainer module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 23, 2017
"""
import os
import pandas as pd

from eliana.lib.mlp import MLP

from eliana.utils import (
    train, build_training_data
)


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
build_training_data(
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
train(
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
