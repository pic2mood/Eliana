"""
.. module:: trainer
    :synopsis: trainer module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 23, 2017
"""
import pandas as pd

from eliana.imports import *

build_training_data(
    dir_images=trainer['test_images'],
    dataset=trainer['dataset']
)
train(
    model=trainer['model'],
    dataset=trainer['dataset']
)

mlp = MLP()
mlp.load_model(path=trainer['model'])

df = pd.read_pickle(trainer['dataset'])
print('Dataset:\n', df)
df = df[['Color', 'Texture']].as_matrix()
for data in df:
    print('Run:', mlp.run(input_=data))
