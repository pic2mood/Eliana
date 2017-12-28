"""
.. module:: trainer
    :synopsis: trainer module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 23, 2017
"""
import pandas as pd

from eliana.imports import *


def train_emotion(emotion, append):

    dir_images = os.path.join(
        trainer['test_images'],
        emotion
    )

    build_training_data(
        dir_images=dir_images,
        dataset=trainer['dataset'],
        tag=emotion,
        append=append
    )
    train(
        model=trainer['model'],
        dataset=trainer['dataset']
    )

    mlp = MLP()
    mlp.load_model(path=trainer['model'])

    df = pd.read_pickle(trainer['dataset'])
    # print('Dataset:\n', df)
    df = df[['Color', 'Texture']].as_matrix()
    for data in df:
        print('Run:', mlp.run(input_=data))


train_emotion('happiness', False)
train_emotion('sadness', True)
train_emotion('fear', True)

# mlp = MLP()
# mlp.load_model(path=trainer['model'])

# input_ = [0.001, 0.009]
# output = [[5]]
# print('\n------\nScore:', mlp.model.score([input_], output))
# print('Run:', mlp.run(input_=input_))
