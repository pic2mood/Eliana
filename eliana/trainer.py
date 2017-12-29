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
        trainer_overall['test_images'],
        emotion
    )

    build_training_data(
        dir_images=dir_images,
        dataset=trainer_overall['dataset'],
        tag=emotion,
        append=append,
        columns=trainer_overall['columns']
    )
    train(
        model=trainer_overall['model'],
        dataset=trainer_overall['dataset'],
        inputs=trainer_overall['columns'][1:6]
    )

    mlp = MLP()
    mlp.load_model(path=trainer_overall['model'])

    df = pd.read_pickle(trainer_overall['dataset'])
    # print('Dataset:\n', df)
    df = df[trainer_overall['columns'][1:6]].as_matrix()
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
