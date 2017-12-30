"""
.. module:: trainer
    :synopsis: trainer module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 23, 2017
"""
import pandas as pd

from eliana.imports import *


def train_emotion(emotion, append, trainer_, mode):

    dir_images = os.path.join(
        trainer_['test_images'],
        emotion
    )

    build_training_data(
        dir_images=dir_images,
        dataset=trainer_['dataset'],
        tag=emotion,
        append=append,
        columns=trainer_['columns'],
        mode=mode
    )
    if mode == 'overall':
        inputs = trainer_['columns'][1:6]
    elif mode == 'oia':
        inputs = trainer_['columns'][1:7]

    train(
        model=trainer_['model'],
        dataset=trainer_['dataset'],
        inputs=inputs
    )

    mlp = MLP()
    mlp.load_model(path=trainer_['model'])

    df = pd.read_pickle(trainer_['dataset'])
    # print('Dataset:\n', df)
    df = df[inputs].as_matrix()
    for data in df:
        print(data)
        print('Run:', mlp.run(input_=data))


train_emotion(
    'happiness',
    append=False,
    trainer_=trainer_w_oia,
    mode='oia'
)
train_emotion(
    'sadness',
    append=True,
    trainer_=trainer_w_oia,
    mode='oia'
)
train_emotion(
    'fear',
    append=True,
    trainer_=trainer_w_oia,
    mode='oia'
)

# mlp = MLP()
# mlp.load_model(path=trainer['model'])

# input_ = [0.001, 0.009]
# output = [[5]]
# print('\n------\nScore:', mlp.model.score([input_], output))
# print('Run:', mlp.run(input_=input_))
