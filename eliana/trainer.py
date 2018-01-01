"""
.. module:: trainer
    :synopsis: trainer module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 23, 2017
"""
import pandas as pd

from eliana.config import *
from eliana.utils import *


def train_emotion(trainer_, combinations=None):

    if combinations is None:
        build_training_data(
            trainer=trainer_w_oia,
        )
    else:
        build_training_data(
            trainer=trainer_w_oia,
            emotion_combinations=combinations
        )

    inputs = trainer_['columns'][1:7]
    train(
        trainer=trainer_,
        inputs=inputs
    )

    mlp = MLP()
    mlp.load_model(path=trainer_['model'])

    df = pd.read_pickle(trainer_['dataset'])
    # print('Dataset:\n', df)
    df = df[inputs].as_matrix()
    for data in df:
        print('Input:', data)
        print('Run:', mlp.run(input_=data))


train_emotion(
    trainer_=trainer_w_oia
)
