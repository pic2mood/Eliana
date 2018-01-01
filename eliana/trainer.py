"""
.. module:: trainer
    :synopsis: trainer module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 23, 2017
"""
from eliana import config
from eliana.utils import *


def train_emotion(trainer_, combinations=None):

    if combinations is None:
        build_training_data(
            trainer=trainer_,
        )
    else:
        build_training_data(
            trainer=trainer_,
            emotion_combinations=combinations
        )

    inputs = trainer_['columns'][1:-2]
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


if __name__ == '__main__':

    import sys

    if len(sys.argv) > 1:

        if sys.argv[1] == 'woia':
            trainer = config.trainer_w_oia

        elif sys.argv[1] == 'noia':
            trainer = config.trainer_no_oia

        else:
            raise ValueError('Invalid argument {0}'.format(sys.argv[1]))

    train_emotion(
        trainer_=trainer
    )
