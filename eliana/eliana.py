"""
.. module:: eliana
    :synopsis: main module for eliana package

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 24, 2017
"""
from eliana.imports import *
from eliana import config
from eliana.lib.mlp import MLP
from eliana.utils import *

import multiprocessing as mp


class Eliana:

    def __init__(
            self,
            trainer,
            parallel=False,
            batch=False,
            montage=False,
            single_path='',
    ):
        self.mlp = MLP()
        self.mlp.load_model(path=trainer['model'])
        self.trainer = trainer
        self.parallel = parallel
        self.batch = batch
        self.montage = montage

        if not self.batch:
            self.single_path = single_path

        if self.parallel:
            self.pool = mp.Pool()

    def run(self, img, img_path):

        input_ = []
        for key, func in self.trainer['features'].items():

            if key == 'top_colors':
                if self.parallel:
                    feature = self.pool.apply_async(
                        func,
                        [img, img_path]
                    ).get()

                else:
                    feature = func(img, img_path)

            elif key == 'top_object':
                # annotator module uses unpickable objects
                #   which can't be pooled
                feature = func(img)

            else:
                if self.parallel:
                    feature = self.pool.apply_async(
                        func,
                        [img]
                    ).get()

                else:
                    feature = func(img)

            # if multiple features in one category
            if isinstance(feature, collections.Sequence):
                for item in feature:
                    input_.append(item)
            else:
                input_.append(feature)

        return self.mlp.run(input_=input_)

    def batch_process(self):

        if self.montage:
            to_montage = []

        dir_images = os.path.join(
            os.getcwd(),
            self.trainer['raw_images_root'],
            'test'
        )

        for i, (img, img_path) in enumerate(
            image_batch_loader(dir_=dir_images, limit=None)
        ):
            print(img_path)

            result = self.run(img, img_path)

            print('Run:', result)

            if self.montage:
                put_text(
                    img,
                    [k for k, v in config.emotions_map.items() if v == result][0]
                )

                to_montage.append(img)

        if self.montage:
            montage_ = montage(to_montage)
            show(montage_)

        if self.parallel:
            # release multiprocessing pool
            self.pool.close()

    def single_process(self):

        print(self.single_path)

        img = image_single_loader(self.single_path)
        result = self.run(img, self.single_path)

        print('Run:', result)

        if self.montage:
            put_text(
                img,
                [k for k, v in config.emotions_map.items() if v == result][0]
            )

            show(img)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        action='store',
        dest='model',
        default='oea',
        help='two eliana models available: oea and oea_less'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=False,
        dest='parallel',
        help='enable parallel processing for faster results. default is false'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        default=False,
        dest='batch',
        help='enable batch processing. default is true'
    )
    parser.add_argument(
        '--montage',
        action='store_true',
        default=False,
        dest='montage',
        help='embed result on the image. default is true'
    )
    parser.add_argument(
        '--single_path',
        dest='single_path',
        default=os.path.join(
            os.getcwd(),
            'training',
            'data',
            'images',
            'test',
            'img17.jpg'
        ),
        help='single image path if batch is disabled.'
    )

    args = parser.parse_args()

    if args.model == 'oea':
        trainer = config.trainer_oea

    elif args.model == 'oea_less':
        trainer = config.trainer_oea_less

    enna = Eliana(
        trainer=trainer,
        parallel=args.parallel,
        batch=args.batch,
        montage=args.montage,
        single_path=args.single_path
    )

    print(args.batch)

    if args.batch:
        enna.batch_process()
    else:
        enna.single_process()
