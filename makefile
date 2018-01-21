#!/bin/bash

args=
dataset='./training/data/oea_dataset.pkl'
single_path='./training/data/images/test/img17.jpg'


test:
	python -m unittest eliana.tests.unit_test

doc:
	bash docs/build-docs
	bash docs/make-html

run:
	python -m eliana.eliana ${args}

train:
	python -m eliana.trainer ${model}

view-dataset:
	python -m eliana.view_dataset ${dataset}
