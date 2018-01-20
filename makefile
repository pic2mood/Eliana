#!/bin/bash

model=oea
dataset='./training/data/oea_dataset.pkl'


test:
	python -m unittest eliana.tests.unit_test

doc:
	bash docs/build-docs
	bash docs/make-html

run:
	python -m eliana.eliana ${model}

train:
	python -m eliana.trainer ${model}

view-dataset:
	python -m eliana.view_dataset ${dataset}
