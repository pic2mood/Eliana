#!/bin/bash

test:
	python -m unittest eliana.tests.unit_test

doc:
	bash docs/build-docs
	bash docs/make-html

run:
	# run integrated test
	python -m eliana.eliana

train:
	python -m eliana.trainer
