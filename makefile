#!/bin/bash

mode=woia

test:
	python -m unittest eliana.tests.unit_test

doc:
	bash docs/build-docs
	bash docs/make-html

run:
	python -m eliana.eliana ${mode}

train:
	python -m eliana.trainer ${mode}
