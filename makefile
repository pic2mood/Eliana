#!/bin/bash

test:
	python -m unittest tests.unit_test_runner

doc:
	bash docs/build-docs
	bash docs/make-html

# run:
	# run integrated test
