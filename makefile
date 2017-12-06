#!/bin/bash

test:
	python -m unittest tests.annotator_unit_test
	python -m unittest tests.color_unit_test
	python -m unittest tests.texture_unit_test

# run:
	# run integrated test
