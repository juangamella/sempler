SUITE = all

# Run tests
tests: test

test: examples
ifeq ($(SUITE),all)
	python -m unittest discover sempler.test
else
	python -m unittest sempler.test.$(SUITE)
endif

# Run the example scripts in the README
examples:
	PYTHONPATH=./ python docs/anm_example.py
	PYTHONPATH=./ python docs/lganm_example.py
	PYTHONPATH=./ python docs/normal_distribution_example.py

.PHONY: test, tests, examples
