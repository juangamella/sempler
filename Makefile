SUITE = all

# Run tests
tests: test examples doctests

test:
ifeq ($(SUITE),all)
	python -m unittest discover sempler.test
else
	python -m unittest sempler.test.$(SUITE)
endif

# Run the doctests
doctests:
	PYTHONPATH=./ python sempler/semi.py
	PYTHONPATH=./ python sempler/anm.py
	PYTHONPATH=./ python sempler/lganm.py
	PYTHONPATH=./ python sempler/normal_distribution.py
	PYTHONPATH=./ python sempler/generators.py

# Run the example scripts in the README
examples:
	PYTHONPATH=./ python docs/anm_example.py
	PYTHONPATH=./ python docs/lganm_example.py
	PYTHONPATH=./ python docs/normal_distribution_example.py
	PYTHONPATH=./ python docs/semi_example.py	

.PHONY: test, tests, examples, doctests
