
SUITE = all

test:
ifeq ($(SUITE),all)
	python -m unittest discover sempler.test
else
	python -m unittest sempler.test.$(SUITE)
endif

tests: test

.PHONY: test, tests
