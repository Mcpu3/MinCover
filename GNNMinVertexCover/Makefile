MAKE := make
PYTHON := python
SOURCE := makefile.py
FLAG := --mode

.PHONY: all
all: make_dataset train test

.PHONY: make_dataset
make_dataset:
	$(PYTHON) $(SOURCE) $(FLAG) $@

.PHONY: train
train:
	$(PYTHON) $(SOURCE) $(FLAG) $@

.PHONY: test
test:
	$(PYTHON) $(SOURCE) $(FLAG) $@

.PHONY: clean
clean: clean_make_dataset clean_train clean_test

.PHONY: clean_make_dataset
clean_make_dataset:
	$(PYTHON) $(SOURCE) $(FLAG) $@

.PHONY: clean_train
clean_train:
	$(PYTHON) $(SOURCE) $(FLAG) $@

.PHONY: clean_test
clean_test:
	$(PYTHON) $(SOURCE) $(FLAG) $@