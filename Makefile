# Default Python (will be overridden on Windows)
PYTHON = python3

# Detect OS
ifeq ($(OS),Windows_NT)
	PYTHON = python
	SET_PYTHONPATH = set PYTHONPATH=src&&
else
	SET_PYTHONPATH = PYTHONPATH=src
endif

.PHONY: run install

# Run the project
run:
	$(SET_PYTHONPATH) $(PYTHON) -m main

# Evaluate a chosen model
eval:
	$(SET_PYTHONPATH) $(PYTHON) -m load-model

# Install dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt
