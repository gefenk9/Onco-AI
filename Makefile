.PHONY: help setup run clean

PYTHON_INTERPRETER ?= python3
VENV_DIR = .venv
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip
VENV_SPACY = $(VENV_DIR)/bin/spacy

PIP_INSTALL = $(VENV_PIP) install

DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make setup           Setup the Python environment and install dependencies"
	@echo "  make clean           Remove generated files (e.g., ${OUTPUT_CSV}, ${XLSX_CONVERT_OUTPUT}) and the virtual environment"
	@echo "  make help            Show this help message"

setup:
	@echo "--- Setting up Python virtual environment and installing dependencies ---"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment in $(VENV_DIR)..."; \
		$(PYTHON_INTERPRETER) -m venv $(VENV_DIR); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment $(VENV_DIR) already exists."; \
	fi
	@echo "Installing dependencies from requirements.txt into $(VENV_DIR)..."
	$(VENV_PIP) install -r requirements.txt
	@echo "--- Python environment setup complete ---"

clean:
	@echo "--- Cleaning generated files ---"
	rm -f $(OUTPUT_CSV) $(XLSX_CONVERT_OUTPUT)
	@echo "Removing virtual environment $(VENV_DIR)..."
	rm -rf $(VENV_DIR)
	@echo "--- Cleaning complete ---"