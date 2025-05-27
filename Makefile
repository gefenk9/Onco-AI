.PHONY: help setup run clean

PYTHON_INTERPRETER ?= python3
VENV_DIR = .venv
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip
VENV_SPACY = $(VENV_DIR)/bin/spacy

PIP_INSTALL = $(VENV_PIP) install

OUTPUT_CSV = cases_with_analysis.csv

DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make setup-env       Create virtual environment (if needed) and install Python dependencies (including hebspacy)"
	@echo "  make setup           Setup the Python environment and install dependencies"
	@echo "  make run             Run the main.py script using the venv"
	@echo "  make clean           Remove generated files (e.g., ${OUTPUT_CSV}) and the virtual environment"
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

run: setup-env # Ensure venv is setup before running
	@echo "--- Running main.py script ---"
	$(VENV_PYTHON) main.py
	@echo "--- main.py script finished ---"

clean:
	@echo "--- Cleaning generated files ---"
	rm -f $(OUTPUT_CSV)
	@echo "Removing virtual environment $(VENV_DIR)..."
	rm -rf $(VENV_DIR)
	@echo "--- Cleaning complete ---"