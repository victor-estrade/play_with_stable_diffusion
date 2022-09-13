.DEFAULT_GOAL := help
PROJECT_NAME := stable_diffusion
PYTHON_VERSION := 3.7.5

###############################
# Virtual environment
###############################

init: ## Initiate virtual environment
	@pyenv virtualenv $(PYTHON_VERSION) $(PROJECT_NAME)
	# @touch init

install: ## Install project dependencies
	@pip install --upgrade pip
	@pip install -r requirements.txt
	@pip install -e .
	# @touch install

install-dev: install ## Install project and dev dependencies
	@pip install -r requirements-dev.txt
	# @touch install-dev

kernel:
	@pip install ipykernel
	@python -m ipykernel install --user --name=$(PROJECT_NAME)
	# @touch kernel

###############################
# Test
###############################

test: ## launch tests
	@pytest tests

test-unit: ## launch unit tests
	@pytest tests/unit

test-functional: ## launch functional tests
	@pytest tests/functional

coverage: ## launch test coverage
	@pytest --cov=src tests --cov-report term

coverage-unit: ## launch unit test coverage
	@pytest --cov=src tests/unit --cov-report term

###############################
# Linter
###############################

lint: autopep8 flake8 black isort ## Lint

autopep8:
	@autopep8 --in-place -r src

flake8:
	@flake8 src

black:
	@black src --check

isort:
	@isort src --check

###############################
# Clean
###############################

clean-kernel: ## Clean jupyter kernel
	@jupyter kernelspec uninstall -f $(PROJECT_NAME) 2> /dev/null || true
	@rm -f kernel

clean-env: ## Clean virtual environment
	@pyenv uninstall -f $(PROJECT_NAME)
	@rm -f init install install-dev

clean-miscellaneous: ## Clean miscellaneous
	@rm -rf build *.egg-info */*/__pycache__ */*/*.c */*/*.so || true

clean: clean-kernel clean-env clean-miscellaneous ## Clean all

###############################
# Help
###############################

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
