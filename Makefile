# OS Detection
OS := $(shell uname -s)
PYTHON_HOME := 'python'
REQUIREMENTS_FILE := 'requirements.txt'

install_requirements: python_install
	@$(PYTHON_HOME)/bin/pip install pipreqs nose
	@$(PYTHON_HOME)/bin/pip install -r $(REQUIREMENTS_FILE)

python_install:
	@pip install --upgrade --user virtualenv
	@virtualenv -p `which python3` $(PYTHON_HOME)

install: install_requirements

update_requirements_file:
	@$(PYTHON_HOME)/bin/pipreqs --use-local --savepath $(REQUIREMENTS_FILE) $(PWD)

.PHONY: install install_requirements update_requirements_file
