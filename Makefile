# OS Detection
OS := $(shell uname -s)
PYTHON_HOME := 'python'
REQUIREMENTS_FILE := 'requirements.txt'

install_requirements:
	@$(PYTHON_HOME)/bin/pip install pipreqs nose
	@$(PYTHON_HOME)/bin/pip install -r $(REQUIREMENTS_FILE)

python_install:
	@pip install --upgrade --user virtualenv
	@virtualenv -p `which python3` $(PYTHON_HOME)

install: python_install install_requirements

update_requirements_file:
	@$(PYTHON_HOME)/bin/pipreqs --use-local --savepath $(REQUIREMENTS_FILE) $(PWD)

.PHONY: install install_dev install_lsf install_requirements update_requirements_file tests clean_logs clean_sessions clean_dev clean_all clean_tmp clean_bin clean lsf_install_requirements lsf_python_install lsf_tests lsf_clean lsf_clean_all lsf_clean_logs
