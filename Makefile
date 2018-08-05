# OS Detection
OS := $(shell uname -s)
PYTHON_HOME := 'python'
REQUIREMENTS_FILE := 'requirements.txt'

install_requirements: $(PYTHON_HOME)
	@$(PYTHON_HOME)/bin/pip install pipreqs nose
	@$(PYTHON_HOME)/bin/pip install -r $(REQUIREMENTS_FILE)

$(PYTHON_HOME):
	@pip install --upgrade --user virtualenv
	@virtualenv -p `which python3` $(PYTHON_HOME)

install: install_requirements

clean:
	@rm -rf $(PYTHON_HOME)

update_requirements_file:
	#@$(PYTHON_HOME)/bin/pipreqs --use-local --savepath $(REQUIREMENTS_FILE) $(PWD)
	@python_install/bin/pip freeze > requirements.txt

.PHONY: install install_requirements update_requirements_file clean
