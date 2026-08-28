.DEFAULT_GOAL := default

#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y radio_ai || :
	@pip install -e .

run:
	python -m radio_ai_package.interface.main
