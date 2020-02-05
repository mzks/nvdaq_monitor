init:
	pip3 install -r requirements.txt --user

test:
	py.test tests

.PHONY: init test
