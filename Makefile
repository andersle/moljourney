.PHONY:	
	clean
	coverage

coverage:
	pytest --cov-report html --cov=moljourney tests/
 
clean:
	find -name \*.pyc -delete
	find -name \*.pyo -delete
	find -name __pycache__ -delete
	find -name \*.so -delete

