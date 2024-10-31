.PHONY: run setup clean lint mypy

run:
	python main.py

setup:
	pip install -r requirements.txt
	python setup.py

clean:
	rm -rf src/__pycache__
	rm -rf src/data/__pycache__
	rm -rf src/model/__pycache__
	rm -rf src/visualization/__pycache__

lint:
	flake8 src/ main.py --max-line-length=100

check: lint