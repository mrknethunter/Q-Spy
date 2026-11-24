install:
	pip install -r requirements.txt

test:
	python -m pytest -q

figures:
	python experiments/h01_pilot_plot.py
	python experiments/q06_noise_prediction.py
	python experiments/e01_concentration.py

.PHONY: install test figures
