install:
	pip install -r requirements.txt

test:
	python -m pytest -q

figures:
	python experiments/h01_pilot_plot.py
	python experiments/q06_noise_prediction.py
	python experiments/e01_concentration.py

e2e-l0:
	python experiments/e2e_l0.py

.PHONY: install test figures e2e-l0
