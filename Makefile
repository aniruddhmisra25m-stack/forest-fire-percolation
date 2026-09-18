PY ?= python3
JOBS ?= 4

.PHONY: install test test-all experiments figures animation ml all

install:            ## install the package and test tools
	$(PY) -m pip install -e ".[dev]"

test:               ## fast tests (~30 s)
	$(PY) -m pytest -q

test-all:           ## include the literature-recovery tests (under a minute in total)
	$(PY) -m pytest -q --runslow

experiments:        ## regenerate every results/ file (~20 min on 2 cores)
	$(PY) scripts/burn_sweep.py --jobs $(JOBS)
	$(PY) scripts/validate_thresholds.py --jobs $(JOBS)
	$(PY) scripts/phase_diagram.py --jobs $(JOBS)
	$(PY) scripts/finite_size_scaling.py --jobs $(JOBS)

figures:            ## redraw every figure from results/ (seconds)
	$(PY) scripts/make_figures.py

animation:          ## the README animation
	$(PY) scripts/make_animation.py

ml:                 ## the superseded random-forest experiment (needs scikit-learn)
	$(PY) exploratory/ml_classifier.py

all: experiments figures animation
