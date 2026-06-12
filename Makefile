.PHONY: install validate test stress lint demo simmap clean

install:        ## editable install of the fiedler package
	pip install -e ".[dev,data]"

validate:       ## prove the from-scratch core against numpy/scipy oracles
	python3 scripts/validate_core.py

test:           ## run the unit test suite
	pytest

stress:         ## P0 rigor gate — randomized fuzz over the spectral core
	python3 scripts/stress_test.py 2000

lint:           ## static checks
	ruff check src tests scripts

demo:           ## run the Fiedler partition demo (writes a figure to results/)
	python3 demos/fiedler_partition.py

simmap:         ## feature-similarity heatmap (pass IMAGE= and QUERY="r c" to customise)
	python3 demos/similarity_map.py $(if $(IMAGE),--image $(IMAGE),) $(if $(QUERY),--query $(QUERY),)

live:           ## live webcam similarity heatmap — 24×24 graph, auto-range, n/c/s keys
	python3 demos/live_similarity.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache *.egg-info src/*.egg-info
