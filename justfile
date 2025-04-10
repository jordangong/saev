docs: lint
    rm -rf docs/saev docs/contrib
    uv run scripts/docs.py --in-paths saev contrib --out-fpath docs/llms.txt
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True saev contrib

test: lint
    uv run pytest --cov saev --cov-report term --cov-report json --json-report --json-report-file pytest.json -n auto saev || true
    uv run coverage-badge -o docs/coverage.svg -f
    uv run scripts/regressions.py

lint: fmt
    uv run ruff check --fix .

fmt:
    uv run ruff format --preview .
    fd -e elm | xargs elm-format --yes

clean:
    rm -f .coverage
    rm -f docs/coverage.svg
    rm -f coverage.json
    rm -f pytest.json
    rm -rf .hypothesis
    uv run python -c 'import datasets; print(datasets.load_dataset("ILSVRC/imagenet-1k").cleanup_cache_files())'

build-semseg: fmt
    cd web && elm make src/Semseg.elm --output apps/semseg/dist/app.js --optimize
    cd web && tailwindcss --input apps/semseg/main.css --output apps/semseg/dist/main.css --minify

build-classification: fmt
    cd web && elm make src/Classification.elm --output apps/classification/dist/app.js --optimize
    cd web && tailwindcss --input apps/classification/main.css --output apps/classification/dist/main.css --minify

build-comparison: fmt
    cd web && elm make src/Comparison.elm --output apps/comparison/dist/app.js --optimize
    cd web && tailwindcss --input apps/comparison/main.css --output apps/comparison/dist/main.css

deploy: build-classification build-semseg
    uv run python scripts/deploy.py
