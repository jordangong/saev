docs: lint
    rm -rf docs/saev docs/contrib
    uv run python scripts/docs.py --in-paths saev contrib --out-fpath docs/llms.txt
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True saev contrib

test: lint
    uv run pytest --cov saev -n auto saev

lint: fmt
    fd -e py | xargs ruff check

fmt:
    fd -e py | xargs isort
    fd -e py | xargs ruff format --preview
    fd -e elm | xargs elm-format --yes

clean:
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
