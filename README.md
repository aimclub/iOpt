# iOpt

### Installation 

Create venv 
```shell
python -m venv .venv
source .venv/bin/activate
```

Install lib in dev mode
```shell
(.venv)$ pip install -U -e .[test]
```

### Running tests and linters 

```shell
pytest .
```

```shell
flake8 iOpt test
mypy iOpt
isort --check .
```
