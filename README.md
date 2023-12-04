# dtw_metrics

Dynamic time warping metrics.

[![python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pre-commit/action/main.svg)](https://results.pre-commit.ci/latest/github/pre-commit/action/main)
[![linter](https://img.shields.io/badge/code%20linting-pylint-blue.svg)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

Dynamic time warping (DTW) is an algorithm used to measure similarity between different data series, which may vary in speed.

## Setup

### Makefile
Setup via Makefile:
- Clone the repository first.

- Help and overview of commands:
    ```bash
    make help
    ```

- Setup (Create the venv, install libraries and pull minimal data setup):
    ```bash
    make setup
    ```

- Activate Python venv:
    ```bash
    make venv
    ```

- Clean project:
    ```bash
    make clean
    ```

### Python venv (Poetry)

- Poetry is used for the virtual Python environment.
  - Install [python poetry](https://github.com/python-poetry/poetry):
    ```bash
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
    ```
    or
    ```bash
    pip install --user poetry
    ```

  - Make sure that the virtual environment is installed in the root folder of this projects:
    ```bash
    poetry config virtualenvs.in-project true
    ```

  - Install dependencies:
    ```bash
    poetry install --no-root
    ```

  - Add packages:
    To add further packages, run:
    ```bash
    poetry add <package-name>
    ```

### Code formatting
- Formatting via pre-commit hook:
  Python code is formatted using pre-commit hooks.
- Install pre-commit:
  ```bash
  pip install pre-commit
  ```
  or
  ```bash
  brew install pre-commit
  ```

- Install pre-commit hooks from the config file .pre-commit-config.yaml
  ```bash
  pre-commit install
  ```

- Run the pre-commit hooks:
  ```bash
  pre-commit run -a
  ```

### Code testing
Testing is done with [pytest](https://docs.pytest.org/). The pytest package is already installed in the poetry venv.
- Run all tests:

  ```bash
  pytest
  ```

- Run individual tests:

  ```bash
  pytest <path-to-test-files>
  ```

## Usage
- Example files in `/examples/`



## Examples

### Symmetric P1 step pattern

![Example image](/docs/images/example_trig_patternP1_sequence.png "Example time series")
Fig 1: Compared time series and warped sequence.

![Example image](/docs/images/example_trig_patternP1_cm.png "Example cost matrix")
Fig 2: Cost matrix and optimal warping path.

![Example image](/docs/images/example_trig_patternP1_acm.png "Example accumulated cost matrix")
Fig 3: Accumulated cost matrix and optimal warping path.



### Symmetric P0 step pattern

![Example image](/docs/images/example_trig_patternP0_sequence.png "Example time series")
Fig 1: Compared time series and warped sequence.

![Example image](/docs/images/example_trig_patternP0_cm.png "Example cost matrix")
Fig 2: Cost matrix and optimal warping path.

![Example image](/docs/images/example_trig_patternP0_acm.png "Example accumulated cost matrix")
Fig 3: Accumulated cost matrix and optimal warping path.



## References
- Müller, Meinard. Information retrieval for music and motion. Vol. 2. Heidelberg: Springer, 2007. https://doi.org/10.1007/978-3-540-74048-3
