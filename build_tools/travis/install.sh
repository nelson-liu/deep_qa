#!/bin/bash
set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for setup of conda itself
pushd .
cd
mkdir -p download
cd download
if [[ ! -d /home/travis/miniconda3 ]]
    then
    if [[ ! -f miniconda.sh ]]
        then
            wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
                -O miniconda.sh
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda3/bin:$PATH
    echo $PATH
    conda update --yes conda
    # If we are running pylint, use Python 3.5.2 due to
    # bug in pylint. https://github.com/PyCQA/pylint/issues/1295
    if [[ "$RUN_PYLINT" == "true" ]]; then
        conda create -n testenv --yes python=3.5.2
    else
        conda create -n testenv --yes python=3.5
    fi
fi
cd ..
popd

# Activate the python environment we created.
source activate testenv

# Install requirements via pip in our conda environment
pip install -r requirements.txt
