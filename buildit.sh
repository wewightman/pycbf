#!/bin/bash
rm -r *.egg-info
rm -r build
rm -r dist
mkdir output
python setup.py bdist_wheel
cd dist
auditwheel show pyusel_pycbf*whl 
auditwheel repair pyusel_pycbf*whl 
rm *.whl
cp wheelhouse/*whl .
rm -r wheelhouse
cd ..
python setup.py sdist
