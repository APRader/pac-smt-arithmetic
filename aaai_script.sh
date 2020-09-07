#!/bin/bash
source venv/bin/activate
python3 incalp_comparison pollution --seed 111921 --noise 0.5
python3 incalp_comparison police --seed 111921 --outliers 0.01
python3 incalp_comparison pollution --seed 111921 --noise 0.1
python3 incalp_comparison police --seed 111921 --outliers 0.005