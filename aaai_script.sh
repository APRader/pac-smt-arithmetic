#!/bin/bash
source venv/bin/activate
python3 incalp_comparison.py cuben --seed 111921 --timeout 1800
python3 incalp_comparison.py police --seed 111921 --timeout 1800
python3 incalp_comparison.py cuben --seed 111921 --timeout 1800 --noise 0.1
python3 incalp_comparison.py cuben --seed 111921 --timeout 1800 --noise 0.25
python3 incalp_comparison.py cuben --seed 111921 --timeout 1800 --outliers 0.01
python3 incalp_comparison.py pollution --seed 111921 --timeout 1800 --outliers 0.01
python3 incalp_comparison.py police --seed 111921 --timeout 1800 --noise 0.1
python3 incalp_comparison.py police --seed 111921 --timeout 1800 --noise 0.25
python3 incalp_comparison.py simplexn --seed 111921 --timeout 1800 --noise 0.1
python3 incalp_comparison.py simplexn --seed 111921 --timeout 1800 --noise 0.25
python3 incalp_comparison.py simplexn --seed 111921 --timeout 1800 --outliers 0.01