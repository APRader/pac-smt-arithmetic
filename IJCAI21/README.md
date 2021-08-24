All experiments can be run from the command line using the file incalp_comparison.py. It uses the following structure:

```
python3 incalp_comparison.py problem [optional commands]
```

where "problem" is either simplexn, cuben, pollution or police.

Optional commands:
```
-s, --seed: The seed for the random number generator.
-n, --noise: Add Gaussian noise to samples with given standard deviation, normalised by number of dimensions.
-o, --outliers: Ratio of outliers, between 0 and 1.
-t, --timeout: Timeout for IncaLP in seconds. Only works for Unix-based systems.
-v, --verbose: Turn on verbose mode.
```

For further information, please refer to the paper to [Rader, A.P., Mocanu, I.G., Belle, V., & Juba, B.A. (2021). Learning Implicitly with Noisy Data in Linear Arithmetic. IJCAI.](https://doi.org/10.24963/ijcai.2021/195) or the extended version on [arXiv](https://arxiv.org/abs/2010.12619).