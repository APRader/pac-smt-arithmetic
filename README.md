# pac-smt-arithmetic
A framework for combining Probably Approximately Correct (PAC) semantics with arithmetic Satisfiability Modulo Theories (SMT).
Queries are expressed as SMT formulas and answered implicitly, i.e. without creating a model, from positive examples. 
You provide a desired validity, and the query will be answered with PAC guarantees of soundness.

## Algorithms


### DecidePAC
The main algorithm for query answering.
It returns True if the query is correct in a proportion of (1-validity) examples.
The more more examples you provide, the higher the confidence
### OptimisePAC
An adaptation of the framework for optimisation problems.
The algorithm returns the estimated optimal value of an objective function given examples.

## Setup
The main code is found in the folder pac and allows you to use DecidePAC and OptimisePAC to answer queries implicitly.
Queries are represented using the Z3Py package. 
You can represent examples as Z3 formulas as well, or use the custom interval class defined in `interval.py`.
For comprehensive usage examples, please see the file `pac_test.py`.

### Required packages
For the functions in the `pac` folder you will only need to install `z3-solver`

For running experiments from the Honours Project 2020 and IJCAI21 folder, you should install the packages given in `requirements.txt`.



## Associated papers
To read more about the framework, you can refer to [Rader, A.P., Mocanu, I.G., Belle, V., & Juba, B.A. (2021). Learning Implicitly with Noisy Data in Linear Arithmetic. IJCAI.](https://doi.org/10.24963/ijcai.2021/195)
The code for the experimetns from that paper can be found in the folder IJCAI21.
An extended version can be found on [arXiv](https://arxiv.org/abs/2010.12619).

The code used for the Bachelor thesis [Rader, A.P. (2020). Learning Implicitly with
Imprecise Data in PAC Semantics. University of Edinburgh.](https://project-archive.inf.ed.ac.uk/ug4/20201893/ug4_proj.pdf) can be found in the Honours Project 2020 folder.

