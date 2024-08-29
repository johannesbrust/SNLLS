# SNLLS
**Stochastic Nonlinear Least-Squares (for Large-Scale Machine Learning)**

PYTHON 3.7 implementations of algorithms from the article

"Nonlinear Least-Squares for Large-Scale Machine Learning using Stochastic Jacobian Estimates", J.J. Brust (2021),  
*Proceedings of the 38th International Conference on Machine Learning, PMLR 139* [[article]](https://arxiv.org/pdf/2107.05598.pdf)

Content:
  * NLLS.py (Full Jacobian nonlinear least-squares algorithm (small data sizes))
  * SNLLS1.py (Rank-1 Stochastic Jacobian algorithm, large-scale)
  * SNLLSL.py (Rank-L Stochastic Jacobian algorithm, large-scale)
  * SNLSS_1_IRISCLASS.py (Driver Experiment I)
    - Dataset: [[Iris]](https://archive.ics.uci.edu/ml/datasets/iris)
    - Includes: NLLS,SNLLS1,SNLLSL, SGD, Adam, Adagrad
  * SNLLS_2_RANKING.py (Driver Experiment II)
    - Dataset: [[MovieLens]](https://grouplens.org/datasets/movielens/)
    - Includes: SNLLS1,SNLLSL, SGD, Adam, Adagrad
  * SNLLS_3_AUTOENCODE.py (Driver Experiment III)
    - Dataset: [[Fashion MNIST]](https://github.com/zalandoresearch/fashion-mnist)
    - Includes: SNLLS1,SNLLSL, SGD, Adam, Adagrad  
  * README.txt
    
  * DATA/ (stored experiment outcomes)

## Example
You can run a driver using a python console:

```In [1]: import os as os
 In [2]: wd = os.getcwd()
 In [2]: rundir = wd+'/'+'SNLLS_1_IRISCLASS.py'
 In [3]: runfile(rundir, wdir=wd)

Size variables: 193, Size data: 96

Run:0

Epoch   NLLS    SNLLS1  SNLLSL  SGD     ADAM    ADAGRAD 

000     0.4108  0.3068  0.2880  0.4107  0.4192  0.4046
001     0.3805  0.1365  0.1647  0.3585  0.4117  0.4358
002     0.3476  0.1314  0.1177  0.1785  0.3930  0.4288
003     0.3244  0.0897  0.0931  0.1576  0.3786  0.4340
004     0.2907  0.0871  0.0815  0.1482  0.3520  0.4323
005     0.2622  0.0828  0.0736  0.1324  0.3178  0.4306
006     0.2350  0.0645  0.0612  0.2184  0.2849  0.4306
007     0.2119  0.0566  0.0547  0.1373  0.2531  0.4323
008     0.1903  0.0521  0.0505  0.1312  0.2369  0.4323
009     0.1726  0.0710  0.0481  0.1186  0.2254  0.4340
010     0.1568  0.0526  0.0481  0.1278  0.2316  0.4392
011     0.1399  0.0538  0.0416  0.1109  0.2273  0.4358
012     0.1258  0.0432  0.0380  0.1192  0.2230  0.4323
013     0.1129  0.0486  0.0397  0.1444  0.2240  0.4306
.       .       .       .       .       .       .
.       .       .       .       .       .       .
```
## Cite
You can cite this work as (bibtex)

```
@article{snlls21,
  author    = {Johannes J. Brust},
  title     = {Nonlinear Least-Squares for Large-Scale Machine Learning using Stochastic Jacobian Estimates},
  journal   = {arXiv preprint arXiv:1412.6980},
  year      = {2021},
  url       = {https://sites.google.com/view/optml-icml2021/accepted-papers?authuser=0}
}
```

<!--
```
@inproceedings{snlls21,
  author    = {Johannes J. Brust},
  title     = {Nonlinear Least-Squares for Large-Scale Machine Learning using Stochastic Jacobian Estimates},
  booktitle = {Proceedings of the Beyond first-order methods in ML systems
               workshop at the 38th International Conference on Machine
               Learning, {ICML} 2021, 18-24 July 2021, Virtual Event},
  publisher = {{PMLR}},
  year      = {2021},
  url       = {https://sites.google.com/view/optml-icml2021/accepted-papers?authuser=0}
}
```
-->



