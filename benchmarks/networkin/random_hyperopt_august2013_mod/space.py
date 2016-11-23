"""
Neural Network (NNet) and Deep Belief Network (DBN) search spaces used in [1]
and [2].

The functions in this file return pyll graphs that can be used as the `space`
argument to e.g. `hyperopt.fmin`. The pyll graphs include hyperparameter
constructs (e.g. `hyperopt.hp.uniform`) so `hyperopt.fmin` can perform
hyperparameter optimization.

See ./skdata_learning_algo.py for example usage of these functions.


[1] Bergstra, J.,  Bardenet, R., Bengio, Y., Kegl, B. (2011). Algorithms
for Hyper-parameter optimization, NIPS 2011.

[2] Bergstra, J., Bengio, Y. (2012). Random Search for Hyper-Parameter
Optimization, JMLR 13:281--305.

"""

"""
CHANGED TO WORK AS SEARCHSPACE IN THE BBoM Framework
"""

__author__ = "James Bergstra"
__license__ = "BSD-3"

import numpy as np

from optimizers.tpe.hyperopt_august2013_mod_src.hyperopt import hp


space = {'momentum':hp.uniform('momentum',0.5,1),
        'learning_rate1': hp.loguniform('learning_rate1', np.log(5*10**(-3)), np.log(5*10**(-1))),
        'learning_rate2': hp.loguniform('learning_rate2', np.log(5*10**(-3)), np.log(5*10**(-1))),
        'learning_rate3': hp.loguniform('learning_rate3', np.log(5*10**(-3)), np.log(5*10**(-1))),
        'weight_cost1': hp.loguniform('weight_cost1', np.log(10**(-5)), 0.001),
        'weight_cost2': hp.loguniform('weight_cost2', np.log(10**(-5)), 0.001),
        'weight_cost3': hp.loguniform('weight_cost3', np.log(10**(-5)), 0.001),
        'dropout1':hp.uniform('dropout1',0.4,0.6),
        'dropout2':hp.uniform('dropout2',0.4,0.6),
        'lr_step': hp.quniform('lr_step', 0.5, 3.5,1),
        'w_init1': hp.loguniform('w_init1',np.log(10**(-2)),0.0),
        'w_init2': hp.loguniform('w_init2',np.log(10**(-2)),0.0),
        'w_init3': hp.loguniform('w_init3',np.log(10**(-2)),0.0)
        
        }

