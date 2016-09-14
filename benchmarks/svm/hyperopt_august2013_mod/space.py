import numpy as np
from optimizers.tpe.hyperopt_august2013_mod_src.hyperopt import hp


param1=hp.uniform('coef0',-1.0,1.0)
param2=hp.quniform('degree',1.5,4.5,1)
space = {'preprocessor': hp.choice('preprocessor', [1,2,3]), # (min_max, scaled, normalized)
	'kernel': hp.choice("kernel", [{"kernel": 2, "coef0": param1, "degree": param2, }, #poly
    {"kernel": 1, },#rbf
    {"kernel": 3, "coef0": param1, }, ]),#sigmoid
        'C': hp.loguniform('C', np.log(0.001), np.log(100000)),
        'gamma': hp.loguniform('gamma', np.log(0.00001), np.log(10)),
	
        }


