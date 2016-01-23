from optimizers.tpe.hyperopt_august2013_mod_src.hyperopt import hp

space = {'x': hp.uniform('x', -2, 2),
         'y': hp.uniform('y', -1, 1)}
