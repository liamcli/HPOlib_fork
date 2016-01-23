from optimizers.tpe.hyperopt_august2013_mod_src.hyperopt import hp

space = {'x': hp.uniform('x', -5, 10),
         'y': hp.uniform('y', 0, 15)}
