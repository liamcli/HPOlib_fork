from optimizers.tpe.hyperopt_august2013_mod_src.hyperopt import hp

space = {'C': hp.quniform('C', 0, 24, 1),
         'alpha': hp.quniform('alpha', 0, 13, 1),
         'epsilon': hp.quniform('epsilon', 0, 3, 1)}
