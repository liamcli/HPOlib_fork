from optimizers.tpe.hyperopt_august2013_mod_src.hyperopt import hp

space = {'Kappa': hp.quniform('Kappa', 0, 5, 1),
         'Tau': hp.quniform('Tau', 0, 5, 1),
         'S': hp.quniform('S', 0, 7, 1)}
