import cPickle
import optimizers.tpe.hyperopt_august2013_mod_src.hyperopt as hyperopt
from functools import partial
def run_nvb(n_arms, train_size, fn, search_space, seed, state_filename):
    fn = partial(fn, train_size=train_size)
    domain = hyperopt.Domain(fn, search_space, rseed=seed)
    tpe_with_seed = partial(hyperopt.tpe.rand.suggest, seed=seed)
    trials = hyperopt.Trials()
    fh = open(state_filename, "w")
    # By this we probably loose the seed; not too critical for a restart
    cPickle.dump({"trials": trials, "domain": domain}, fh)
    fh.close()

    for i in range(n_arms+1):
        # in exhaust, the number of evaluations is max_evals - num_done
        rval = hyperopt.FMinIter(tpe_with_seed, domain, trials, max_evals=i)
        rval.exhaust()
        fh = open(state_filename, "w")
        cPickle.dump({"trials": trials, "domain": domain}, fh)
        fh.close()

    best = hyperopt.space_eval(search_space,trials.argmin)
    return best

