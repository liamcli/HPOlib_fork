"""
Random search - presented as hyperopt.fmin_random
"""
import logging
import pickle
import optimizers.tpe.hyperopt_august2013_mod_src.hyperopt as hyperopt
from optimizers.tpe.hyperopt_august2013_mod_src.hyperopt.base import miscs_update_idxs_vals

logger = logging.getLogger(__name__)


def initial_run(new_ids, domain, trials, seed=123):
    logger.info('generating trials for new_ids: %s' % str(new_ids))

    rval = []
    for new_id in new_ids:
        idxs, vals = pickle.load(open('../hyperopt_init.pkl','rb'))
        new_result = domain.new_result()
        new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
        miscs_update_idxs_vals([new_misc], idxs, vals)
        rval.extend(trials.new_trial_docs([new_id],
                [None], [new_result], [new_misc]))
    return rval

def suggest(new_ids, domain, trials, seed=123):
    if len(trials.trials)==0:
        return initial_run(new_ids,domain,trials,seed)
    return hyperopt.tpe.suggest(new_ids,domain,trials,seed)