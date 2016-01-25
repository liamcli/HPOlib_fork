import optimizers.tpe.hyperopt_august2013_mod_src.hyperopt as hyperopt
from functools import partial

def run_hyperbandSHA(n, s, eta, R, fn,search_space, seed):
    domain = hyperopt.Domain(fn, search_space, rseed=seed)
    trials = hyperopt.Trials()
    arms=[]
    for j in range(n):
        arm = hyperopt.tpe.rand.suggest([j], domain, trials, seed=seed)
        vals = arm[0]['misc']['vals']
        # unpack the one-element lists to values
        # and skip over the 0-element lists
        rval = {}
        for k, v in vals.items():
            if v:
                rval[k] = v[0]
        arms.append(hyperopt.space_eval(search_space,rval))
    results=[]
    for i in range(s+1):
        num_pulls = int(R*eta**(i-s))
        num_arms = int( n*eta**(-i))
        fn = partial(fn, train_size=num_pulls)
        print '%d\t%d' %(num_arms,num_pulls)
        for arm in arms:
            results.append([arm,fn(arm)])
        # pick the top results
        n_k1 = int( n*eta**(-i-1) )
        results = sorted(results,key=lambda x: x[1])
        if s-i-1>=0:
            arms= [ x[0] for x in results[0:n_k1] ]
        else:
            break

