#!/usr/bin/env python

from argparse import ArgumentParser
from importlib import import_module
import logging
import traceback
import os
import sys
from HPOlib.benchmark_util import get_openml_dataset
import HPOlib.cv as cv
import numpy as np
from run_hyperbandSHA import run_hyperbandSHA

logger = logging.getLogger("HPOlib.optimizers.hyperband.SHAcall")



def main():
    # RESTORE IS NOT SUPPORTED WITH THIS OPTIMIZER
    prog = "hyperbandSHA"
    description = "hyperbandSHA"

    parser = ArgumentParser(description=description, prog=prog)

    parser.add_argument("-p", "--space",
                        dest="spaceFile", help="Where is the space.py located?")
    parser.add_argument("-m", "--maxEvals",
                        dest="maxEvals", help="How many evaluations?")
    parser.add_argument("-s", "--seed", default="1",
                        dest="seed", type=int, help="Seed for Hyperband")
    parser.add_argument("-fixB",type=int, help="Fix budget each round")
    parser.add_argument("--random", default=False, action="store_true",
                        dest="random", help="Use a random search")
    parser.add_argument("--cwd", help="Change the working directory before "
                                      "optimizing.")
    parser.add_argument("--tid", type=int, help="Which open_ml task id to use.")
    parser.add_argument("--datadir", help="Where to save the open ml data.")

    args, unknown = parser.parse_known_args()

    if args.cwd:
        os.chdir(args.cwd)

    if not os.path.exists(args.spaceFile):
        logger.critical("Search space not found: %s" % args.spaceFile)
        sys.exit(1)
    # First remove ".py"
    space, ext = os.path.splitext(os.path.basename(args.spaceFile))

    # Then load dict searchSpace and out function cv.py
    sys.path.append("./")
    sys.path.append("")

    # Load Openml data and get number of datapoints to determine input for Hyperband
    X,y = get_openml_dataset(args.tid, args.datadir)
    y = np.atleast_1d(y)
    if y.ndim == 1:
    # reshape is necessary to preserve the data contiguity against vs
    # [:, np.newaxis] that does not.
        y = np.reshape(y, (-1, 1))
    n_obs = y.shape[0]

    # Set minimum and maximum training set size
    min_train_size = max(min(int(2./27.*n_obs),1000),20)
    max_train_size=int(2./3.*n_obs)
    k = 0

    # Use the hyperparameter configuration sampler and infrastructure from hyperopt
    seed_and_arms = args.seed
    module = import_module(space)
    search_space = module.space
    fn = cv.main

    # Run until time budget is consumed, rely on sigterm from HPOlib wrapper
    while True:
        eta = 3.
        def logeta(x):
            return np.log(x)/np.log(eta)

        # s_max defines the number of inner loops per unique value of B
        # it also specifies the maximum number of rounds
        R = float(max_train_size)
        r = float(min_train_size)
        if args.fixB:
            B=(int(logeta(R/r))+1)*max_train_size
        else:
            B = int((2**k)*max_train_size)
        k+=1
        print "\nBudget B = %d" % B
        print '###################'

        ell_max = int(min(B/R-1,int(logeta(R/r))))
        ell = ell_max
        try:
            while ell >= 0:

                # specify the number of arms and the number of times each arm is pulled per stage within this innerloop
                n = int( B/R*eta**ell/(ell+1.) )

                if n> 0:
                    s = 0
                    while (n)*R*(s+1.)*eta**(-s)>B:
                        s+=1

                    print
                    print 's=%d, n=%d' %(s,n)
                    print 'n_i\tr_k'

                    run_hyperbandSHA(n,s,eta,R,fn,search_space,seed_and_arms)
                    seed_and_arms=seed_and_arms+n
                ell-=1
        except Exception as e:
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
