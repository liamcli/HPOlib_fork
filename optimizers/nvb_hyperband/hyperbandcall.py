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
from run_nvb import run_nvb

logger = logging.getLogger("HPOlib.optimizers.nvb.nvbcall")



def main():
    # THIS METHOD DOES NOT SUPPORT RESTORE
    prog = "nvb"
    description = "nvb"

    parser = ArgumentParser(description=description, prog=prog)

    parser.add_argument("-p", "--space",
                        dest="spaceFile", help="Where is the space.py located?")
    parser.add_argument("-m", "--maxEvals",
                        dest="maxEvals", help="How many evaluations?")
    parser.add_argument("-s", "--seed", default="1",
                        dest="seed", type=int, help="Seed for nvb")
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

    module = import_module(space)
    search_space = module.space
    fn = cv.main  # doForTPE

    # Now run TPE, emulate fmin.fmin()
    state_filename = "state.pkl"

    while True:
        X,y = get_openml_dataset(args.tid, args.datadir)
        y = np.atleast_1d(y)
        if y.ndim == 1:
        # reshape is necessary to preserve the data contiguity against vs
        # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        n_obs = y.shape[0]

        min_train_size = min(int(1./12.*n_obs),2000)
        max_train_size=int(2./3.*n_obs)
        k = 0
        seed_and_arms = args.seed
        while True:
            if args.fixB:
                B=4*max_train_size
            else:
                B = int((2**k)*max_train_size)
            k+=1
            print "\nBudget B = %d" % B
            print "\nmin_train_size: " + str(min_train_size) + ", max_train_size: " + str(max_train_size)
            print '###################'
            try:
                num_pulls = int(max_train_size)
                num_arms = int(B/num_pulls)

                while num_pulls>=min_train_size:
                    if num_arms>2:
                        print "Starting num_pulls=%d, num_arms=%d" %(num_pulls,num_arms)
                        best_config = run_nvb(num_arms,num_pulls,fn,search_space,seed_and_arms,state_filename)
                        if num_pulls<max_train_size:
                            # run best_config on full sample size
                            fn(best_config)
                        seed_and_arms = seed_and_arms + num_arms
                    num_pulls = int(num_pulls/2)
                    num_arms = int(B/num_pulls)
            except Exception as e:
                print(traceback.format_exc())

if __name__ == "__main__":
    main()
