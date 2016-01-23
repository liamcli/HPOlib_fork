import cPickle
import logging
import os
import sys
import HPOlib.wrapping_util as wrappingUtil

__authors__ = ["Lisha Li "]


logger = logging.getLogger("HPOlib.optimizers.nvb_hyperband")


def check_dependencies():
    try:
        import nose
        logger.debug("\tNose: %s\n" % str(nose.__version__))
    except ImportError:
        raise ImportError("Nose cannot be imported. Are you sure it's "
                          "installed?")
    try:
        import networkx
        logger.debug("\tnetworkx: %s\n" % str(networkx.__version__))
    except ImportError:
        raise ImportError("Networkx cannot be imported. Are you sure it's "
                          "installed?")
    try:
        import pymongo
        logger.debug("\tpymongo: %s\n" % str(pymongo.version))
        from bson.objectid import ObjectId
    except ImportError:
        raise ImportError("Pymongo cannot be imported. Are you sure it's"
                          " installed?")
    try:
        import numpy
        logger.debug("\tnumpy: %s" % str(numpy.__version__))
    except ImportError:
        raise ImportError("Numpy cannot be imported. Are you sure that it's"
                          " installed?")
    try:
        import scipy
        logger.debug("\tscipy: %s" % str(scipy.__version__))
    except ImportError:
        raise ImportError("Scipy cannot be imported. Are you sure that it's"
                          " installed?")



def build_hyperband_call(config, options, optimizer_dir):
    call = "python " + os.path.dirname(os.path.realpath(__file__)) + \
           "/hyperbandcall.py"
    openml_data_dir = config.get("EXPERIMENT", "openml_data_dir")
    tid = int(config.get("EXPERIMENT", "openml_tid"))
    call = ' '.join([call, '-p', os.path.join(optimizer_dir, os.path.basename(config.get('HYPERBAND', 'space'))),
                     "-m", config.get('HYPERBAND', 'number_evals'),
                     "-s", str(options.seed),
                     "--cwd", optimizer_dir,
                     "--tid",str(tid),
                     "--datadir",openml_data_dir])
    return call


# noinspection PyUnusedLocal
def main(config, options, experiment_dir, experiment_directory_prefix, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed
    # experiment_dir:   Experiment directory/Benchmarkdirectory
    # **kwargs:         Nothing so far
    time_string = wrappingUtil.get_time_string()
    cmd = ""

    # Add path_to_optimizer to PYTHONPATH and to sys.path
    if not 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = config.get('HYPERBAND', 'path_to_optimizer')
    else:
        os.environ['PYTHONPATH'] = config.get('HYPERBAND', 'path_to_optimizer') + os.pathsep + os.environ['PYTHONPATH']
    sys.path.append(config.get('HYPERBAND', 'path_to_optimizer'))
    optimizer_str = os.path.splitext(os.path.basename(__file__))[0]

    # Find experiment directory
    optimizer_dir = os.path.join(experiment_dir,
                                 experiment_directory_prefix
                                 + optimizer_str + "_" +
                                 str(options.seed) + "_" +
                                 time_string)

    # Build call
    cmd = build_hyperband_call(config, options, optimizer_dir)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
        space = config.get('HYPERBAND', 'space')
        abs_space = os.path.abspath(space)
        parent_space = os.path.join(experiment_dir, optimizer_str, space)
        if os.path.exists(abs_space):
            space = abs_space
        elif os.path.exists(parent_space):
            space = parent_space
        else:
            raise Exception("HYPERBAND search space not found. Searched at %s and "
                            "%s" % (abs_space, parent_space))
        # Copy the hyperopt search space
        if not os.path.exists(os.path.join(optimizer_dir, os.path.basename(space))):
            os.symlink(os.path.join(experiment_dir, optimizer_str, space),
                       os.path.join(optimizer_dir, os.path.basename(space)))

    import hyperopt
    path_to_loaded_optimizer = os.path.abspath(os.path.dirname(os.path.dirname(hyperopt.__file__)))

    logger.info("### INFORMATION ################################################################")
    logger.info("# You are running:                                                             #")
    logger.info("# %76s #" % path_to_loaded_optimizer)
    logger.info("################################################################################")

    return cmd, optimizer_dir
