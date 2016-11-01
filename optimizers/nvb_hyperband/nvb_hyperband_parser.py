import logging
import os
import sys

logger = logging.getLogger("HPOlib.optimizers.nvb_hyperband.nvb_hyperband_parser")


def manipulate_config(config):
    if not config.has_section('HYPERBAND'):
        config.add_section('HYPERBAND')

    # optional cases
    if not config.has_option('HYPERBAND', 'space'):
        raise Exception("HYPERBAND:space not specified in .cfg")

    if not config.has_option('HYPERBAND', 'number_evals'):
        config.set('HYPERBAND', 'number_evals', config.get('HPOLIB', 'number_of_jobs'))

    path_to_optimizer = config.get('HYPERBAND', 'path_to_optimizer')
    if not os.path.isabs(path_to_optimizer):
        path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

    path_to_optimizer = os.path.normpath(path_to_optimizer)
    if not os.path.exists(path_to_optimizer):
        logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
        sys.exit(1)

    config.set('HYPERBAND', 'path_to_optimizer', path_to_optimizer)

    return config