import time
import numpy as np
import os
import six
import signal
import traceback
from HPOlibConfigSpace import configuration_space
import HPOlib.benchmark_util as benchmark_util
import HPOlib.wrapping_util as wrapping_util

from autosklearn.constants import *
from autosklearn.cli.base_interface import _get_base_dict,empty_signal_handler #use get_base_dict in evaluator method to get more details
from autosklearn.util.pipeline import get_configuration_space
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator
from autosklearn.evaluation.cv_evaluator import CVEvaluator
from autosklearn.evaluation.util import calculate_score
from autosklearn.util.backend import Backend
from sklearn import cross_validation

def denormalize_score(score, task, label_num):
    if hasattr(score, '__len__'):
        score = score[ACC_METRIC]

    if (task != MULTICLASS_CLASSIFICATION) or label_num==1:
        base_accuracy = 0.5  # random predictions for binary case
    else:
        base_accuracy = 1. / label_num
    # Normalize: 0 for random, 1 for perfect
    eps = np.float(1e-15)
    #score = (accuracy - base_accuracy) / np.maximum(eps, (1 - base_accuracy))
    new_score = score * np.maximum(eps, (1 - base_accuracy)) + base_accuracy
    return new_score
def sample_data(X,Y,sample_size,task,seed=42):
    num_data_points = X.shape[0]
    num_labels = Y.shape[1] if len(Y.shape) > 1 else 1
    X_train, X_valid, Y_train, Y_valid = None, None, None, None
    if X.shape[0] != Y.shape[0]:
        raise ValueError('The first dimension of the X and Y array must '
                         'be equal.')

    # If one class only has one sample, put it into the training set
    if task==BINARY_CLASSIFICATION:
        classes, y_indices = np.unique(Y, return_inverse=True)
        if np.min(np.bincount(y_indices)) < 2:
            classes_with_one_sample = np.bincount(y_indices) < 2
            sample_idxs = []
            Y_old = Y
            indices = np.ones(Y.shape, dtype=bool)
            for i, class_with_one_sample in enumerate(classes_with_one_sample):
                if not class_with_one_sample:
                    continue
                sample_idx = np.argwhere(Y == classes[i])[0][0]
                indices[sample_idx] = False
                sample_idxs.append(sample_idx)
            Y = Y[indices]

    if num_labels > 1:
        sss = None
    else:
        try:
            sss = cross_validation.StratifiedShuffleSplit(
                Y,
                n_iter=1,
                test_size=None,
                train_size=sample_size,
                random_state=seed)
        except ValueError:
            sss = None
    if sss is None:
        sss = cross_validation.ShuffleSplit(Y.shape[0],
                                                    n_iter=1,
                                                    test_size=None,
                                                    train_size=sample_size,
                                                    random_state=seed)

    assert len(sss) == 1, 'Splitting data went wrong'

    for train_index, valid_index in sss:
        if task==BINARY_CLASSIFICATION:
            try:
                Y = Y_old
                for sample_idx in sorted(sample_idxs):
                    train_index[train_index >= sample_idx] += 1
                    valid_index[valid_index >= sample_idx] += 1
                for sample_idx in sample_idxs:
                    train_index = np.append(train_index, np.array(sample_idx))
            except UnboundLocalError:
                pass

        X_train, X_valid = X[train_index], X[valid_index]
        Y_train, Y_valid = Y[train_index], Y[valid_index]

    assert X_train.shape[0] + X_valid.shape[0] == num_data_points
    assert Y_train.shape[0] + Y_valid.shape[0] == num_data_points

    return X_train, Y_train

class ValTestEvaluator(AbstractEvaluator):

    def __init__(self, datamanager, configuration=None,
                 with_predictions=False,
                 all_scoring_functions=False,
                 seed=1,
                 output_dir=None,
                 output_y_test=False,
                 num_run=None):
        super(ValTestEvaluator, self).__init__(
            datamanager, configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_dir=output_dir,
            output_y_test=output_y_test,
            num_run=num_run)

        self.task = datamanager.info['task']
        self.X_train = datamanager.data['X_train']
        self.Y_train = datamanager.data['Y_train']
        self.X_optimization = datamanager.data['X_valid']
        self.Y_optimization = datamanager.data['Y_valid']
        #X_valid and X_test already set by base class
        self.Y_valid = datamanager.data['Y_valid']
        self.Y_test = datamanager.data['Y_test']
        self.label_num = 1 if len(self.Y_test.shape)==1 else self.Y_test.shape[1]


    def fit(self):
        self.model.fit(self.X_train, self.Y_train)

    def iterative_fit(self):
        Xt, fit_params = self.model.pre_transform(self.X_train, self.Y_train)
        if not self.model.estimator_supports_iterative_fit():
            print("Model does not support iterative_fit(), reverting to " \
                "regular fit().")

            self.model.fit_estimator(Xt, self.Y_train, **fit_params)
            return

        n_iter = 1
        while not self.model.configuration_fully_fitted():
            self.model.iterative_fit(Xt, self.Y_train, n_iter=n_iter,
                                     **fit_params)
            self.file_output()
            n_iter += 2

    def predict_test(self):
        if self.X_test is not None:
            Y_test_pred = self.predict_function(self.X_test, self.model,
                                                self.task_type)
        else:
            Y_test_pred = None

        err_test = 1 - denormalize_score(calculate_score(
            self.Y_test, Y_test_pred, self.task_type,
            self.metric, self.D.info['label_num'],
            all_scoring_functions=self.all_scoring_functions),self.task,self.label_num)
        return err_test
    def predict(self):
        Y_optimization_pred = self.predict_function(self.X_optimization,
                                                    self.model, self.task_type)
        if self.X_test is not None:
            Y_test_pred = self.predict_function(self.X_test, self.model,
                                                self.task_type)
        else:
            Y_test_pred = None

        err_val = 1 - denormalize_score(calculate_score(
            self.Y_optimization, Y_optimization_pred, self.task_type,
            self.metric, self.D.info['label_num'],
            all_scoring_functions=self.all_scoring_functions),self.task,self.label_num)



        err_test = 1 - denormalize_score(calculate_score(
            self.Y_test, Y_test_pred, self.task_type,
            self.metric, self.D.info['label_num'],
            all_scoring_functions=self.all_scoring_functions),self.task,self.label_num)
        return err_val, err_test

def main(params, tid, openml_data_dir, split_seed,do_cv,**kwargs):
    print 'Params: ', params

    # digits = sklearn.datasets.load_digits()
    # X = digits.data
    # y = digits.target
    # indices = np.arange(X.shape[0])
    # np.random.shuffle(indices)
    # X = X[indices]
    # y = y[indices]
    X,y = benchmark_util.get_openml_dataset(tid, openml_data_dir)
    y = np.atleast_1d(y)
    compute_test_error = True
    test_error=1
    if y.ndim == 1:
    # reshape is necessary to preserve the data contiguity against vs
    # [:, np.newaxis] that does not.
        y = np.reshape(y, (-1, 1))
    #X_train = X[:1000]
    #y_train = y[:1000]
    #X_test = X[1000:]
    #y_test = y[1000:]
    n_obs = y.shape[0]
    np.random.seed(1)
    shuffle_ind = np.random.permutation(n_obs)
    X=X[shuffle_ind]
    y=y[shuffle_ind]
    classes=[]
    if len(y.shape)==1:
        n_outputs = 1
    else:
        n_outputs = y.shape[1]
    for k in six.moves.range(n_outputs):
        classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
        classes.append(classes_k)

    if n_outputs > 1:
        task = MULTILABEL_CLASSIFICATION
    else:
        if len(classes[0]) == 2:
            task = BINARY_CLASSIFICATION
        else:
            task = MULTICLASS_CLASSIFICATION
    if y.shape[1] == 1:
        y = y.flatten()
    #Do 10% val and 10% test
    val_index = int(2./3.*n_obs)
    test_index = int(5./6.*n_obs)
    if params.has_key('train_size'):
        print "Dataset size: " + str(n_obs) + ", Training size: " + str(params['train_size'])
        if int(params['train_size']) < val_index:
            compute_test_error=False
            X_train, Y_train = sample_data(X[0:val_index],y[0:val_index],int(params['train_size']),task)
            while len(np.unique(Y_train))<len(np.unique(y)):
                shuffle_ind = np.random.permutation(n_obs)
                X=X[shuffle_ind]
                y=y[shuffle_ind]
                X_train, Y_train = sample_data(X[0:val_index],y[0:val_index],int(params['train_size']),task,split_seed)
        params.pop('train_size')
    else:
        X_train=X[0:val_index]
        Y_train=y[0:val_index]
    X_valid = X[val_index:test_index]
    Y_valid = y[val_index:test_index]
    X_test = X[test_index:]
    Y_test = y[test_index:]

    D = XYDataManager(X_train, Y_train, task=task,
                            metric=ACC_METRIC,
                            feat_type=None,
                            dataset_name=None,
                            encode_labels=False)

    if params is not None:
        for key in params:
            try:
                params[key] = int(params[key])
            except Exception:
                try:
                    params[key] = float(params[key])
                    if params[key]-np.floor(params[key])==0:
                        params[key]=int(params[key])
                except Exception:
                    pass


        cs = get_configuration_space(D.info)
        configuration = configuration_space.Configuration(cs,params)
    else:
        configuration = None
    global evaluator
    if do_cv:
        D.data['X_test']=np.concatenate((X_valid,X_test))
        D.data['Y_test']=np.concatenate((Y_valid,Y_test))
        D.data['X_valid']=X_valid
        D.data['Y_valid']=Y_valid
        evaluator = CVEvaluator(D,configuration,cv_folds=10,seed=1,num_run=1)
        evaluator.fit()
        signal.signal(15, empty_signal_handler)
        backend = Backend(None, os.getcwd())
        if os.path.exists(backend.get_model_dir()):
            backend.save_model(evaluator.model, 1, 1)
        try:
            val_error = evaluator.predict()
            val_error = 1-denormalize_score(1-val_error,task,n_outputs)
        except Exception as e:
            print(traceback.format_exc())
        if compute_test_error:
            evaluator = ValTestEvaluator(D, configuration,
                                 seed=1,
                                 num_run=1,with_predictions=True)
            evaluator.fit()
            signal.signal(15, empty_signal_handler)
            backend = Backend(None, os.getcwd())
            if os.path.exists(backend.get_model_dir()):
                backend.save_model(evaluator.model, 1, 1)
            try:
                test_error=evaluator.predict_test()
            except Exception as e:
                print(traceback.format_exc())
        return val_error, test_error,len(Y_train)

    else:
        D.data['X_valid'] = X_valid
        D.data['Y_valid'] = Y_valid
        D.data['X_test'] = X_test
        D.data['Y_test'] = Y_test
        evaluator = ValTestEvaluator(D, configuration,
                                     seed=1,
                                     num_run=1,with_predictions=True)
        evaluator.fit()
        signal.signal(15, empty_signal_handler)
        backend = Backend(None, os.getcwd())
        if os.path.exists(backend.get_model_dir()):
            backend.save_model(evaluator.model, 1, 1)
        try:
            val_error,test_error = evaluator.predict()
            return val_error, test_error, len(Y_train)
        except Exception as e:
            print(traceback.format_exc())



if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    config = wrapping_util.load_experiment_config_file()
    openml_data_dir = config.get("EXPERIMENT", "openml_data_dir")
    tid = np.int(config.get("EXPERIMENT", "openml_tid"))
    split_seed = np.int(config.get("EXPERIMENT", "data_split_seed"))
    val_error, test_error, train_size = main(params, tid, openml_data_dir,split_seed,False,**args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s, %f, %s, %d" % \
        ("SAT", abs(duration), val_error, -1, "test_error", test_error, "train_size", train_size)
