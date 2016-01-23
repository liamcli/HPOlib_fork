from optimizers.tpe.hyperopt_august2013_mod_src.hyperopt import hp
import optimizers.tpe.hyperopt_august2013_mod_src.hyperopt.pyll as pyll

param_0 = hp.choice("balancing:strategy", [
    {"balancing:strategy": "none", },
    {"balancing:strategy": "weighting", },
    ])
param_1 = hp.uniform("LOG10_Q1_classifier:k_nearest_neighbors:n_neighbors", -0.301021309861, 2.00216606176)
param_2 = hp.uniform("LOG10_Q1_classifier:passive_aggressive:n_iter", 0.653213478873, 3.00021709297)
param_3 = hp.uniform("LOG10_Q1_classifier:sgd:n_iter", 0.653213478873, 3.00021709297)
param_4 = hp.uniform("LOG10_classifier:adaboost:learning_rate", -4.0, 0.301029995664)
param_5 = hp.uniform("LOG10_classifier:gradient_boosting:learning_rate", -4.0, 0.0)
param_6 = hp.uniform("LOG10_classifier:lda:tol", -5.0, -1.0)
param_7 = hp.uniform("LOG10_classifier:liblinear_svc:C", -1.50514997832, 4.51544993496)
param_8 = hp.uniform("LOG10_classifier:liblinear_svc:tol", -5.0, -1.0)
param_9 = hp.uniform("LOG10_classifier:libsvm_svc:C", -1.50514997832, 4.51544993496)
param_10 = hp.uniform("LOG10_classifier:libsvm_svc:gamma", -4.51544993496, 0.903089986992)
param_11 = hp.uniform("LOG10_classifier:libsvm_svc:tol", -5.0, -1.0)
param_12 = hp.uniform("LOG10_classifier:multinomial_nb:alpha", -2.0, 2.0)
param_13 = hp.uniform("LOG10_classifier:passive_aggressive:C", -5.0, 1.0)
param_14 = hp.uniform("LOG10_classifier:sgd:alpha", -6.0, -1.0)
param_15 = hp.choice("classifier:adaboost:algorithm", [
    {"classifier:adaboost:algorithm": "SAMME", },
    {"classifier:adaboost:algorithm": "SAMME.R", },
    ])
param_16 = pyll.scope.int(hp.quniform("classifier:adaboost:max_depth", 0.50001, 10.5, 1.0))
param_17 = pyll.scope.int(hp.quniform("classifier:adaboost:n_estimators", 49.50001, 500.5, 1.0))
param_18 = hp.choice("classifier:decision_tree:criterion", [
    {"classifier:decision_tree:criterion": "entropy", },
    {"classifier:decision_tree:criterion": "gini", },
    ])
param_19 = hp.uniform("classifier:decision_tree:max_depth", 0.0, 2.0)
param_20 = hp.choice("classifier:decision_tree:max_features", [
    {"classifier:decision_tree:max_features": "1.0", },
    ])
param_21 = hp.choice("classifier:decision_tree:max_leaf_nodes", [
    {"classifier:decision_tree:max_leaf_nodes": "None", },
    ])
param_22 = pyll.scope.int(hp.quniform("classifier:decision_tree:min_samples_leaf", 0.50001, 20.5, 1.0))
param_23 = pyll.scope.int(hp.quniform("classifier:decision_tree:min_samples_split", 1.50001, 20.5, 1.0))
param_24 = hp.choice("classifier:decision_tree:min_weight_fraction_leaf", [
    {"classifier:decision_tree:min_weight_fraction_leaf": "0.0", },
    ])
param_25 = hp.choice("classifier:decision_tree:splitter", [
    {"classifier:decision_tree:splitter": "best", },
    ])
param_26 = hp.choice("classifier:extra_trees:bootstrap", [
    {"classifier:extra_trees:bootstrap": "False", },
    {"classifier:extra_trees:bootstrap": "True", },
    ])
param_27 = hp.choice("classifier:extra_trees:criterion", [
    {"classifier:extra_trees:criterion": "entropy", },
    {"classifier:extra_trees:criterion": "gini", },
    ])
param_28 = hp.choice("classifier:extra_trees:max_depth", [
    {"classifier:extra_trees:max_depth": "None", },
    ])
param_29 = hp.uniform("classifier:extra_trees:max_features", 0.5, 5.0)
param_30 = pyll.scope.int(hp.quniform("classifier:extra_trees:min_samples_leaf", 0.50001, 20.5, 1.0))
param_31 = pyll.scope.int(hp.quniform("classifier:extra_trees:min_samples_split", 1.50001, 20.5, 1.0))
param_32 = hp.choice("classifier:extra_trees:min_weight_fraction_leaf", [
    {"classifier:extra_trees:min_weight_fraction_leaf": "0.0", },
    ])
param_33 = hp.choice("classifier:extra_trees:n_estimators", [
    {"classifier:extra_trees:n_estimators": "100", },
    ])
param_34 = hp.choice("classifier:gradient_boosting:loss", [
    {"classifier:gradient_boosting:loss": "deviance", },
    ])
param_35 = pyll.scope.int(hp.quniform("classifier:gradient_boosting:max_depth", 0.50001, 10.5, 1.0))
param_36 = hp.uniform("classifier:gradient_boosting:max_features", 0.5, 5.0)
param_37 = hp.choice("classifier:gradient_boosting:max_leaf_nodes", [
    {"classifier:gradient_boosting:max_leaf_nodes": "None", },
    ])
param_38 = pyll.scope.int(hp.quniform("classifier:gradient_boosting:min_samples_leaf", 0.50001, 20.5, 1.0))
param_39 = pyll.scope.int(hp.quniform("classifier:gradient_boosting:min_samples_split", 1.50001, 20.5, 1.0))
param_40 = hp.choice("classifier:gradient_boosting:min_weight_fraction_leaf", [
    {"classifier:gradient_boosting:min_weight_fraction_leaf": "0.0", },
    ])
param_41 = hp.choice("classifier:gradient_boosting:n_estimators", [
    {"classifier:gradient_boosting:n_estimators": "100", },
    ])
param_42 = hp.uniform("classifier:gradient_boosting:subsample", 0.01, 1.0)
param_43 = hp.choice("classifier:k_nearest_neighbors:p", [
    {"classifier:k_nearest_neighbors:p": "1", },
    {"classifier:k_nearest_neighbors:p": "2", },
    ])
param_44 = hp.choice("classifier:k_nearest_neighbors:weights", [
    {"classifier:k_nearest_neighbors:weights": "distance", },
    {"classifier:k_nearest_neighbors:weights": "uniform", },
    ])
param_45 = pyll.scope.int(hp.quniform("classifier:lda:n_components", 0.50001, 250.5, 1.0))
param_46 = hp.uniform("classifier:lda:shrinkage_factor", 0.0, 1.0)
param_47 = hp.choice("classifier:lda:shrinkage", [
    {"classifier:lda:shrinkage": "None", },
    {"classifier:lda:shrinkage": "auto", },
    {"classifier:lda:shrinkage": "manual", "classifier:lda:shrinkage_factor": param_46, },
    ])
param_48 = hp.choice("classifier:liblinear_svc:dual", [
    {"classifier:liblinear_svc:dual": "False", },
    ])
param_49 = hp.choice("classifier:liblinear_svc:fit_intercept", [
    {"classifier:liblinear_svc:fit_intercept": "True", },
    ])
param_50 = hp.choice("classifier:liblinear_svc:intercept_scaling", [
    {"classifier:liblinear_svc:intercept_scaling": "1", },
    ])
param_51 = hp.choice("classifier:liblinear_svc:loss", [
    {"classifier:liblinear_svc:loss": "hinge", },
    {"classifier:liblinear_svc:loss": "squared_hinge", },
    ])
param_52 = hp.choice("classifier:liblinear_svc:multi_class", [
    {"classifier:liblinear_svc:multi_class": "ovr", },
    ])
param_53 = hp.choice("classifier:liblinear_svc:penalty", [
    {"classifier:liblinear_svc:penalty": "l1", },
    {"classifier:liblinear_svc:penalty": "l2", },
    ])
param_54 = hp.uniform("classifier:libsvm_svc:coef0", -1.0, 1.0)
param_55 = pyll.scope.int(hp.quniform("classifier:libsvm_svc:degree", 0.50001, 5.5, 1.0))
param_56 = hp.choice("classifier:libsvm_svc:kernel", [
    {"classifier:libsvm_svc:kernel": "poly", "classifier:libsvm_svc:coef0": param_54, "classifier:libsvm_svc:degree": param_55, },
    {"classifier:libsvm_svc:kernel": "rbf", },
    {"classifier:libsvm_svc:kernel": "sigmoid", "classifier:libsvm_svc:coef0": param_54, },
    ])
param_57 = hp.choice("classifier:libsvm_svc:max_iter", [
    {"classifier:libsvm_svc:max_iter": "-1", },
    ])
param_58 = hp.choice("classifier:libsvm_svc:shrinking", [
    {"classifier:libsvm_svc:shrinking": "False", },
    {"classifier:libsvm_svc:shrinking": "True", },
    ])
param_59 = hp.choice("classifier:multinomial_nb:fit_prior", [
    {"classifier:multinomial_nb:fit_prior": "False", },
    {"classifier:multinomial_nb:fit_prior": "True", },
    ])
param_60 = hp.choice("classifier:passive_aggressive:fit_intercept", [
    {"classifier:passive_aggressive:fit_intercept": "True", },
    ])
param_61 = hp.choice("classifier:passive_aggressive:loss", [
    {"classifier:passive_aggressive:loss": "hinge", },
    {"classifier:passive_aggressive:loss": "squared_hinge", },
    ])
param_62 = pyll.scope.int(hp.quniform("classifier:proj_logit:max_epochs", 0.50001, 20.5, 1.0))
param_63 = hp.uniform("classifier:qda:reg_param", 0.0, 10.0)
param_64 = hp.choice("classifier:random_forest:bootstrap", [
    {"classifier:random_forest:bootstrap": "False", },
    {"classifier:random_forest:bootstrap": "True", },
    ])
param_65 = hp.choice("classifier:random_forest:criterion", [
    {"classifier:random_forest:criterion": "entropy", },
    {"classifier:random_forest:criterion": "gini", },
    ])
param_66 = hp.choice("classifier:random_forest:max_depth", [
    {"classifier:random_forest:max_depth": "None", },
    ])
param_67 = hp.uniform("classifier:random_forest:max_features", 0.5, 5.0)
param_68 = hp.choice("classifier:random_forest:max_leaf_nodes", [
    {"classifier:random_forest:max_leaf_nodes": "None", },
    ])
param_69 = pyll.scope.int(hp.quniform("classifier:random_forest:min_samples_leaf", 0.50001, 20.5, 1.0))
param_70 = pyll.scope.int(hp.quniform("classifier:random_forest:min_samples_split", 1.50001, 20.5, 1.0))
param_71 = hp.choice("classifier:random_forest:min_weight_fraction_leaf", [
    {"classifier:random_forest:min_weight_fraction_leaf": "0.0", },
    ])
param_72 = hp.choice("classifier:random_forest:n_estimators", [
    {"classifier:random_forest:n_estimators": "100", },
    ])
param_73 = hp.choice("classifier:sgd:average", [
    {"classifier:sgd:average": "False", },
    {"classifier:sgd:average": "True", },
    ])
param_74 = hp.uniform("classifier:sgd:eta0", 1e-07, 0.1)
param_75 = hp.choice("classifier:sgd:fit_intercept", [
    {"classifier:sgd:fit_intercept": "True", },
    ])
param_76 = hp.uniform("classifier:sgd:power_t", 1e-05, 1.0)
param_77 = hp.choice("classifier:sgd:learning_rate", [
    {"classifier:sgd:learning_rate": "constant", },
    {"classifier:sgd:learning_rate": "invscaling", "classifier:sgd:power_t": param_76, },
    {"classifier:sgd:learning_rate": "optimal", },
    ])
param_78 = hp.uniform("LOG10_classifier:sgd:epsilon", -5.0, -1.0)
param_79 = hp.choice("classifier:sgd:loss", [
    {"classifier:sgd:loss": "hinge", },
    {"classifier:sgd:loss": "log", },
    {"classifier:sgd:loss": "modified_huber", "LOG10_classifier:sgd:epsilon": param_78, },
    {"classifier:sgd:loss": "perceptron", },
    {"classifier:sgd:loss": "squared_hinge", },
    ])
param_80 = hp.uniform("LOG10_classifier:sgd:l1_ratio", -9.0, 0.0)
param_81 = hp.choice("classifier:sgd:penalty", [
    {"classifier:sgd:penalty": "elasticnet", "LOG10_classifier:sgd:l1_ratio": param_80, },
    {"classifier:sgd:penalty": "l1", },
    {"classifier:sgd:penalty": "l2", },
    ])
param_82 = hp.choice("classifier:__choice__", [
    {"classifier:__choice__": "adaboost", "LOG10_classifier:adaboost:learning_rate": param_4, "classifier:adaboost:algorithm": param_15, "classifier:adaboost:max_depth": param_16, "classifier:adaboost:n_estimators": param_17, },
    {"classifier:__choice__": "decision_tree", "classifier:decision_tree:criterion": param_18, "classifier:decision_tree:max_depth": param_19, "classifier:decision_tree:max_features": param_20, "classifier:decision_tree:max_leaf_nodes": param_21, "classifier:decision_tree:min_samples_leaf": param_22, "classifier:decision_tree:min_samples_split": param_23, "classifier:decision_tree:min_weight_fraction_leaf": param_24, "classifier:decision_tree:splitter": param_25, },
    {"classifier:__choice__": "extra_trees", "classifier:extra_trees:bootstrap": param_26, "classifier:extra_trees:criterion": param_27, "classifier:extra_trees:max_depth": param_28, "classifier:extra_trees:max_features": param_29, "classifier:extra_trees:min_samples_leaf": param_30, "classifier:extra_trees:min_samples_split": param_31, "classifier:extra_trees:min_weight_fraction_leaf": param_32, "classifier:extra_trees:n_estimators": param_33, },
    {"classifier:__choice__": "gaussian_nb", },
    {"classifier:__choice__": "gradient_boosting", "LOG10_classifier:gradient_boosting:learning_rate": param_5, "classifier:gradient_boosting:loss": param_34, "classifier:gradient_boosting:max_depth": param_35, "classifier:gradient_boosting:max_features": param_36, "classifier:gradient_boosting:max_leaf_nodes": param_37, "classifier:gradient_boosting:min_samples_leaf": param_38, "classifier:gradient_boosting:min_samples_split": param_39, "classifier:gradient_boosting:min_weight_fraction_leaf": param_40, "classifier:gradient_boosting:n_estimators": param_41, "classifier:gradient_boosting:subsample": param_42, },
    {"classifier:__choice__": "k_nearest_neighbors", "LOG10_Q1_classifier:k_nearest_neighbors:n_neighbors": param_1, "classifier:k_nearest_neighbors:p": param_43, "classifier:k_nearest_neighbors:weights": param_44, },
    {"classifier:__choice__": "lda", "LOG10_classifier:lda:tol": param_6, "classifier:lda:n_components": param_45, "classifier:lda:shrinkage": param_47, },
    {"classifier:__choice__": "liblinear_svc", "LOG10_classifier:liblinear_svc:C": param_7, "LOG10_classifier:liblinear_svc:tol": param_8, "classifier:liblinear_svc:dual": param_48, "classifier:liblinear_svc:fit_intercept": param_49, "classifier:liblinear_svc:intercept_scaling": param_50, "classifier:liblinear_svc:loss": param_51, "classifier:liblinear_svc:multi_class": param_52, "classifier:liblinear_svc:penalty": param_53, },
    {"classifier:__choice__": "libsvm_svc", "LOG10_classifier:libsvm_svc:C": param_9, "LOG10_classifier:libsvm_svc:gamma": param_10, "LOG10_classifier:libsvm_svc:tol": param_11, "classifier:libsvm_svc:kernel": param_56, "classifier:libsvm_svc:max_iter": param_57, "classifier:libsvm_svc:shrinking": param_58, },
    {"classifier:__choice__": "multinomial_nb", "LOG10_classifier:multinomial_nb:alpha": param_12, "classifier:multinomial_nb:fit_prior": param_59, },
    {"classifier:__choice__": "passive_aggressive", "LOG10_Q1_classifier:passive_aggressive:n_iter": param_2, "LOG10_classifier:passive_aggressive:C": param_13, "classifier:passive_aggressive:fit_intercept": param_60, "classifier:passive_aggressive:loss": param_61, },
    {"classifier:__choice__": "proj_logit", "classifier:proj_logit:max_epochs": param_62, },
    {"classifier:__choice__": "qda", "classifier:qda:reg_param": param_63, },
    {"classifier:__choice__": "random_forest", "classifier:random_forest:bootstrap": param_64, "classifier:random_forest:criterion": param_65, "classifier:random_forest:max_depth": param_66, "classifier:random_forest:max_features": param_67, "classifier:random_forest:max_leaf_nodes": param_68, "classifier:random_forest:min_samples_leaf": param_69, "classifier:random_forest:min_samples_split": param_70, "classifier:random_forest:min_weight_fraction_leaf": param_71, "classifier:random_forest:n_estimators": param_72, },
    {"classifier:__choice__": "sgd", "LOG10_Q1_classifier:sgd:n_iter": param_3, "LOG10_classifier:sgd:alpha": param_14, "classifier:sgd:average": param_73, "classifier:sgd:eta0": param_74, "classifier:sgd:fit_intercept": param_75, "classifier:sgd:learning_rate": param_77, "classifier:sgd:loss": param_79, "classifier:sgd:penalty": param_81, },
    ])
param_83 = hp.choice("imputation:strategy", [
    {"imputation:strategy": "mean", },
    {"imputation:strategy": "median", },
    {"imputation:strategy": "most_frequent", },
    ])
param_84 = hp.uniform("LOG10_one_hot_encoding:minimum_fraction", -4.0, -0.301029995664)
param_85 = hp.choice("one_hot_encoding:use_minimum_fraction", [
    {"one_hot_encoding:use_minimum_fraction": "False", },
    {"one_hot_encoding:use_minimum_fraction": "True", "LOG10_one_hot_encoding:minimum_fraction": param_84, },
    ])
param_86 = hp.uniform("LOG10_Q1_preprocessor:kitchen_sinks:n_components", 1.69460528667, 4.00002171418)
param_87 = hp.uniform("LOG10_Q1_preprocessor:nystroem_sampler:n_components", 1.69460528667, 4.00002171418)
param_88 = hp.uniform("LOG10_preprocessor:liblinear_svc_preprocessor:C", -1.50514997832, 4.51544993496)
param_89 = hp.uniform("LOG10_preprocessor:liblinear_svc_preprocessor:tol", -5.0, -1.0)
param_90 = hp.choice("preprocessor:extra_trees_preproc_for_classification:bootstrap", [
    {"preprocessor:extra_trees_preproc_for_classification:bootstrap": "False", },
    {"preprocessor:extra_trees_preproc_for_classification:bootstrap": "True", },
    ])
param_91 = hp.choice("preprocessor:extra_trees_preproc_for_classification:criterion", [
    {"preprocessor:extra_trees_preproc_for_classification:criterion": "entropy", },
    {"preprocessor:extra_trees_preproc_for_classification:criterion": "gini", },
    ])
param_92 = hp.choice("preprocessor:extra_trees_preproc_for_classification:max_depth", [
    {"preprocessor:extra_trees_preproc_for_classification:max_depth": "None", },
    ])
param_93 = hp.uniform("preprocessor:extra_trees_preproc_for_classification:max_features", 0.5, 5.0)
param_94 = pyll.scope.int(hp.quniform("preprocessor:extra_trees_preproc_for_classification:min_samples_leaf", 0.50001, 20.5, 1.0))
param_95 = pyll.scope.int(hp.quniform("preprocessor:extra_trees_preproc_for_classification:min_samples_split", 1.50001, 20.5, 1.0))
param_96 = hp.choice("preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf", [
    {"preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf": "0.0", },
    ])
param_97 = hp.choice("preprocessor:extra_trees_preproc_for_classification:n_estimators", [
    {"preprocessor:extra_trees_preproc_for_classification:n_estimators": "100", },
    ])
param_98 = hp.choice("preprocessor:fast_ica:algorithm", [
    {"preprocessor:fast_ica:algorithm": "deflation", },
    {"preprocessor:fast_ica:algorithm": "parallel", },
    ])
param_99 = hp.choice("preprocessor:fast_ica:fun", [
    {"preprocessor:fast_ica:fun": "cube", },
    {"preprocessor:fast_ica:fun": "exp", },
    {"preprocessor:fast_ica:fun": "logcosh", },
    ])
param_100 = pyll.scope.int(hp.quniform("preprocessor:fast_ica:n_components", 9.50001, 2000.5, 1.0))
param_101 = hp.choice("preprocessor:fast_ica:whiten", [
    {"preprocessor:fast_ica:whiten": "False", },
    {"preprocessor:fast_ica:whiten": "True", "preprocessor:fast_ica:n_components": param_100, },
    ])
param_102 = hp.choice("preprocessor:feature_agglomeration:affinity", [
    {"preprocessor:feature_agglomeration:affinity": "cosine", },
    {"preprocessor:feature_agglomeration:affinity": "euclidean", },
    {"preprocessor:feature_agglomeration:affinity": "manhattan", },
    ])
param_103 = hp.choice("preprocessor:feature_agglomeration:linkage", [
    {"preprocessor:feature_agglomeration:linkage": "average", },
    {"preprocessor:feature_agglomeration:linkage": "complete", },
    {"preprocessor:feature_agglomeration:linkage": "ward", },
    ])
param_104 = pyll.scope.int(hp.quniform("preprocessor:feature_agglomeration:n_clusters", 1.50001, 400.5, 1.0))
param_105 = hp.choice("preprocessor:feature_agglomeration:pooling_func", [
    {"preprocessor:feature_agglomeration:pooling_func": "max", },
    {"preprocessor:feature_agglomeration:pooling_func": "mean", },
    {"preprocessor:feature_agglomeration:pooling_func": "median", },
    ])
param_106 = pyll.scope.int(hp.quniform("preprocessor:gem:N", 4.50001, 20.5, 1.0))
param_107 = hp.uniform("preprocessor:gem:precond", 0.0, 0.5)
param_108 = hp.uniform("LOG10_preprocessor:kernel_pca:gamma", -4.51544993496, 0.903089986992)
param_109 = hp.uniform("preprocessor:kernel_pca:coef0", -1.0, 1.0)
param_110 = pyll.scope.int(hp.quniform("preprocessor:kernel_pca:degree", 1.50001, 5.5, 1.0))
param_111 = hp.choice("preprocessor:kernel_pca:kernel", [
    {"preprocessor:kernel_pca:kernel": "cosine", },
    {"preprocessor:kernel_pca:kernel": "poly", "LOG10_preprocessor:kernel_pca:gamma": param_108, "preprocessor:kernel_pca:coef0": param_109, "preprocessor:kernel_pca:degree": param_110, },
    {"preprocessor:kernel_pca:kernel": "rbf", "LOG10_preprocessor:kernel_pca:gamma": param_108, },
    {"preprocessor:kernel_pca:kernel": "sigmoid", "preprocessor:kernel_pca:coef0": param_109, },
    ])
param_112 = pyll.scope.int(hp.quniform("preprocessor:kernel_pca:n_components", 9.50001, 2000.5, 1.0))
param_113 = hp.uniform("preprocessor:kitchen_sinks:gamma", 0.3, 2.0)
param_114 = hp.choice("preprocessor:liblinear_svc_preprocessor:dual", [
    {"preprocessor:liblinear_svc_preprocessor:dual": "False", },
    ])
param_115 = hp.choice("preprocessor:liblinear_svc_preprocessor:fit_intercept", [
    {"preprocessor:liblinear_svc_preprocessor:fit_intercept": "True", },
    ])
param_116 = hp.choice("preprocessor:liblinear_svc_preprocessor:intercept_scaling", [
    {"preprocessor:liblinear_svc_preprocessor:intercept_scaling": "1", },
    ])
param_117 = hp.choice("preprocessor:liblinear_svc_preprocessor:loss", [
    {"preprocessor:liblinear_svc_preprocessor:loss": "hinge", },
    {"preprocessor:liblinear_svc_preprocessor:loss": "squared_hinge", },
    ])
param_118 = hp.choice("preprocessor:liblinear_svc_preprocessor:multi_class", [
    {"preprocessor:liblinear_svc_preprocessor:multi_class": "ovr", },
    ])
param_119 = hp.choice("preprocessor:liblinear_svc_preprocessor:penalty", [
    {"preprocessor:liblinear_svc_preprocessor:penalty": "l1", },
    ])
param_120 = hp.uniform("LOG10_preprocessor:nystroem_sampler:gamma", -4.51544993496, 0.903089986992)
param_121 = hp.uniform("preprocessor:nystroem_sampler:coef0", -1.0, 1.0)
param_122 = pyll.scope.int(hp.quniform("preprocessor:nystroem_sampler:degree", 1.50001, 5.5, 1.0))
param_123 = hp.choice("preprocessor:nystroem_sampler:kernel", [
    {"preprocessor:nystroem_sampler:kernel": "cosine", },
    {"preprocessor:nystroem_sampler:kernel": "poly", "LOG10_preprocessor:nystroem_sampler:gamma": param_120, "preprocessor:nystroem_sampler:coef0": param_121, "preprocessor:nystroem_sampler:degree": param_122, },
    {"preprocessor:nystroem_sampler:kernel": "rbf", "LOG10_preprocessor:nystroem_sampler:gamma": param_120, },
    {"preprocessor:nystroem_sampler:kernel": "sigmoid", "LOG10_preprocessor:nystroem_sampler:gamma": param_120, "preprocessor:nystroem_sampler:coef0": param_121, },
    ])
param_124 = hp.uniform("preprocessor:pca:keep_variance", 0.5, 0.9999)
param_125 = hp.choice("preprocessor:pca:whiten", [
    {"preprocessor:pca:whiten": "False", },
    {"preprocessor:pca:whiten": "True", },
    ])
param_126 = pyll.scope.int(hp.quniform("preprocessor:polynomial:degree", 1.50001, 3.5, 1.0))
param_127 = hp.choice("preprocessor:polynomial:include_bias", [
    {"preprocessor:polynomial:include_bias": "False", },
    {"preprocessor:polynomial:include_bias": "True", },
    ])
param_128 = hp.choice("preprocessor:polynomial:interaction_only", [
    {"preprocessor:polynomial:interaction_only": "False", },
    {"preprocessor:polynomial:interaction_only": "True", },
    ])
param_129 = pyll.scope.int(hp.quniform("preprocessor:random_trees_embedding:max_depth", 1.50001, 10.5, 1.0))
param_130 = hp.choice("preprocessor:random_trees_embedding:max_leaf_nodes", [
    {"preprocessor:random_trees_embedding:max_leaf_nodes": "None", },
    ])
param_131 = pyll.scope.int(hp.quniform("preprocessor:random_trees_embedding:min_samples_leaf", 0.50001, 20.5, 1.0))
param_132 = pyll.scope.int(hp.quniform("preprocessor:random_trees_embedding:min_samples_split", 1.50001, 20.5, 1.0))
param_133 = hp.choice("preprocessor:random_trees_embedding:min_weight_fraction_leaf", [
    {"preprocessor:random_trees_embedding:min_weight_fraction_leaf": "1.0", },
    ])
param_134 = pyll.scope.int(hp.quniform("preprocessor:random_trees_embedding:n_estimators", 9.50001, 100.5, 1.0))
param_135 = hp.uniform("preprocessor:select_percentile_classification:percentile", 1.0, 99.0)
param_136 = hp.choice("preprocessor:select_percentile_classification:score_func", [
    {"preprocessor:select_percentile_classification:score_func": "chi2", },
    {"preprocessor:select_percentile_classification:score_func": "f_classif", },
    ])
param_137 = hp.uniform("preprocessor:select_rates:alpha", 0.01, 0.5)
param_138 = hp.choice("preprocessor:select_rates:mode", [
    {"preprocessor:select_rates:mode": "fdr", },
    {"preprocessor:select_rates:mode": "fpr", },
    {"preprocessor:select_rates:mode": "fwe", },
    ])
param_139 = hp.choice("preprocessor:select_rates:score_func", [
    {"preprocessor:select_rates:score_func": "chi2", },
    {"preprocessor:select_rates:score_func": "f_classif", },
    ])
param_140 = hp.choice("preprocessor:__choice__", [
    {"preprocessor:__choice__": "extra_trees_preproc_for_classification", "preprocessor:extra_trees_preproc_for_classification:bootstrap": param_90, "preprocessor:extra_trees_preproc_for_classification:criterion": param_91, "preprocessor:extra_trees_preproc_for_classification:max_depth": param_92, "preprocessor:extra_trees_preproc_for_classification:max_features": param_93, "preprocessor:extra_trees_preproc_for_classification:min_samples_leaf": param_94, "preprocessor:extra_trees_preproc_for_classification:min_samples_split": param_95, "preprocessor:extra_trees_preproc_for_classification:min_weight_fraction_leaf": param_96, "preprocessor:extra_trees_preproc_for_classification:n_estimators": param_97, },
    {"preprocessor:__choice__": "fast_ica", "preprocessor:fast_ica:algorithm": param_98, "preprocessor:fast_ica:fun": param_99, "preprocessor:fast_ica:whiten": param_101, },
    {"preprocessor:__choice__": "feature_agglomeration", "preprocessor:feature_agglomeration:affinity": param_102, "preprocessor:feature_agglomeration:linkage": param_103, "preprocessor:feature_agglomeration:n_clusters": param_104, "preprocessor:feature_agglomeration:pooling_func": param_105, },
    {"preprocessor:__choice__": "gem", "preprocessor:gem:N": param_106, "preprocessor:gem:precond": param_107, },
    {"preprocessor:__choice__": "kernel_pca", "preprocessor:kernel_pca:kernel": param_111, "preprocessor:kernel_pca:n_components": param_112, },
    {"preprocessor:__choice__": "kitchen_sinks", "LOG10_Q1_preprocessor:kitchen_sinks:n_components": param_86, "preprocessor:kitchen_sinks:gamma": param_113, },
    {"preprocessor:__choice__": "liblinear_svc_preprocessor", "LOG10_preprocessor:liblinear_svc_preprocessor:C": param_88, "LOG10_preprocessor:liblinear_svc_preprocessor:tol": param_89, "preprocessor:liblinear_svc_preprocessor:dual": param_114, "preprocessor:liblinear_svc_preprocessor:fit_intercept": param_115, "preprocessor:liblinear_svc_preprocessor:intercept_scaling": param_116, "preprocessor:liblinear_svc_preprocessor:loss": param_117, "preprocessor:liblinear_svc_preprocessor:multi_class": param_118, "preprocessor:liblinear_svc_preprocessor:penalty": param_119, },
    {"preprocessor:__choice__": "no_preprocessing", },
    {"preprocessor:__choice__": "nystroem_sampler", "LOG10_Q1_preprocessor:nystroem_sampler:n_components": param_87, "preprocessor:nystroem_sampler:kernel": param_123, },
    {"preprocessor:__choice__": "pca", "preprocessor:pca:keep_variance": param_124, "preprocessor:pca:whiten": param_125, },
    {"preprocessor:__choice__": "polynomial", "preprocessor:polynomial:degree": param_126, "preprocessor:polynomial:include_bias": param_127, "preprocessor:polynomial:interaction_only": param_128, },
    {"preprocessor:__choice__": "random_trees_embedding", "preprocessor:random_trees_embedding:max_depth": param_129, "preprocessor:random_trees_embedding:max_leaf_nodes": param_130, "preprocessor:random_trees_embedding:min_samples_leaf": param_131, "preprocessor:random_trees_embedding:min_samples_split": param_132, "preprocessor:random_trees_embedding:min_weight_fraction_leaf": param_133, "preprocessor:random_trees_embedding:n_estimators": param_134, },
    {"preprocessor:__choice__": "select_percentile_classification", "preprocessor:select_percentile_classification:percentile": param_135, "preprocessor:select_percentile_classification:score_func": param_136, },
    {"preprocessor:__choice__": "select_rates", "preprocessor:select_rates:alpha": param_137, "preprocessor:select_rates:mode": param_138, "preprocessor:select_rates:score_func": param_139, },
    ])
param_141 = hp.choice("rescaling:__choice__", [
    {"rescaling:__choice__": "min/max", },
    {"rescaling:__choice__": "none", },
    {"rescaling:__choice__": "normalize", },
    {"rescaling:__choice__": "standardize", },
    ])

space = {"balancing:strategy": param_0, "classifier:__choice__": param_82, "imputation:strategy": param_83, "one_hot_encoding:use_minimum_fraction": param_85, "preprocessor:__choice__": param_140, "rescaling:__choice__": param_141}
