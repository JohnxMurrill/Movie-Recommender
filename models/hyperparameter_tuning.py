from bayesian_personalized_rankings.bpr import BPR
from logistic_matrix_factorization.lmf import LMF
from als.als import AlternatingLeastSquares

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11)}

samples = 8  # number of random samples 
randomCV = RandomizedSearchCV(BPR, param_distributions=param_dist, n_iter=samples,cv=3)