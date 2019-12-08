from __future__ import absolute_import
import fastlmm.association as association
import scipy as sp
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
#!! from fastlmm.external.sklearn.metrics.scorer import SCORERS, Scorer
from fastlmm import inference
import fastlmm.util.util as util
import pdb
from six.moves import range



class testCV(association.varcomp_test):
    __slots__ = ["G0","greater_is_better","model","scoring","n_folds",
                 "n_folds_params","verbose","n_jobs_grid",
                 "data_permutation","scores_null","params_null","nullModel",
                 "altModel"]
    
    def __init__(self,Y,X=None, G0=None, appendbias=False, model = None,
                 n_folds_params=10, n_folds=10, scoring = None, verbose = False,
                 n_jobs_grid=1, data_permutation=None, nested=True, greater_is_better=None,
                 nullModel=None, altModel=None):#Note that this code will break as we don't know what it does
        association.varcomp_test.__init__(self,Y=Y,X=X,appendbias=appendbias)

        assert model is None, "Shouldn't we remove this parameter?"
        self.n_jobs_grid = n_jobs_grid
        self.verbose = verbose
        self.n_folds = n_folds
        self.n_folds_params = n_folds_params
        self.G0=G0
        self.nullModel = nullModel
        self.altModel = altModel

        if data_permutation is None:
            data_permutation = util.generate_permutation(self.Y.shape[0],93828231) #permute with an arbitrary seed
        self.data_permutation = data_permutation

        if 'param_grid' in nullModel:
            param_grid = nullModel['param_grid']
        else:
            param_grid = self._getParamGrid(G0, None, nullModel)

        if scoring is None:
            (self.scoring, self.greater_is_better) = self._getScoring()
        else:
            self.scoring = scoring
            self.greater_is_better = greater_is_better
  
        model = self._getModel(nullModel, param_grid)
        nested = self._isNested(nullModel)

        self.scores_null, self.params_null = self.score_nestedCV(None, model, param_grid,
                                     self.nullModel['effect'], nested)

    def _getScoring(self):
        if self.nullModel['link'] == 'linear':
            scoring = 'mse'
            greater_is_better=False
        elif self.nullModel['link'] == 'logistic':
            scoring = 'binomial'
            greater_is_better=True
        else:
            assert False, 'Unknown link function.'

        return (scoring, greater_is_better)

    def _getModel(self, modelDesc,  param_grid):
        if modelDesc['effect']=='fixed':
            return self._getFixedEffectModel(modelDesc['penalty'], modelDesc['link'],
                                             param_grid)
        elif modelDesc['effect']=='mixed':
            return self._getMixedEffectModel(modelDesc['link'], modelDesc['approx'],
                                             param_grid)
        else:
            assert False

    def _getFixedEffectModel(self, penalty, link, param_grid):
        assert penalty in set(['l1','l2'])

        if link == 'linear':
            if penalty == 'l2':
                model = linear_model.Ridge(alpha=param_grid['alpha'][0], fit_intercept=True,
                                                solver = 'auto')
            elif penalty == 'l1':
                model = linear_model.LassoCV(alphas=param_grid['alpha'] ,cv=self.n_folds_params,
                                                      precompute='auto',fit_intercept=True)

        elif link == 'logistic':
            model = linear_model.LogisticRegression(penalty=penalty, dual=False,
                                           tol=0.0001, C=param_grid['C'][0], fit_intercept=True,
                                           intercept_scaling=1, class_weight=None, random_state=None)
        else:
            assert False, 'Unknown link function.'

        return model

    def _getMixedEffectModel(self, link, approx, param_grid):
        C = inference.makeBin2KernelAsEstimator(link, approx)
        return C()

    def _isNested(self, modelDesc):
        if modelDesc['effect'] == 'fixed' and modelDesc['link'] == 'linear'\
           and modelDesc['penalty'] == 'l1':
            return False
        return True

    def _getParamGrid(self, G0, G1, modelDesc):
        if modelDesc['effect'] == 'fixed':
            return self._getParamGridFixedEffectModel(G0, G1, modelDesc['link'])
        elif modelDesc['effect'] == 'mixed':
            return self._getParamGridMixedEffectModel(G0, G1)

        assert False

    def _getParamGridFixedEffectModel(self, G0, G1, link):
        if link == 'linear':
            param_grid = dict(alpha=0.5*sp.logspace(-5, 5, 20))
        elif link == 'logistic':
            param_grid = dict(C=sp.logspace(-5, 5, 20))
        else:
            assert False

        return param_grid

    def _getParamGridMixedEffectModel(self, G0, G1):
        param_grid = dict(sig02=sp.arange(0.0,2.1,0.4),
                          sig12=sp.arange(0.0,2.1,0.4),
                          sign2=[None],
                          beta=[None])

        if G0 is None:
            param_grid['sig02'] = [0.0]
        if G1 is None:
            param_grid['sig12'] = [0.0]

        return param_grid

    def testG(self, G1, type='',i_exclude=None):

        pv=1.0
        stat=1.0

        if 'param_grid' in self.altModel:
            param_grid = self.altModel['param_grid']
        else:
            param_grid = self._getParamGrid(self.G0, G1, self.altModel)

        model = self._getModel(self.altModel, param_grid)
        nested = self._isNested(self.altModel)

        scores,params = self.score_nestedCV(G1, model, param_grid, self.altModel['effect'], nested)

        if self.greater_is_better:
            stat = 2.0*(scores - self.scores_null).mean()
        else:
            stat = 2.0*(self.scores_null-scores).mean()

        test={
              'pv':pv,
              'stat':stat,
              'scores':scores,
              'scores0':self.scores_null,
              'params':params,
              'params0':self.params_null,
              'type':type # is it OK to have an object here instead of a name?
              }
        return test

    # the effect parameter should not be used here, but I dont have a better an idea for now
    def score_nestedCV(self, G1, model, param_grid, effect, nested):
        k_fold = model_selection.KFold(n_splits=self.n_folds).split(list(range(self.Y.shape[0])))
        i_fold=0
        scores = sp.zeros(self.n_folds)
        params = list()

        for train, test in k_fold:
            (trainData, trainY) = self._packData(G1, train, effect)
            (testData, testY) = self._packData(G1, test, effect)

            if nested:
                clf = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs = self.n_jobs_grid,
                                   cv=self.n_folds_params, scoring=self.scoring, verbose=self.verbose)

                clf.fit(trainData, trainY.flatten())

                params.append(clf.best_params_)

                scores[i_fold] = clf.score(testData, testY.flatten(), method_scorer=False)
            else:

                model.fit(trainData, trainY.flatten())
                scores[i_fold] = SCORERS[self.scoring](model, testData, testY.flatten())
            i_fold+=1

        return scores,params
    
    def _packData(self, G1, indices2select, effect):
        if effect == 'fixed':
            if G1 is None and self.G0 is None:
                data = self.X[self.data_permutation][indices2select]
            elif G1 is None:
                data = sp.column_stack((self.G0[self.data_permutation][indices2select],
                                        self.X[self.data_permutation][indices2select]))
            elif self.G0 is None:
                data = sp.column_stack((G1[self.data_permutation][indices2select],
                                        self.X[self.data_permutation][indices2select]))
            else:
                data = sp.column_stack((self.G0[self.data_permutation][indices2select],
                                        G1[self.data_permutation][indices2select],
                                        self.X[self.data_permutation][indices2select]))
        elif effect == 'mixed':
            X = self.X[self.data_permutation]
            if self.G0 is not None:
                G0 = self.G0[self.data_permutation]
            if G1 is not None:
                G1 = G1[self.data_permutation]

            data = []
            for i in range(len(indices2select)):

                lis = [X[indices2select[i]]]
                if G0 is not None:
                    lis.append( G0[indices2select[i]] )
                if G1 is not None:
                    lis.append( G1[indices2select[i]] )

                data.append( lis )
            else:
                assert False, 'Unkown effect type.'

        return (data, self.Y[self.data_permutation][indices2select])
