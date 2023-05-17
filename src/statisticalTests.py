import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

class dickyFuller:
    regStr = "regression"
    adfStr = "adf"
    pvalStr = "pvalue"
    usedLagStr = "usedlag"
    nobsStr = "nobs"
    percent1Str = "critical values: 1%"
    percent5Str = "critical values: 5%"
    percent10Str = "critical values: 10%"
    icBestStr = "icbest"

    def __init__(self):
        self.regressions = ["c", "ct", "ctt"] # c = constant; ct = constant and trend; ctt = constant and trend and quadratic trend
        self.autolag = "AIC"

    def createDict(self):
        return {
            self.regStr: [],
            self.adfStr: [],
            self.pvalStr: [],
            self.usedLagStr: [],
            self.nobsStr: [],
            self.percent1Str: [],
            self.percent5Str: [],
            self.percent10Str: [],
            self.icBestStr: [],
        }

    def getDickyFullerScores(self, data: pd.DataFrame):
        dictDickyFuller = self.createDict()
        for reg in self.regressions:
            result = adfuller(data, regression=reg, autolag=self.autolag) # Augmented dicky-fuller unit root test 

            dictDickyFuller[self.regStr].append(reg)
            dictDickyFuller[self.adfStr].append(result[0])
            dictDickyFuller[self.pvalStr].append(result[1])
            dictDickyFuller[self.usedLagStr].append(result[2])
            dictDickyFuller[self.nobsStr].append(result[3])
            dictDickyFuller[self.percent1Str].append(result[4]["1%"])
            dictDickyFuller[self.percent5Str].append(result[4]["5%"])
            dictDickyFuller[self.percent10Str].append(result[4]["10%"])
            dictDickyFuller[self.icBestStr].append(result[5])
            
        return pd.DataFrame(dictDickyFuller)


class modelEvaluation:
    def __init__(self, nObservations):
        self.nObservations = nObservations

    def getBestModel(self, timeSerie,pMax=5, dMax=1, qMax=2, trend="n"):        
        """ Find the best ARIMA model from AIC
        Args:
            timeSerie (pd.DataFrame): input data
            pMax (int): max order of autoregressive model
            dMax (int): max order of derivatives
            qMax (int): max order of moving average model
            trend (str) : n, c, t, ct
        """
        
        bestAIC = np.inf 
        bestOrder = None
        bestModel = None

        for p in range(pMax):
            for d in range(dMax):
                for q in range(qMax):
                    temporaryModel = ARIMA(endog=timeSerie, order=(p,d,q), trend=trend).fit()
                    temporaryAIC = temporaryModel.aic
                    if temporaryAIC < bestAIC:
                        bestAIC = temporaryAIC
                        bestOrder = (p, d, q)
                        bestModel = temporaryModel
        print('AIC: {:6.5f} | order: {}'.format(bestAIC, bestOrder))
        return bestAIC, bestOrder, bestModel

