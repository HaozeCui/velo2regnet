import numpy as np
from sklearn.linear_model import lars_path

def lambdaLasso(subExpMat, subVelMat, larray, m):
    arrayIndex = np.zeros((len(larray), m, m))
    for j in range(m):
        _, _, coefsAll = lars_path(subExpMat, subVelMat[:,j], method='lasso', verbose=True)
        for i in range(len(larray)):
            coefs = coefsAll[:,0:larray[i] + 10]
            arrayIndex[i,j] = np.sum(coefs != 0,1) * (coefs[:,-1] != 0)
    return arrayIndex


def TIGRESSLasso(expressionMat, velocity, m, index, larray, alpha):
    subExpMat = expressionMat[index]
    subVelMat = velocity[index]
    beta = np.random.rand(m) * (1 - alpha) + alpha
    subExpMat = beta * subExpMat
    rankAppearance = np.zeros((len(larray), m, m))
    lassoResult = lambdaLasso(subExpMat, subVelMat, larray, m)
    for i in range(len(larray)):
        appRank = m - np.argsort(np.argsort(lassoResult[i], 1), 1)
        appRank[appRank > larray[i]] = m ** 2
        rankAppearance[i] = appRank
    return rankAppearance
