import lasso
import scoreMatrix
import distance

import math
import numpy as np
from numpy import random as rd


HALFRUN = 10
ALPHA = 0.4
larray = [30, 50, 70, 90, 100]
L = len(larray)
dir = "./"
pseudoTimeVec = np.loadtxt(dir + "pseudotime.txt")
expressionMat = np.transpose(np.loadtxt(dir + "datamatrix.txt"))

order = np.argsort(pseudoTimeVec)
pseudoTimeVec = pseudoTimeVec[order]
expressionMat = expressionMat[order]

if len(pseudoTimeVec) != len(expressionMat):
    assert("NOT proper dimensions!")
    exit(0)

cellNum = np.size(expressionMat, 0)
geneNum = np.size(expressionMat, 1)

rank = distance.distanceToRank(distance.KLdistance(expressionMat))

velocity = np.loadtxt(dir + "velocity.txt")
arrayMat = np.zeros((HALFRUN*2, L, geneNum, geneNum))

expressionMat = (expressionMat - np.min(expressionMat, 0)) / (np.max(expressionMat, 0) - np.min(expressionMat, 0))

for i in range(HALFRUN):
    flag = math.floor(cellNum/2)
    index = rd.permutation(range(cellNum))
    index1 = index[:flag]
    index2 = index[flag:]
    arrayMat[i] = lasso.TIGRESSLasso(expressionMat, velocity, geneNum, index1, larray, ALPHA)
    arrayMat[HALFRUN + i] = lasso.TIGRESSLasso(expressionMat, velocity, geneNum, index2, larray, ALPHA)

realNetwork = np.loadtxt(dir + "A.txt")

appArrayRankMat, ROCScoreAreaMat, PRScoreAreaMat, tprArr, fprArr, ppvArr = scoreMatrix.scoreMat(arrayMat, realNetwork, larray, geneNum, HALFRUN * 2, rank)
print(ROCScoreAreaMat)
print(PRScoreAreaMat)
