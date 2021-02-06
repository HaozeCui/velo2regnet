import numpy as np

def compROC(appRank, rmDiagRed, listnnz):
    P = np.count_nonzero(rmDiagRed)
    N = np.size(rmDiagRed) - len(rmDiagRed) - P
    appWoDiag = appRank ^ np.diag(np.diag(appRank))
    appWoDiagRed = appWoDiag[listnnz][:,listnnz]
    TN = np.count_nonzero((appWoDiagRed == 0) * (rmDiagRed == 0)) - len(listnnz)
    TP = np.count_nonzero((appWoDiagRed != 0) * (rmDiagRed != 0))
    if N == 0:
        fpr = 1
    else:
        fpr = 1 - TN/N
    if P == 0:
        tpr = 0
    else:
        tpr = TP/P
    if TP + N - TN == 0:
        ppv = 1
    else:
        ppv = TP/(TP + N - TN)    
    return tpr, fpr, ppv

def rankMatrices(appIndex, realNetwork, l, m, runNum, rank2):
    scoreMatArea = np.sum(1 / appIndex, axis = 0) / runNum / (l + 1)
    rank = np.reshape(m * m - np.argsort(np.argsort(np.reshape(scoreMatArea,(1,m*m)), 1), 1), (m,m))
    TPRArea = np.zeros(m * m + 1)
    FPRArea = np.zeros(m * m + 1)
    PPVArea = np.zeros(m * m + 1)
    rmDiag = realNetwork - np.diag(np.diag(realNetwork))
    listnnz = np.union1d(np.where(np.sum(rmDiag, 0) > 0), np.where(np.sum(rmDiag, 1) > 0))
    rmDiagRed = rmDiag[listnnz][:,listnnz]
    for i in range(m * m + 1):
        appRank = np.logical_and(rank < i + 1, rank2 < i + 1)
        TPRArea[i], FPRArea[i], PPVArea[i] = compROC(appRank, rmDiagRed, listnnz)

    AUROC = np.trapz(TPRArea, FPRArea)
    AUPR = np.trapz(PPVArea, TPRArea)
    return rank, AUROC, AUPR, TPRArea, FPRArea, PPVArea

def scoreMat(arrayMat, realNetwork, LArray, m, runNum, rank):
    ROCScoreAreaMat = np.zeros(len(LArray))
    PRScoreAreaMat = np.zeros(len(LArray))
    tprArr = np.zeros((len(LArray), m * m + 1))
    fprArr = np.zeros((len(LArray), m * m + 1))
    ppvArr = np.zeros((len(LArray), m * m + 1))
    appArrayRankMat= np.zeros((len(LArray), m, m))
    for i in range(len(LArray)):
        appArrayRankMat[i], ROCScoreAreaMat[i], PRScoreAreaMat[i], tprArr[i], fprArr[i], ppvArr[i] = rankMatrices(arrayMat[:,i], realNetwork, LArray[i], m, runNum, rank)

    return appArrayRankMat, ROCScoreAreaMat, PRScoreAreaMat, tprArr, fprArr, ppvArr
