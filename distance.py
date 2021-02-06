import numpy as np

# calculate the KL distance of the row / column vectors
# matrix is a rowNum \times colNum matrix
# isColVec = 0 : row vector
# isColVec = 1 : column vector, 
def KLdistance(matrix, isColVec = True):
    rowNum = np.size(matrix, 0)
    colNum = np.size(matrix, 1)
    if isColVec:
        distance = np.zeros((colNum, colNum))
        tempMat = matrix + 1.0 / rowNum
        tempMat = tempMat / np.sum(tempMat, 0)
        for i in range(colNum):
            for j in range(colNum):
                distance[i, j] = np.sum(tempMat[:, i] * np.log(tempMat[:, i] / tempMat[:, j]))
        return distance
    else:
        distance = np.zeros((rowNum, rowNum))
        tempMat = matrix + 1.0 / colNum
        tempMat = tempMat / np.resize(np.sum(tempMat, 1),(rowNum, 1))
        for i, j in range(rowNum):
            distance[i, j] = np.sum(tempMat[i, :] * np.log(tempMat[i, :] / tempMat[j, :]))
        return distance        

# get rank according to distance
# element of diag always have largest rank
def distanceToRank(matrix):
    m = np.size(matrix, 0)
    negDisMat = np.max(matrix) + 1 - matrix
    negDisMat = negDisMat - np.diag(np.diag(negDisMat))
    return np.reshape(m * m - np.argsort(np.argsort(np.reshape(negDisMat,(1,m*m)), 1), 1), (m,m))
