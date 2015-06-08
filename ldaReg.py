#dpylint: disable-msg=C0103
import numpy as np


def ldaRegWeights(X, y, reg=[0.05]):

    """
    Fit the Regularized LDA model according to the given training data and
    parameters.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y : array, shape = [n_samples]
        Target values (integers)

    reg : array of float
        Amount of regularization to use.
    """

    n, m = X.shape
    classLabel = np.unique(y)
    k = len(classLabel)
    nReg = len(reg)

    nGroup = np.zeros(k)
    groupMean = np.zeros((m, k))
    pooledCov = np.zeros((m, m))

    #Perform intermediate calculations
    for i in np.arange(k):
        # Establish location and size of each class
        group = (y == classLabel[i])

        nGroup[i] = np.sum(group)

        temp = X[group, :]

        #Calculate group mean vectors
        groupMean[:, i] = np.mean(temp, axis=0)

        #Accumulate pooled cov information
        pooledCov = pooledCov + ((nGroup[i] - 1) / (n - k)) * \
            np.cov(temp, rowvar=0)

    # Calculate prior probs
    priorProb = nGroup / n

    # Loop over classes to calculate LDA coefs
    W = np.zeros((pooledCov.shape[0] + 1, len(classLabel), nReg))
    for ri in np.arange(nReg):
        pooledCov2 = pooledCov * (1 - reg[ri]) + (reg[ri] / m) * \
            np.trace(pooledCov) * np.eye(m)
        temp = np.linalg.solve(pooledCov2, groupMean)

        wtsPt1 = np.diag(-0.5 * np.dot(groupMean.T, temp)
                         + np.log(priorProb)).reshape(1, -1)
        W[:, :, ri] = np.concatenate((wtsPt1, temp))

    return W
