import numpy as np
from scipy.stats import chi2
from scipy.special import gammaln


def fminESS(p, alpha=.05, eps=.05, ess=None):
    """
    Minimum effective sample size
    """

    crit = chi2.ppf(1 - alpha, p)
    foo = 2. / p

    if ess is None:
        logminESS = foo * np.log(2.) + np.log(np.pi) - foo * np.log(p) -\
            foo * gammaln(p / 2.) - 2. * np.log(eps) + np.log(crit)
        return np.round(np.exp(logminESS))
    else:
        if isinstance(ess, str):
            raise ValueError("Only numeric entry allowed for ess")
        logEPS = .5 * foo * np.log(2.) + .5 * np.log(np.pi) -\
            .5 * foo * np.log(p) - .5 * foo * gammaln(p / 2.) -\
            .5 * np.log(ess) + .5 * np.log(crit)
        return np.exp(logEPS)


def multiESS(X, b='sqroot', Noffsets=10, Nb=None):
    """
    Compute multivariate effective sample size of a single Markov chain X,
    using the multivariate dependence structure of the process.
    X: MCMC samples of shape (n, p)
    n: number of samples
    p: number of parameters
    b: specifies the batch size for estimation of the covariance matrix in
       Markov chain CLT. It can take a numeric value between 1 and n/2, or a
       char value between:
    'sqroot'    b=floor(n^(1/2)) (for chains with slow mixing time; default)
    'cuberoot'  b=floor(n^(1/3)) (for chains with fast mixing time)
    'lESS'      pick the b that produces the lowest effective sample size
                for a number of b ranging from n^(1/4) to n/max(20,p); this
                is a conservative choice
    If n is not divisible by b Sigma is recomputed for up to Noffsets subsets
    of the data with different offsets, and the output mESS is the average over
    the effective sample sizes obtained for different offsets.
    Nb specifies the number of values of b to test when b='less'
    (default NB=200). This option is unused for other choices of b.
    Original source: https://github.com/lacerbi/multiESS
    Reference:
    Vats, D., Flegal, J. M., & Jones, G. L. "Multivariate Output Analysis
    for Markov chain Monte Carlo", arXiv preprint arXiv:1512.07713 (2015).
    """

    # MCMC samples and parameters
    n, p = X.shape

    if p > n:
        raise ValueError(
            "More dimensions than data points, cannot compute effective "
            "sample size.")

    # Input check for batch size B
    if isinstance(b, str):
        if b not in ['sqroot', 'cuberoot', 'less']:
            raise ValueError(
                "Unknown string for batch size. Allowed arguments are "
                "'sqroot', 'cuberoot' and 'lESS'.")
        if b != 'less' and Nb is not None:
            raise Warning(
                "Nonempty parameter NB will be ignored (NB is used "
                "only with 'lESS' batch size B).")
    else:
        if not 1. < b < (n / 2):
            raise ValueError(
                "The batch size B needs to be between 1 and N/2.")

    # Compute multiESS for the chain
    mESS = multiESS_chain(X, n, p, b, Noffsets, Nb)

    return mESS


def multiESS_chain(Xi, n, p, b, Noffsets, Nb):
    """
    Compute multiESS for a MCMC chain.
    """

    if b == 'sqroot':
        b = [int(np.floor(n ** (1. / 2)))]
    elif b == 'cuberoot':
        b = [int(np.floor(n ** (1. / 3)))]
    elif b == 'less':
        b_min = np.floor(n ** (1. / 4))
        b_max = max(np.floor(n / max(p, 20)), np.floor(np.sqrt(n)))
        if Nb is None:
            Nb = 200
        # Try NB log-spaced values of B from B_MIN to B_MAX
        b = set(map(int, np.round(np.exp(
            np.linspace(np.log(b_min), np.log(b_max), Nb)))))

    # Sample mean
    theta = np.mean(Xi, axis=0)
    # Determinant of sample covariance matrix
    if p == 1:
        detLambda = np.cov(Xi.T)
    else:
        detLambda = np.linalg.det(np.cov(Xi.T))

    # Compute mESS
    mESS_i = []
    for bi in b:
        mESS_i.append(multiESS_batch(Xi, n, p, theta, detLambda, bi, Noffsets))
    # Return lowest mESS
    mESS = np.min(mESS_i)

    return mESS


def multiESS_batch(Xi, n, p, theta, detLambda, b, Noffsets):
    """
    Compute multiESS for a given batch size B.
    """

    # Compute batch estimator for SIGMA
    a = int(np.floor(n / b))
    Sigma = np.zeros((p, p))
    offsets = np.sort(list(set(map(int, np.round(
        np.linspace(0, n - np.dot(a, b), Noffsets))))))

    for j in offsets:
        # Swapped a, b in reshape compared to the original code.
        Y = Xi[j + np.arange(a * b), :].reshape((a, b, p))
        Ybar = np.squeeze(np.mean(Y, axis=1))
        Z = Ybar - theta
        for i in range(a):
            if p == 1:
                Sigma += Z[i] ** 2
            else:
                Sigma += Z[i][np.newaxis, :].T * Z[i]

    Sigma = (Sigma * b) / (a - 1) / len(offsets)
    mESS = n * (detLambda / np.linalg.det(Sigma)) ** (1. / p)

    return mESS

def main():
    """
    Test against the original code. Requires Octave and oct2py installed, and
    the original 'multiESS.m' file in the dir.
    """

    nchains = np.random.randint(2, 10)
    print("Chains: ", nchains)
    for k in range(nchains):

        nruns = np.random.randint(100, 20000)
        nparams = np.random.randint(1, 2)

        samples_chains = np.random.uniform(0., 1., (nchains, nruns, nparams))
        chain = samples_chains[k]

        print("N, p: ", nruns, nparams)
        mESS_python = multiESS(chain, b="sqroot")

        print("mESS_py: {:.2f}".format(mESS_python))


if __name__ == '__main__':
    main()