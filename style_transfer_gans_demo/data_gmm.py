"""
Created on Sun Mar 19 13:22:00 2017
@author: cli
"""
import functools
import matplotlib.cm as cm
import matplotlib.pyplot as pyplot
import numpy
import scipy

class Distribution:
    """
    Gaussian Mixture Distribution

    Parameters
    ----------
    means: tuple of ndarray.
       Specifies the means for the gaussian components.
    variances: tuple of ndarray.
       Specifies the variances for the gaussian components.
    priors: tuple of ndarray
       Specifies the prior distribution of the components.
    """
    def __init__(self, means=None, variances=None, priors=None, rng=None, seed=None):
        if means is None:
            means = map(
                lambda x: 10.0 * numpy.array(x),
                [[0, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]])
        self.ncomponents = len(means)
        self.dim = means[0].shape[0]
        self.means = means
        if priors is None:
            priors = [1.0/self.ncomponents for _ in range(self.ncomponents)]
        self.priors = priors
        if variances is None:
            variances = [numpy.eye(self.dim) for _ in range(self.ncomponents)]
        self.variances = variances
        assert len(means) == len(variances), "Mean variances mismatch"
        assert len(variances) == len(priors), "prior mismatch"
        if rng is None:
            rng = numpy.random.RandomState(seed=seed)
        self.rng = rng

    def _gaussian_pdf(self, x, mean, variance):
        """
        DOCSTRING
        """
        return scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=variance)

    def _sample_gaussian(self, mean, variance):
        """
        sampling unit gaussians
        """
        epsilons = self.rng.normal(size=(self.dim, ))
        return mean + numpy.linalg.cholesky(variance).dot(epsilons)
   
    def _sample_prior(self, nsamples):
        """
        DOCSTRING
        """
        return self.rng.choice(
            a=self.ncomponents, size=(nsamples, ), replace=True, p=self.priors)
   
    def pdf(self, x):
        """
        Evaluates the the probability density function at the given point x
        """
        pdfs = map(
            lambda m, v, p: p*self._gaussian_pdf(x, m, v),
            self.means, self.variances, self.priors)
        return functools.reduce(lambda x, y: x + y, pdfs, 0.0)
    
    def sample(self, nsamples):
        """
        Sampling priors
        """
        samples = []
        fathers = self._sample_prior(nsamples=nsamples).tolist()
        for father in fathers:
            samples.append(self._sample_gaussian(
                self.means[father], self.variances[father]))
        return numpy.array(samples), numpy.array(fathers)

class PlotGMM:
    """
    DOCSTRING
    """
    def plot(self, dataset, save_path):
        """
        DOCSTRING
        """
        figure, axes = pyplot.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
        ax = axes
        ax.set_aspect('equal')
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
        ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.axis('on')
        ax.set_title('$\mathbf{x} \sim $GMM$(\mathbf{x})$')
        x = dataset.data['samples']
        targets = dataset.data['label']
        axes.scatter(
            x[:, 0], x[:, 1], marker='.', alpha=0.3,
            c=cm.Set1(targets.astype(float)/2.0/2.0))
        pyplot.tight_layout()
        pyplot.savefig(save_path, transparent=True, bbox_inches='tight')

class SampleGMM:
    """
    Toy dataset containing points sampled from a gaussian mixture distribution.

    The dataset contains 3 sources:
        * samples
        * label
        * densities
    """
    def __init__(self, num_examples, means=None, variances=None, priors=None, **kwargs):
        rng = kwargs.pop('rng', None)
        if rng is None:
            seed = kwargs.pop('seed', 0)
            rng = numpy.random.RandomState(seed)
        gaussian_mixture = Distribution(
            means=means, variances=variances, priors=priors, rng=rng)
        self.means = gaussian_mixture.means
        self.variances = gaussian_mixture.variances
        self.priors = gaussian_mixture.priors
        features, labels = gaussian_mixture.sample(nsamples=num_examples)
        densities = gaussian_mixture.pdf(x=features)
        data ={'samples': features, 'label': labels, 'density': densities}
        self.data = data

if __name__ == '__main__':
    means = map(lambda x: numpy.array(x), [[0, 0], [2, 2], [-1, -1], [1, -1], [-1, 1]])
    std = 0.1
    variances = [numpy.eye(2) * std for _ in means]
    priors = [1.0/len(means) for _ in means]
    gaussian_mixture = GaussianMixtureDistribution(
        means=means, variances=variances, priors=priors)
    dataset = GaussianMixture(
        1000, means, variances, priors, sources=('features', ))
    save_path = './gmm_data.pdf'
    draw_GMM(dataset, save_path)
