import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tf.keras.layers

class GaussianMixture(tf.Module):
    def __init__(self, dimensions, n_components, name = None):
        super().__init__(name = name)
        assert isinstance(n_components, int), "n_components must be integer"
        assert isinstance(dimensions, int), "dimensions must be integer"
    
        
        self.dimensions = dimensions
        self.n_components = n_components
        
        #logits for categorical distribution
        self.categorical_logits = tf.Variable(tf.random.normal(shape = [n_components]), name = "categorical_logits")  
        
        #means for the gaussian distributions
        self.means = [tf.Variable(tf.random.normal(shape = [dimensions]), name = "mean") for i in range(n_components)]
        
        #transformed, exponentiated variables for variances
        self.variances = [tfp.util.TransformedVariable(tf.exp(tf.random.normal(shape = [dimensions]) + 1), bijector = tfb.Exp(), name = "variance") for i in range(n_components)]
        
        #gaussian components are created
        self.components = [tfd.MultivariateNormalDiag(loc = self.means[i], scale_diag = self.variances[i]) for i in range(n_components)]
        
        #mixture dist is created
        self.dist = tfd.Mixture(cat = tfd.Categorical(logits = self.categorical_logits), components = self.components)
        
    @tf.function()
    def log_prob(self, val):
        return self.dist.log_prob(val)
    
    @tf.function()
    def sample(self, sample_shape):
        return self.dist.sample(sample_shape)
    
    @tf.function()
    def calc_mixture_params(self):
        cat_prob = tf.nn.softmax(self.categorical_logits(data))
        means = [self.means[i] for i in range(self.n_components)]
        variances = [self.variances[i] for i in range(self.n_components)]
        return (cat_prob, means, variances)