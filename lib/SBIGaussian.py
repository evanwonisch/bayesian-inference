import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfl = tf.keras.layers

class DistParametriser(tf.Module):
    def __init__(self, input_dim, output_shape, layer_sizes = [1], activation = "relu", name = None):
        super().__init__(name = name)
        assert isinstance(input_dim, int), "input_dim must be integer"
        assert isinstance(output_shape, list), "input_dim must be integer"
        
        output_dim = tf.reduce_prod(output_shape).numpy()
        
        self.layers = []
        
        #backwards
        self.layers.append(tfl.Reshape(output_shape))
        self.layers.append(tfl.Dense(output_dim, activation = activation))
        
        for i in range(len(layer_sizes)):
            layer = tfl.Dense(layer_sizes[-i - 1], activation = "relu")
            self.layers.append(layer)
            
        self.layers[-1].build(input_shape = (input_dim))
        
        #build
        x = tf.expand_dims(tf.random.normal(shape = [input_dim]), axis = 0)
        for i in reversed(range(len(self.layers))):
            x = self.layers[i](x)
        
    @tf.function()
    def __call__(self, x):
        for i in reversed(range(len(self.layers))):
            x = self.layers[i](x)
        return x

class SBIGaussianMixture(tf.Module):
    def __init__(self, data_dim, param_dim, n_components = 1, layer_sizes = [5,5], name = None):
        super().__init__(name = name)
        assert isinstance(n_components, int), "n_components must be integer"
        assert isinstance(data_dim, int), "data_dim must be integer"
        assert isinstance(param_dim, int), "param_dim must be integer"
    
        
        self.param_dim = param_dim
        self.data_dim = data_dim

        self.n_components = n_components
        
        #logit-generators for categorical distribution
        self.categorical_logits = DistParametriser(input_dim = data_dim, output_shape = [n_components], layer_sizes = layer_sizes, activation = tf.identity, name = "cat_logits_gen")
        
        #mean-generators for the gaussian distributions
        self.means = [DistParametriser(input_dim = data_dim, output_shape = [param_dim], layer_sizes = layer_sizes, activation = tf.identity, name = "mean_gen") for i in range(n_components)]
        
        #variance_generators
        varmin = 0.01
        varmax = 10.
        self.variances = [DistParametriser(input_dim = data_dim, output_shape = [param_dim], layer_sizes = layer_sizes, activation = lambda x: 0.5*(varmax - varmin) * (tf.tanh(x) + 1) + varmin, name = "var_gen") for i in range(n_components)]
        
    @tf.function()
    def build_dist(self, data):
        
        #gaussian components are created
        components = [tfd.MultivariateNormalDiag(loc = self.means[i](data), scale_diag = self.variances[i](data)) for i in range(self.n_components)]
         
        #mixture dist is created
        dist = tfd.Mixture(cat = tfd.Categorical(logits = self.categorical_logits(data)), components = components)
        
        return dist

    @tf.function()
    def log_prob(self, data, param):
        return self.build_dist(data).log_prob(param)
    
    @tf.function()
    def sample(self, data, sample_shape):
        return self.build_dist(data).sample(sample_shape)
    
    @tf.function()
    def calc_mixture_params(self, data):
        cat_prob = tf.nn.softmax(self.categorical_logits(data))
        means = [self.means[i](data) for i in range(self.n_components)]
        variances = [self.variances[i](data) for i in range(self.n_components)]
        return (cat_prob, means, variances)