import tensorflow as tf
from dataclasses import dataclass
from typing import Union, List, Dict, Optional
import numpy as np
from copy import deepcopy

from keras.layers import Lambda
from keras import backend as K

import tensorflow_probability as tfp

def negative_loglikelihood(targets, estimated_distribution):
    
    targets = tf.cast(targets, dtype = tf.float32)
    return -estimated_distribution.log_prob(targets)

def negative_loglikelihood_(y_true, y_pred):
    loc, scale = tf.unstack(tf.cast(y_pred, dtype = tf.float32), axis=-1)
    y_true = tf.cast(y_true, dtype = tf.float32)

    truncated_normal = tfp.distributions.TruncatedNormal(
        loc,
        scale + 1.0E-5,
        0.0,
        1000.0,
        validate_args=False,
        allow_nan_stats=True,
        name='TruncatedNormal'
    )
        
    return -truncated_normal.log_prob(y_true)

tfd = tfp.distributions
tfpl = tfp.layers

class IndependentGamma(tfpl.DistributionLambda):
    """An independent Gamma Keras layer."""

    def __init__(self,
                 event_shape=(),
                 convert_to_tensor_fn=tfd.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        super(IndependentGamma, self).__init__(
            lambda t: self.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )
        self._event_shape = event_shape
        self._validate_args = validate_args

    def new(self, params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'IndependentGamma'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = tf.reshape(event_shape, [-1])
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape
            ], axis=0)
            alpha_params, beta_params = tf.split(params, 2, axis=-1)
            alpha_params = tf.nn.softplus(tf.cast(alpha_params, dtype = tf.float32)) + 1.0E-5
            beta_params = tf.nn.softplus(tf.cast(beta_params, dtype = tf.float32)) + 1.0E-5
            
            return tfd.Independent(
                tfd.Gamma(
                    concentration=tf.reshape(alpha_params, output_shape),
                    rate=tf.reshape(beta_params, output_shape),
                    validate_args=validate_args),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'IndependentGamma_params_size'):
            event_shape = tf.convert_to_tensor(event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(2) * np.prod(event_shape)

    def get_config(self):
        """Returns the config of this layer."""
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': self.convert_to_tensor_fn,
            'validate_args': self._validate_args
        }
        base_config = super(IndependentGamma, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class IndependentFoldedNormal(tfpl.DistributionLambda):
    """An independent folded normal Keras layer."""

    def __init__(self,
                 event_shape=(),
                 convert_to_tensor_fn=tfd.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        super(IndependentFoldedNormal, self).__init__(
            lambda t: self.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )
        self._event_shape = event_shape
        self._validate_args = validate_args

    def new(self, params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'IndependentFoldedNormal'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = tf.reshape(event_shape, [-1])
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape
            ], axis=0)
            loc_params, scale_params = tf.split(params, 2, axis=-1)
            loc_params = tf.cast(loc_params, dtype = tf.float32)  + 1.0E-6
            scale_params = tf.cast(scale_params, dtype = tf.float32) + 1.0E-6

            return tfd.Independent(
                tfd.TransformedDistribution(
                    distribution=tfd.Normal(
                        loc=tf.math.softplus(tf.reshape(loc_params, output_shape)),
                        scale=tf.math.softplus(tf.reshape(scale_params, output_shape)),
                        validate_args=validate_args),
                    bijector=tfb.AbsoluteValue(),
                    validate_args=validate_args),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'IndependentFoldedNormal_params_size'):
            event_shape = tf.convert_to_tensor(event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(2) * np.prod(event_shape)

    def get_config(self):
        """Returns the config of this layer."""
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': self.convert_to_tensor_fn,
            'validate_args': self._validate_args
        }
        base_config = super(IndependentFoldedNormal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IndependentTruncNormal(tfpl.DistributionLambda):
    """An independent truncated normal Keras layer."""

    def __init__(self,
                 event_shape=(),
                 low=  0.000,
                 high= 100.0, #float("inf"),
                 convert_to_tensor_fn=tfd.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        self.low = low
        self.high = high
        super(IndependentTruncNormal, self).__init__(
            lambda t: self.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )
        self._event_shape = event_shape
        self._validate_args = validate_args

    def new(self, params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'IndependentTruncNormal'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = tf.reshape(event_shape, [-1])
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape
            ], axis=0)
            loc_params, scale_params = tf.split(params, 2, axis=-1)
            loc_params = tf.cast(loc_params, dtype = tf.float32)  + 1.0E-5
            scale_params = tf.cast(loc_params, dtype = tf.float32) + 1.0E-5
            
            return tfd.Independent(
                tfd.TruncatedNormal(
                    loc=tf.reshape(loc_params, output_shape),
                    scale=tf.math.softplus(tf.reshape(scale_params, output_shape)),
                    low=self.low,
                    high=self.high,
                    validate_args=validate_args),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'IndependentTruncNormal_params_size'):
            event_shape = tf.convert_to_tensor(event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(2) * np.prod(event_shape)

    def get_config(self):
        """Returns the config of this layer."""
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': self.convert_to_tensor_fn,
            'validate_args': self._validate_args
        }
        base_config = super(IndependentTruncNormal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

tfb = tfp.bijectors
class BetaPrime(tfb.Bijector):
    """Bijector for the beta prime distribution."""
    def __init__(self, validate_args=False, name="beta_prime"):
        super(BetaPrime, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=0, name=name)

    def _forward(self, x):
        return x / (1 - x)

    def _inverse(self, y):
        return y / (1 + y)

    def _forward_log_det_jacobian(self, x):
        return - tf.math.log1p(-x)

    def _inverse_log_det_jacobian(self, y):
        return - tf.math.log1p(y)

class IndependentBetaPrime(tfpl.DistributionLambda):
    """An independent Beta prime Keras layer."""
    def __init__(self,
                 event_shape=(),
                 convert_to_tensor_fn=tfd.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        super(IndependentBetaPrime, self).__init__(
            lambda t: self.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )
        self._event_shape = event_shape
        self._validate_args = validate_args

    def new(self, params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'IndependentBetaPrime'):
            
            params = tf.cast(tf.convert_to_tensor(params, name='params'), tf.float32)
            event_shape = tf.reshape(event_shape, [-1])
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape
            ], axis=0)
            concentration1_params, concentration0_params = tf.split(params, 2, axis=-1)
            concentration1_params = tf.math.softplus(tf.reshape(concentration1_params, output_shape))
            concentration0_params = tf.math.softplus(tf.reshape(concentration0_params, output_shape))

            return tfd.Independent(
                tfd.TransformedDistribution(
                    distribution=tfd.Beta(
                        concentration1=concentration1_params,
                        concentration0=concentration0_params,
                        validate_args=validate_args),
                    bijector=BetaPrime(),
                    validate_args=validate_args),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'IndependentBetaPrime_params_size'):
            event_shape = tf.convert_to_tensor(event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(2) * np.prod(event_shape)

    def get_config(self):
        """Returns the config of this layer."""
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': self.convert_to_tensor_fn,
            'validate_args': self._validate_args
        }
        base_config = super(IndependentBetaPrime, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@dataclass
class HyperParameter:
    possible_values: Dict[str, Union[str, List[Union[int, float]]]]
    value: Union[int, float, str] = None
    
    def __post_init__(self):
        self.randomize()

    def randomize(self):
        """
        Randomizes this hyperparameter based on its possible_values.
        """
        value_type = self.possible_values['type']
        possible_values = self.possible_values['values']
        
        if value_type == 'list':
            self.value = np.random.choice(possible_values)
        elif value_type == 'power_2_range':
            power_low, power_high = map(int, np.log2(self.possible_values['values']))
            power = np.random.randint(power_low, power_high + 1)
            self.value = 2**power
        elif value_type == 'int_range':
            low, high = self.possible_values['values']
            self.value = np.random.randint(low, high + 1)
        elif value_type == 'float_range':
            self.value = np.random.uniform(*possible_values)

    def mutate(self, mutation_rate: float):
        """
        Returns a new HyperParameter with a mutated value, based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_param: New HyperParameter instance with potentially mutated value.
        """
        mutated_param = deepcopy(self) 
        if np.random.random() < mutation_rate:
            value_type = self.possible_values['type']
            possible_values = self.possible_values['values']
            if value_type in ['int_range', 'power_2_range', 'float_range']:
                mutation = np.random.normal(scale=(possible_values[1] - possible_values[0]) / 10)
                new_value = self.value + mutation
                # Make sure new value is in the allowed range
                new_value = min(max(new_value, *possible_values))
                mutated_param.value = new_value
            else:
                mutated_param.randomize()
            return mutated_param
        else:
            return deepcopy(self)
        
def hp(N):
    return HyperParameter({'type': 'list', 'values': [N]})

def ensure_hp(parameter):
    return parameter if isinstance(parameter, HyperParameter) else hp(parameter)

@dataclass
class BaseLayer:
    layer_type: str
    activation: Union[HyperParameter, str]
    mutable_attributes: List
    
    def randomize(self):
        """
        Randomizes all mutable attributes of this layer.
        """
        for attribute in self.mutable_attributes:
            attribute.randomize()
            
    def mutate(self, mutation_rate: float):
        """
        Returns a new layer with mutated hyperparameters based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_layer: New BaseLayer instance with potentially mutated hyperparameters.
        """
        mutated_layer = deepcopy(self)
        for attribute_name in mutated_layer.mutable_attributes:
            mutated_value = getattr(mutated_layer, attribute_name).mutate(mutation_rate)
            setattr(mutated_layer, attribute_name, mutated_value)
        return mutated_layer

@dataclass
class DenseLayer(BaseLayer):
    units: [HyperParameter, int]

    def __init__(self, units: HyperParameter, activation: HyperParameter):
        """
        Initializes a DenseLayer instance.
        
        Args:
        units: HyperParameter specifying the number of units in this layer.
        activation: HyperParameter specifying the activation function for this layer.
        """
        self.layer_type = "Dense"
        self.activation = ensure_hp(activation)
        self.units = ensure_hp(units)
        self.mutable_attributes = [self.activation, self.units]

@dataclass
class ConvLayer(BaseLayer):
    filters: HyperParameter
    kernel_size: HyperParameter
    strides: HyperParameter
    
    def __init__(self, 
        filters: HyperParameter, 
        kernel_size: HyperParameter, 
        activation: HyperParameter, 
        strides: HyperParameter = hp(1)
        ):
        """
        Initializes a ConvLayer instance.
        
        Args:
        filters: HyperParameter specifying the number of filters in this layer.
        kernel_size: HyperParameter specifying the kernel size in this layer.
        activation: HyperParameter specifying the activation function for this layer.
        strides: HyperParameter specifying the stride length for this layer.
        """
        self.layer_type = "Convolutional"
        self.activation = ensure_hp(activation)
        self.filters = ensure_hp(filters)
        self.kernel_size = ensure_hp(kernel_size)
        self.strides = ensure_hp(strides)

        self.padding = hp("same")
        
        self.mutable_attributes = [self.activation, self.filters, self.kernel_size, self.strides]
        
@dataclass
class PoolLayer(BaseLayer):
    pool_size: HyperParameter
    strides: HyperParameter
    
    def __init__(self, 
        pool_size: HyperParameter, 
        strides: Optional[Union[HyperParameter, int]] = None
        ):
        """
        Initializes a PoolingLayer instance.
        
        Args:
        pool_size: HyperParameter specifying the size of the pooling window.
        strides: HyperParameter specifying the stride length for moving the pooling window.
        """
        self.layer_type = "Pooling"
        self.pool_size = ensure_hp(pool_size)
        
        if strides is None:
            self.strides = self.pool_size
        else:
            self.strides = ensure_hp(strides)
        
        self.padding = hp("same")
        self.mutable_attributes = [self.pool_size, self.strides]
        
class DropLayer(BaseLayer):
    rate: HyperParameter
    
    def __init__(self, rate: Union[HyperParameter, float]):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: HyperParameter specifying the dropout rate for this layer.
        """
        self.layer_type = "Dropout"
        self.rate = ensure_hp(rate)
        self.mutable_attributes = [self.rate]

def randomizeLayer(layer_types: List[str], default_layers: Dict[str, BaseLayer]) -> BaseLayer:
    """
    Returns a randomized layer of a random type.
    
    Args:
    layer_types: List of possible layer types.
    default_layers: Dictionary mapping layer types to default layers of that type.
    
    Returns:
    layer: New BaseLayer instance of a random type, with randomized hyperparameters.
    """
    layer_type = np.random.choice(layer_types)
    layer = default_layers[layer_type]
    layer.randomize()
    return layer

"""
class TruncatedNormal(tf.keras.layers.Layer):
    def __init__(self, event_shape=1, **kwargs):
        super(TruncatedNormal, self).__init__(**kwargs)
        self.event_shape = event_shape

    def build(self, input_shape):
        # Input should be of shape (batch_size, 2) 
        # with the first column as the mean and the second column as the standard deviation
        assert input_shape[-1] == 2

    def call(self, inputs):
        loc, scale = tf.unstack(inputs, axis=-1)
        return loc, scale

    def get_config(self):
        return {'event_shape': self.event_shape}
"""

def cap_value(x):
    return K.clip(x, 1.0e-5, 1000)  # values will be constrained to [-1, 1]

class ModelBuilder:
    def __init__(self, layers: List[BaseLayer], optimizer: str, loss: str, batch_size: int):
        """
        Initializes a ModelBuilder instance.
        
        Args:
        layers: List of BaseLayer instances making up the model.
        optimizer: Optimizer to use when training the model.
        loss: Loss function to use when training the model.
        batch_size: Batch size to use when training the model.
        """        
        self.layers = layers
        self.batch_size = ensure_hp(batch_size)
        self.optimizer = ensure_hp(optimizer)
        self.loss = ensure_hp(loss)
        
        self.fitness = []
        
        self.metrics = []

    def build_model(self, input_shape: Union[int, tuple], output_shape: int):
        """
        Builds the model.
        
        Args:
        input_shape: Shape of the input data.
        output_shape: Shape of the output data.
        """        
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='onsource'))
        model.add(tf.keras.layers.Reshape((-1, 1)))

        for layer in self.layers:
            if layer.layer_type == "Dense":
                model.add(
                    tf.keras.layers.Dense(
                        layer.units.value, 
                        activation=layer.activation.value
                    )
                )
            elif layer.layer_type == "Convolutional":
                model.add(
                    tf.keras.layers.Conv1D(
                        layer.filters.value, 
                        (layer.kernel_size.value,), 
                        strides=(layer.strides.value,), 
                        activation=layer.activation.value,
                        padding = layer.padding.value
                    )
                )
            elif layer.layer_type == "Pooling":
                model.add(
                    tf.keras.layers.MaxPool1D(
                        (layer.pool_size.value,),
                        strides=(layer.strides.value,),
                        padding = layer.padding.value
                    )
                )
            elif layer.layer_type == "Dropout":
                model.add(
                    tf.keras.layers.Dropout(
                        layer.rate.value
                    )
                )
            else:
                raise ValueError(f"Layer type '{layer.layer_type.value}' not recognized")
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(2, activation='linear', dtype='float32', bias_initializer=tf.keras.initializers.Constant([1.0, 2.0])))#, name='snr',  bias_initializer=tf.keras.initializers.Constant([1.0, 2.0])))  # Different biases for each unit))
        #model.add(Lambda(cap_value))
        
        model.add(IndependentFoldedNormal(1, name='snr'))
        
        #model.add(tfp.layers.IndependentNormal(1, name = 'snr'))
        
        model.compile(optimizer=self.optimizer.value, loss=self.loss.value,
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                
        self.model = model

    def train_model(
        self, 
        train_dataset: tf.data.Dataset, 
        num_batches: int, 
        num_epochs: int = 1
        ):
        """
        Trains the model.
        
        Args:
        train_dataset: Dataset to train on.
        num_epochs: Number of epochs to train for.
        """
        self.metrics.append(
            self.model.fit(
                train_dataset, 
                epochs=num_epochs, 
                steps_per_epoch=num_batches,
                batch_size=self.batch_size.value
            )
        )
        
    def validate_model(self, validation_dataset: tf.data.Dataset):
        pass

    def test_model(self, validation_datasets: tf.data.Dataset, num_batches: int):
        """
        Tests the model.
        
        Args:
        validation_datasets: Dataset to test on.
        batch_size: Batch size to use when testing.
        """
        
        self.fitness.append(1.0 / self.model.evaluate(validation_datasets, steps=num_batches)[0])
        
        return self.fitness[-1]
    
    def check_death(self, patience):
        return np.all(self.fitness[-int(patience)] > self.fitness[-int(patience)+1:])
        
    def summary(self):
        """
        Prints a summary of the model.
        """
        self.model.summary()
        
    @staticmethod
    def crossover(parent1: 'ModelBuilder', parent2: 'ModelBuilder') -> 'ModelBuilder':
        """
        Creates a new model whose hyperparameters are a combination of two parent models.
        The child model is then returned.
        """
        # Determine the shorter and longer layers lists
        short_layers, long_layers = (parent1.layers, parent2.layers) if len(parent1.layers) < len(parent2.layers) else (parent2.layers, parent1.layers)

        # Choose a random split point in the shorter layers list
        split_point = np.random.randint(1, len(short_layers))

        # Choose which parent to take the first part from
        first_part, second_part = (short_layers[:split_point], long_layers[split_point:]) if np.random.random() < 0.5 else (long_layers[:split_point], short_layers[split_point:])
        child_layers = first_part + second_part

        child_model = ModelBuilder(child_layers, parent1.optimizer, parent1.loss, parent1.batch_size)

        return child_model
    
    def mutate(self, mutation_rate: float) -> 'ModelBuilder':
        """
        Returns a new model with mutated layers based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_model: New ModelBuilder instance with potentially mutated layers.
        """
        mutated_layers = [layer.mutate(mutation_rate) for layer in self.layers]
        mutated_model = ModelBuilder(mutated_layers, self.optimizer, self.loss, self.batch_size)

        return mutated_model

class Population:
    def __init__(
        self, 
        initial_population_size: int, 
        max_population_size: int,
        genome_template: int,
        input_size : int,
        output_size : int
    ):
        self.initial_population_size = initial_population_size
        self.current_population_size = initial_population_size
        self.max_population_size = max_population_size
        self.population = \
            self.initilize_population(
                genome_template, 
                input_size, 
                output_size
            )
        self.fitnesses = np.ones(self.current_population_size)
        
    def initilize_population(self, genome_template, input_size, output_size):
    
        population = []
        for j in range(self.initial_population_size):   
            layers = []
            
            genome_template['base']['num_layers'].randomize()
            genome_template['base']['optimizer'].randomize()
            self.batch_size = genome_template['base']['batch_size'].randomize()
            
            for i in range(genome_template['base']['num_layers'].value):
                layers.append(
                    randomizeLayer(*genome_template['layers'][i])
                )
                
            # Create an instance of DenseModelBuilder with num_neurons list
            builder = ModelBuilder(
                layers, 
                optimizer = genome_template['base']['optimizer'], 
                loss = hp(negative_loglikelihood), 
                batch_size = genome_template['base']['batch_size']
            )

            # Build the model with input shape (input_dim,)
            builder.build_model(
                input_shape=(input_size,), 
                output_shape = output_size
            )
            population.append(builder)
            builder.summary()
            
        return population
    
    def roulette_wheel_selection(self):
        """
        Performs roulette wheel selection on the population.

        Args:
            population (list): The population of individuals.
            fitnesses (list): The fitness of each individual in the population.

        Returns:
            The selected individual from the population.
        """

        # Convert the fitnesses to probabilities.
        total_fit = sum(self.fitnesses)
        prob = [fit/total_fit for fit in self.fitnesses]

        # Calculate the cumulative probabilities.
        cumulative_probs = np.cumsum(prob)

        # Generate a random number in the range [0, 1).
        r = np.random.rand()

        # Find the index of the individual to select.
        for i in range(len(self.population)):
            if r <= cumulative_probs[i]:
                return i

        # If we've gotten here, just return the last individual in the population.
        # This should only happen due to rounding errors, and should be very rare.
        return self.population[-1]
    
    def train_population(
        self, 
        num_generations, 
        num_train_examples, 
        num_validate_examples, 
        num_examples_per_batch, 
        ds
    ):
                
        num_train_batches = int(num_train_examples // num_examples_per_batch)
        
        num_validate_batches = int(num_validate_examples // num_examples_per_batch)
        
        for i in range(self.current_population_size):
            training_ds = ds.take(num_train_batches)
            validation_ds = ds.take(num_validate_batches)
            
            model = self.population[i]
            model.train_model(training_ds, num_train_batches)
            self.fitnesses[i] = \
                model.test_model(validation_ds, num_validate_batches)
        
        print(self.fitnesses)
        
        for _ in range(self.current_population_size*(num_generations - 1)):
            training_ds = ds.take(num_train_batches)
            validation_ds = ds.take(num_validate_batches)
            
            i = self.roulette_wheel_selection()
            self.population[i].train_model(training_ds, num_train_batches)
            self.fitnesses[i] = \
                model.test_model(validation_ds, num_validate_batches)
            
            print("is_alive:", model.check_death(10))
                        
            print(self.fitnesses)
            print(mean(self.fitnesses))