# Built-in imports
from typing import List, Dict
import os

# Dependency imports: 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, ELU
from tensorflow.keras.models import Model

# Import the GravyFlow module.
import gravyflow as gf

log_dir = 'logs/profile'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    profile_batch = '500,520'
)


# Set up the environment using gf.env() and return a tf.distribute.Strategy object.
env : tf.distribute.Strategy = gf.env(
	memory_to_allocate_tf=8000
)

def create_gabbard(
        input_shape_onsource : int, 
        input_shape_offsource : int
    ) -> tf.keras.Model:
    
    # Define the inputs based on the dictionary keys and expected shapes
    # Replace `input_shape_onsource` and `input_shape_offsource` with the actual shapes
    onsource_input = Input(shape=input_shape_onsource, name="ONSOURCE")
    offsource_input = Input(shape=input_shape_offsource, name="OFFSOURCE")

    # Pass the inputs to your custom Whiten layer
    # Assuming your Whiten layer can handle multiple inputs
    whiten_output = gf.Whiten()([onsource_input, offsource_input])

    x = gf.Reshape()(whiten_output)
    
    # Convolutional and Pooling layers
    x = Conv1D(8, 64, padding='same', name="Convolutional_1")(x)
    x = ELU(name="ELU_1")(x)
    x = MaxPooling1D(pool_size=4, strides=4, name="Pooling_1", padding="same")(x)
    
    x = Conv1D(8, 32, padding='same', name="Convolutional_2")(x)
    x = ELU(name="ELU_2")(x)
    x = Conv1D(16, 32, padding='same', name="Convolutional_3")(x)
    x = ELU(name="ELU_3")(x)
    x = MaxPooling1D(pool_size=4, strides=4, name="Pooling_3", padding="same")(x)
    
    # Flatten layer
    x = Flatten(name="Flatten")(x)
    
    # Dense layers with dropout
    x = Dense(64, name="Dense_1")(x)
    x = ELU(name="ELU_7")(x)
    x = Dropout(0.5, name="Dropout_1")(x)
    
    x = Dense(64, name="Dense_2")(x)
    x = ELU(name="ELU_8")(x)
    x = Dropout(0.5, name="Dropout_2")(x)
    
    outputs = Dense(1, activation='sigmoid', name="INJECTION_MASKS")(x)
    
    # Create model
    model = Model(inputs=[onsource_input, offsource_input], outputs=outputs, name="custom")
    
    return model

def adjust_features(features, labels):
    labels['INJECTION_MASKS'] = labels['INJECTION_MASKS'][0]
    return features, labels

with env:
    # This object will be used to obtain real interferometer data based on specified parameters.
    ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, # Specify the observing run (e.g., O3).
        data_quality=gf.DataQuality.BEST,  # Choose the quality of the data (e.g., BEST).
        data_labels=[                      # Define the types of data to include.
            gf.DataLabel.NOISE, 
            gf.DataLabel.GLITCHES
        ],
        segment_order=gf.SegmentOrder.RANDOM, # Order of segment retrieval (e.g., RANDOM).
        force_acquisition=True,               # Force the acquisition of new data.
        cache_segments=False                  # Choose not to cache the segments.
    )

    # Initialize the noise generator wrapper:
    # This wrapper will use the ifo_data_obtainer to generate real noise based on the specified parameters.
    noise: gf.NoiseObtainer = gf.NoiseObtainer(
        ifo_data_obtainer=ifo_data_obtainer, # Use the previously set up IFODataObtainer object.
        noise_type=gf.NoiseType.REAL,        # Specify the type of noise as REAL.
        ifos=gf.IFO.L1                       # Specify the interferometer (e.g., LIGO Livingston L1).
    )

    scaling_method : gf.ScalingMethod = gf.ScalingMethod(
        value=gf.Distribution(
            min_=8.0,
            max_=15.0,
            type_=gf.DistributionType.UNIFORM
        ),
        type_=gf.ScalingTypes.SNR
    )

    # Define a uniform distribution for the mass of the first object in solar masses.
    mass_1_distribution_msun : gf.Distribution = gf.Distribution(
        min_=10.0, 
        max_=60.0, 
        type_=gf.DistributionType.UNIFORM
    )

    # Define a uniform distribution for the mass of the second object in solar masses.
    mass_2_distribution_msun : gf.Distribution = gf.Distribution(
        min_=10.0, 
        max_=60.0, 
        type_=gf.DistributionType.UNIFORM
    )

    # Define a uniform distribution for the inclination of the binary system in radians.
    inclination_distribution_radians : gf.Distribution = gf.Distribution(
        min_=0.0, 
        max_=np.pi, 
        type_=gf.DistributionType.UNIFORM
    )

    # Initialize a PhenomD waveform generator with the defined distributions.
    # This generator will produce waveforms with randomly varied masses and inclination angles.
    phenom_d_generator : gf.WaveformGenerator = gf.cuPhenomDGenerator(
        mass_1_msun=mass_1_distribution_msun,
        mass_2_msun=mass_2_distribution_msun,
        inclination_radians=inclination_distribution_radians,
        scaling_method=scaling_method,
        injection_chance=0.5 # Set so half produced examples will not contain this signal
    )
    
    training_dataset : tf.data.Dataset = gf.Dataset(       
        noise_obtainer=noise,
        waveform_generators=phenom_d_generator,
        input_variables=[
            gf.ReturnVariables.ONSOURCE, 
            gf.ReturnVariables.OFFSOURCE, 
        ],
        output_variables=[
            gf.ReturnVariables.INJECTION_MASKS
        ]
    ).map(adjust_features)

    validation_dataset : tf.data.Dataset = gf.Dataset(       
        noise_obtainer=noise,
        waveform_generators=phenom_d_generator,
        seed=1001, # Implement different seed to generate different waveforms,
        group="validate", # Ensure noise is pulled from those marked for validation.
        input_variables=[
            gf.ReturnVariables.ONSOURCE, 
            gf.ReturnVariables.OFFSOURCE, 
        ],
        output_variables=[
            gf.ReturnVariables.INJECTION_MASKS
        ]
    ).map(adjust_features)

    testing_dataset : tf.data.Dataset = gf.Dataset(       
        noise_obtainer=noise,
        waveform_generators=phenom_d_generator,
        seed=1002, # Implement different seed to generate different waveforms,
        group="test", # Ensure noise is pulled from those marked for validation.
        input_variables=[
            gf.ReturnVariables.ONSOURCE, 
            gf.ReturnVariables.OFFSOURCE, 
        ],
        output_variables=[
            gf.ReturnVariables.INJECTION_MASKS
        ]
    ).map(adjust_features)
    
    for input_example, _ in training_dataset.take(1):
        input_shape_onsource = input_example["ONSOURCE"].shape[1:]  # Exclude batch dimension    
        input_shape_offsource = input_example["OFFSOURCE"].shape[1:] 
    
    model = create_gabbard(input_shape_onsource, input_shape_offsource)

    # Now you can print the model summary
    model.summary()
    
    # Model compilation
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Or any other loss function appropriate for your task
        metrics=['accuracy']
    )

    examples_per_epoch : int = int(1E56)
    num_validation_examples : int = int(1E4)
    num_testing_examples : int = int(1E4)
    
    history = model.fit(
        training_dataset,
        epochs=10,  # Number of epochs to train for
        steps_per_epoch=examples_per_epoch // gf.Defaults.num_examples_per_batch,
        validation_data=validation_dataset,
        validation_steps=num_validation_examples // gf.Defaults.num_examples_per_batch, #Ensure this is set as dataset is uncapped
        callbacks=[tensorboard_callback]
    )

    model.evaluate(
        testing_dataset, 
        steps=num_testing_examples // gf.Defaults.num_examples_per_batch
    )
