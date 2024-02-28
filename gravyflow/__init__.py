"""
This module initializes the GravyFlow library, setting up necessary imports and configurations.
"""

# Standard library imports
import warnings

# Suppress specific LAL warning when running in an ipython kernel
warnings.filterwarnings("ignore", category=UserWarning, message="Wswiglal-redir-stdio")

# Conditional import with specific exception handling
try:
    from .cuphenom.python.cuphenom import imrphenomd
except ImportError as e:
    print(f"Failed to import cuphenom because: {e}.")

# Local application/library specific imports
from .src.dataset.config import Defaults
from .src.utils.tensor import (
    DistributionType, Distribution, randomise_arguments,
    replace_nan_and_inf_with_zero, expand_tensor, batch_tensor,
    crop_samples, rfftfreq, get_element_shape, check_tensor_integrity,
    set_random_seeds
)
from .src.utils.gpu import (setup_cuda, find_available_GPUs, get_tf_memory_usage, env, 
    get_memory_array, get_gpu_utilization_array)
from .src.utils.io import (
    open_hdf5_file, ensure_directory_exists, replace_placeholders,
    transform_string, snake_to_capitalized_spaces, is_redirected, load_history,
    CustomHistorySaver, EarlyStoppingWithLoad, PrintWaitCallback, save_dict_to_hdf5,
    get_file_parent_path
)
from .src.utils.processes import (Heart, HeartbeatCallback, Process, Manager, 
    explain_exit_code)
from .src.dataset.tools.psd import psd
from .src.dataset.tools.snr import snr, scale_to_snr
from .src.dataset.features.waveforms.wnb import wnb
from .src.dataset.conditioning.conditioning import spectrogram, spectrogram_shape
from .src.model.genetics import HyperParameter, HyperInjectionGenerator, ModelGenome
from .src.utils.git import get_current_repo
from .src.model.model import (
    BaseLayer, Reshape, DenseLayer, FlattenLayer, ConvLayer,
    PoolLayer, DropLayer, BatchNormLayer, WhitenLayer, WhitenPassLayer,
    Model, PopulationSector, Population
)
from .src.dataset.conditioning.whiten import whiten, Whiten, WhitenPass
from .src.dataset.conditioning.pearson import rolling_pearson
from .src.dataset.conditioning.detector import IFO, Network
from .src.dataset.noise.acquisition import (
    DataQuality, DataLabel, SegmentOrder, AcquisitionMode, ObservingRun,
    IFOData, IFODataObtainer
)
from .src.dataset.noise.noise import NoiseType, NoiseObtainer
from .src.dataset.features.injection import (
    ScalingOrdinality, ScalingType, ScalingTypes, ScalingMethod, ReturnVariables,
    WaveformGenerator, WaveformParameter, WaveformParameters, WNBGenerator, 
    cuPhenomDGenerator, IncoherentGenerator, InjectionGenerator,
    roll_vector_zero_padding, generate_mask, is_not_inherited, 
    batch_injection_parameters
)
from .src.dataset.dataset import data, Dataset
from .src.utils.plotting import (
    generate_strain_plot, generate_psd_plot, generate_spectrogram, generate_correlation_plot
)
from .src.model.validate import Validator
from .src.dataset.features.glitch import GlitchType, get_glitch_times, get_glitch_segments
from .src.utils.alert import send_email

PATH = get_file_parent_path()