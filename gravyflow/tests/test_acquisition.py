import pytest
import numpy as np
import gravyflow as gf

@pytest.mark.parametrize("seed, ifo", [(1000, gf.IFO.L1)])
def test_valid_data_segments_acquisition(
        data_obtainer : gf.IFODataObtainer, 
        seed : int, 
        ifo : gf.IFO
    ) -> None:
    
    """
    Test to ensure the acquisition of valid data segments meets expected criteria.
    """
    segments = data_obtainer.get_valid_segments(seed=seed, ifos=[ifo])
    
    np.testing.assert_array_less(
        10000, 
        len(segments), 
        err_msg=f"Num segments found, {len(segments)}, is too low!"
    )