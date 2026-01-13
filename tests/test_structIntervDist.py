import pytest
import numpy as np
from sid_py.structIntervDist import structIntervDist

def test_sid_example_from_rd():
    # Example graphs from structIntervDist.Rd
    G = np.array([
        [0,1,1,1,1],
        [0,0,1,1,1],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ]).T # Transpose if needed based on R's rbind vs Python's row-major

    H1 = np.array([
        [0,1,1,1,1],
        [0,0,1,1,1],
        [0,0,0,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ]).T

    result = structIntervDist(G, H1)
    # Check if result contains the 'sid' key as defined in R
    assert "sid" in result
    assert result["sid"] >= 0