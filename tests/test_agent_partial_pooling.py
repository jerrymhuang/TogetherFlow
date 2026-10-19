"""Scientific checks for observable summaries and the matched pooling design."""
import os
os.environ.setdefault('KERAS_BACKEND','jax')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import unittest
import numpy as np
from run_agent_partial_pooling import observed_features, make_bank, BEACONS, DT, wrap


class PartialPoolingExampleTests(unittest.TestCase):
    def test_noiseless_single_agent_weight_is_recoverable_from_observations(self):
        positions=np.zeros((1,100,1,2))
        rotations=np.zeros((1,100,1))
        target=np.arctan2(BEACONS[0,1],BEACONS[0,0])
        for t in range(1,100):
            rotations[:,t]=rotations[:,t-1]+DT*.7*wrap(target-rotations[:,t-1])
        features=observed_features(positions,rotations)
        np.testing.assert_allclose(features[...,0],.7,atol=1e-6)
        np.testing.assert_allclose(features[...,4],0,atol=1e-6)

    def test_features_follow_agent_permutation(self):
        rng=np.random.default_rng(15)
        positions=rng.uniform(-2,2,(3,20,4,2))
        rotations=rng.uniform(0,2*np.pi,(3,20,4))
        perm=[2,0,3,1]
        expected=observed_features(positions,rotations)[:,perm]
        actual=observed_features(positions[:,:,perm],rotations[:,:,perm])
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        self.assertTrue(np.isfinite(actual).all())

    def test_complete_pooling_collapses_and_uses_same_feature_shape(self):
        complete,ct,_=make_bank(4,112,4,'complete')
        partial,pt,_=make_bank(4,112,4,'partial')
        np.testing.assert_allclose(ct[:,4:],np.repeat(ct[:,:1],4,axis=1),atol=1e-7)
        np.testing.assert_array_equal(ct[:,3],0)
        self.assertTrue(np.any(np.std(pt[:,4:],axis=1)>0))
        self.assertEqual(complete['features'].shape,partial['features'].shape)
        self.assertEqual(complete['targets'].shape,(4,3))
        self.assertEqual(partial['targets'].shape,(4,8))


if __name__=='__main__':
    unittest.main()
