"""Scientific invariants for individual-radius partial pooling."""
import os
os.environ.setdefault('KERAS_BACKEND','jax')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import unittest
import numpy as np
from numba import njit
from togetherflow import TogetherFlowSimulator
from togetherflow.influences import combined_influences
from run_agent_radius_pooling import (radius_hierarchy_prior, observed_features,
    transform_targets,physical_targets,PARAMETER_NAMES)


@njit
def zero_radius_spread():
    return np.array([.3,2.5,1.,.2,0.,0.],dtype=np.float32)


class IndividualRadiusTests(unittest.TestCase):
    def test_radii_control_receiver_neighborhoods_asymmetrically(self):
        args=dict(agent_positions=np.array([[0.,0.],[1.,0.],[3.,0.]]),
            agent_rotations=np.zeros(3),beacon_positions=np.array([[2.,2.]]),
            beacon_strengths=np.ones(1),salience_sensitivity=0.,reference_radii=np.zeros(0),
            influence_weights=np.ones(3),velocity=0.,internal_focus=0.,
            obstacles=np.zeros((0,3)),beacon_assignment=np.zeros(0,dtype=np.int64))
        actual=combined_influences(**args,agent_radii=np.array([.5,1.5,4.]))
        np.testing.assert_array_equal(actual[2],[0,1,2])
        scalar=combined_influences(**args,sensing_radius=2.5)
        homogeneous=combined_influences(**args,agent_radii=np.full(3,2.5))
        for a,b in zip(scalar,homogeneous):
            np.testing.assert_array_equal(a,b)

    def simulator(self,export,prior=radius_hierarchy_prior):
        return TogetherFlowSimulator(num_agents=12,time_horizon=1,seed=991,
            prior=prior,param_names=('mu_w','mu_r','v','noise','sigma_w','sigma_r'),
            include_agent_weights=True,include_agent_radii=export)

    def test_export_does_not_change_observations(self):
        a=self.simulator(False).sample(4)
        b=self.simulator(True).sample(4)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(b['agent_radii'].shape,(4,12))
        self.assertTrue(np.all(b['agent_radii']>0))
        initial=b['positions'][:,0].reshape(4,12,2)
        distances=np.linalg.norm(initial[:,:,None,:]-initial[:,None,:,:],axis=-1)
        expected=((distances>0)&(distances<=b['agent_radii'][:,:,None])).sum(axis=-1)
        np.testing.assert_array_equal(b['neighbors'][:,1],expected)
        self.assertTrue(np.any(np.std(b['agent_radii'],axis=1)>0))
        np.testing.assert_array_equal(b['agent_radii'],self.simulator(True).sample(4)['agent_radii'])

    def test_zero_radius_spread_collapses(self):
        data=self.simulator(True,zero_radius_spread).sample(4)
        np.testing.assert_allclose(data['agent_radii'],2.5)
        np.testing.assert_allclose(data['agent_weights'],.3,atol=1e-7)
        with self.assertRaises(ValueError):
            TogetherFlowSimulator(include_agent_radii=True)

    def test_radius_upper_bounds_radii_and_keeps_median(self):
        @njit
        def wide_spread():
            return np.array([.3,8.,1.,.2,0.,3.],dtype=np.float32)
        kwargs=dict(num_agents=12,time_horizon=1,seed=991,prior=wide_spread,
            param_names=('mu_w','mu_r','v','noise','sigma_w','sigma_r'),
            include_agent_weights=True,include_agent_radii=True)
        bounded=TogetherFlowSimulator(radius_upper=10.,**kwargs).sample(64)['agent_radii']
        self.assertTrue(np.all((bounded>0)&(bounded<10)))
        np.testing.assert_allclose(np.median(bounded),8.,rtol=.1)
        self.assertGreater(TogetherFlowSimulator(**kwargs).sample(64)['agent_radii'].max(),10)
        collapsed=self.simulator(True,zero_radius_spread)
        collapsed.radius_upper=10.
        np.testing.assert_allclose(collapsed.sample(4)['agent_radii'],2.5,rtol=1e-6)
        with self.assertRaises(ValueError):
            TogetherFlowSimulator(radius_upper=0.)

    def test_observable_features_follow_agent_order(self):
        rng=np.random.default_rng(2)
        pos=rng.uniform(-2,2,(2,10,12,2))
        rot=rng.uniform(0,2*np.pi,(2,10,12))
        permutation=rng.permutation(12)
        a=observed_features(pos,rot)[:,permutation]
        b=observed_features(pos[:,:,permutation],rot[:,:,permutation])
        np.testing.assert_allclose(a,b,atol=1e-6)
        self.assertTrue(np.isfinite(a).all())

    def test_all_thirty_parameter_transforms_roundtrip(self):
        self.assertEqual(len(PARAMETER_NAMES),30)
        truth=np.array([[.4,.2,2.5,.3,1.,.2]+[.5]*12+[2.]*12])
        np.testing.assert_allclose(physical_targets(transform_targets(truth)),truth,rtol=1e-6)


if __name__=='__main__': unittest.main()
