"""Independent finite-state and lifecycle checks of diagnostic tempering."""

from copy import deepcopy
import json
import unittest

import numpy as np

from quantum_lattice.lattice import square_lattice
from quantum_lattice.simulation.replica_exchange import (
    ReplicaExchangeChain, _swap_log_ratio,
)
from quantum_lattice.simulation.solvable import _compiled_kernel, _in_reference_orbit


class ReplicaExchangeTests(unittest.TestCase):
    def setUp(self):
        self.lattice = square_lattice(3)
        self.fields = np.linspace(-.55,.7,18)

    def make(self, **updates):
        settings = {"betas": [0,.25,.6,1], "backend": "python",
                    "swap_interval_sweeps": 3}
        settings.update(updates)
        return ReplicaExchangeChain(self.lattice,self.fields,1729,**settings)

    def test_swap_ratio_matches_direct_joint_weights_and_detailed_balance(self):
        # Independent root-fixed pseudospins; no estimator or swap helper is
        # used to define the product target or the target state permutation.
        theta=np.ones((256,9),dtype=np.int8)
        theta[:,1:]=1-2*((np.arange(256)[:,None]>>np.arange(8))&1)
        ends=self.lattice.edge_vertex_indices
        states=theta[:,ends[:,0]]*theta[:,ends[:,1]]
        scores=states@self.fields
        for bi,bj in ((0,.25),(.25,.6),(.6,1)):
            wi=np.exp(bi*scores); wi/=wi.sum()
            wj=np.exp(bj*scores); wj/=wj.sum()
            target=wi[:,None]*wj[None,:]
            direct_log=(bi*scores[None,:]+bj*scores[:,None]
                        -bi*scores[:,None]-bj*scores[None,:])
            log_ratio=(bi-bj)*(scores[None,:]-scores[:,None])
            np.testing.assert_allclose(log_ratio,direct_log,atol=2e-15)
            acceptance=np.exp(np.minimum(0,log_ratio))
            flux=target*acceptance
            np.testing.assert_allclose(flux,flux.T,atol=5e-20,rtol=2e-14)
            np.testing.assert_allclose(_swap_log_ratio(bi,bj,scores[3],scores[119]),
                                       direct_log[3,119],atol=2e-15)
        # Cold-favored configuration exchanges into the hot slot only rarely.
        self.assertEqual(_swap_log_ratio(0,1,-2,3),-5)
        self.assertEqual(_swap_log_ratio(0,1,3,-2),5)

    def test_initialization_independent_temperature_fields_and_valid_orbits(self):
        chain=self.make()
        for beta,local in zip(chain.betas,chain.chains):
            np.testing.assert_array_equal(local.fields,beta*self.fields)
            self.assertTrue(_in_reference_orbit(self.lattice,local.spins))
        self.assertEqual(len({json.dumps(c.snapshot()["rng_state"],sort_keys=True)
                              for c in chain.chains}),4)
        self.assertIs(chain.spins,chain.chains[-1].spins)
        self.assertIs(chain.target_spins,chain.spins)
        self.assertEqual(chain.diagnostics()["roundtrips"],[0,0,0,0])
        np.testing.assert_array_equal(chain.visits,np.eye(4,dtype=int))
        with self.assertRaises(ValueError):
            chain.betas[1]=.3

    def test_exchange_swaps_states_and_labels_only_with_alternating_parity(self):
        chain=ReplicaExchangeChain(self.lattice,np.zeros(18),90,betas=[0,.3,.6,1],
                                   backend="python",swap_interval_sweeps=1)
        before=[c.snapshot() for c in chain.chains]
        chain._exchange()
        np.testing.assert_array_equal(chain.walker_labels,[1,0,3,2])
        np.testing.assert_array_equal(chain.swap_attempts,[1,0,1])
        np.testing.assert_array_equal(chain.swap_accepts,[1,0,1])
        for i,j in enumerate([1,0,3,2]):
            after=chain.chains[i].snapshot()
            self.assertEqual(after["spins"],before[j]["spins"])
            for key in ("rng_state","attempted","accepted","lazy","fields"):
                self.assertEqual(after[key],before[i][key])
        chain._exchange()
        np.testing.assert_array_equal(chain.walker_labels,[1,3,0,2])
        np.testing.assert_array_equal(chain.swap_attempts,[1,1,1])
        np.testing.assert_array_equal(chain.swap_accepts,[1,1,1])

    def test_chunking_and_json_restore_are_bit_identical(self):
        direct=self.make(); direct.advance(47)
        chunked=self.make()
        for amount in [0,1,4,2,13,9,18]:
            chunked.advance(amount)
        self.assertEqual(chunked.snapshot(),direct.snapshot())
        restored=ReplicaExchangeChain.from_snapshot(self.lattice,self.fields,
            json.loads(json.dumps(chunked.snapshot())),backend="python")
        for amount in [2,0,3,11]:
            restored.advance(amount)
        direct.advance(16)
        self.assertEqual(restored.snapshot(),direct.snapshot())
        self.assertEqual(restored.diagnostics(),direct.diagnostics())
        for local in restored.chains:
            self.assertTrue(_in_reference_orbit(self.lattice,local.spins))

    def test_visits_roundtrip_counts_and_even_odd_attempts(self):
        # At zero field every swap is accepted, so walker motion is exact.
        chain=ReplicaExchangeChain(self.lattice,np.zeros(18),91,betas=[0,1],
                                   backend="python",swap_interval_sweeps=1)
        chain.advance(5)
        np.testing.assert_array_equal(chain.swap_attempts,[3])
        np.testing.assert_array_equal(chain.swap_accepts,[3])
        np.testing.assert_array_equal(chain.walker_labels,[1,0])
        # Initial cold walker completes cold->hot->cold on the second swap;
        # initial hot walker begins counting only after first reaching cold.
        np.testing.assert_array_equal(chain.roundtrips,[1,1])
        np.testing.assert_array_equal(chain.visits.sum(axis=0),[6,6])
        np.testing.assert_array_equal(chain.visits.sum(axis=1),[6,6])
        restored=ReplicaExchangeChain.from_snapshot(self.lattice,np.zeros(18),
            chain.snapshot(),backend="python")
        self.assertEqual(chain.snapshot(),restored.snapshot())

    @unittest.skipIf(_compiled_kernel is None,"Numba unavailable")
    def test_numba_and_python_have_identical_rng_and_checkpoint_continuation(self):
        python=self.make(); numba=self.make(backend="numba")
        python.advance(19); numba.advance(19)
        self.assertEqual(python.snapshot(),numba.snapshot())
        restored=ReplicaExchangeChain.from_snapshot(self.lattice,self.fields,
            numba.snapshot(),backend="python")
        restored.advance(8); numba.advance(8)
        self.assertEqual(restored.snapshot(),numba.snapshot())

    def test_zero_field_and_plus_start(self):
        chain=ReplicaExchangeChain(self.lattice,np.zeros(18),9,backend="python",start="plus")
        for local in chain.chains:
            np.testing.assert_array_equal(local.spins,np.ones(18))
        chain.advance(20)
        for local in chain.chains:
            self.assertEqual(local.accepted+local.lazy,local.attempted)
        self.assertTrue(all(a==b for a,b in zip(chain.swap_accepts,chain.swap_attempts)))

    def test_invalid_settings_and_fields_are_rejected(self):
        for betas in ([0,0,1],[0,.5,.4,1],[.1,1],[0,.9],[0,np.nan,1],
                      [0,np.inf,1],[0,1+0j],[False,True],[1],None,"ladder"):
            with self.subTest(betas=betas),self.assertRaises(ValueError):
                self.make(betas=betas)
        for update in ({"swap_interval_sweeps":0},{"swap_interval_sweeps":True},
                       {"swap_interval_sweeps":2.5},{"start":"edges"},{"start":[]},
                       {"lazy_probability":None},{"lazy_probability":True},
                       {"lazy_probability":np.nan},{"lazy_probability":1}):
            with self.subTest(update=update),self.assertRaises(ValueError):
                self.make(**update)
        for fields in (np.zeros(17),np.full(18,np.inf),np.ones(18,dtype=complex)):
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                ReplicaExchangeChain(self.lattice,fields,9)
        for seed in (True,-1,[],[-1,2]):
            with self.subTest(seed=seed),self.assertRaises(ValueError):
                ReplicaExchangeChain(self.lattice,self.fields,seed)
        for amount in (-1,True,.5):
            with self.subTest(sweeps=amount),self.assertRaises(ValueError):
                self.make().advance(amount)

    def test_snapshot_is_independent_and_corruption_is_rejected(self):
        chain=self.make(); chain.advance(19)
        baseline=chain.snapshot()
        detached=chain.snapshot(); detached["chains"][0]["spins"][0]*=-1
        self.assertEqual(chain.snapshot(),baseline)
        changes=[("version",2),("fields",np.zeros(18).tolist()),("sweeps",18),
                 ("exchange_rounds",5),("chains",[]),("walker_labels",[0,0,2,3]),
                 ("swap_attempts",[0,0,0]),("swap_accepts",[99,99,99]),
                 ("visits",np.zeros((4,4),int).tolist()),("roundtrips",[999]*4),
                 ("roundtrip_phase",[256]*4),("swap_rng_state",{}),
                 ("swap_interval_sweeps",7)]
        for key,value in changes:
            damaged=deepcopy(baseline); damaged[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                ReplicaExchangeChain.from_snapshot(self.lattice,self.fields,damaged,backend="python")
        damaged=deepcopy(baseline); damaged["chains"][0]["spins"][0]*=-1
        with self.assertRaisesRegex(ValueError,"reference star orbit"):
            ReplicaExchangeChain.from_snapshot(self.lattice,self.fields,damaged,backend="python")


if __name__ == "__main__":
    unittest.main()
