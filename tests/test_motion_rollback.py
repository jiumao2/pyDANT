"""Regression checks for retained motion, corrected waveforms and final clustering.

Run with: python -m unittest discover -s tests -p test_motion_rollback.py
All synthetic inputs and outputs are confined to temporary directories.
"""
import contextlib
import copy
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

motion_module = importlib.import_module('pyDANT.MotionEstimation')
clustering_module = importlib.import_module('pyDANT.IterativeClustering')
from pyDANT.utils import Motion


class TestMotionRollback(unittest.TestCase):
    def check_run(self, counts, retained_fit, linear=False, initial_kind='zero', stop=True):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'Figures').mkdir()
            sessions = np.tile([1, 2], 4)
            locations = np.column_stack((np.zeros(8), [100., 120., 200., 222., 300., 340., 400., 441.]))
            waveforms = np.random.default_rng(5).normal(size=(8, 8, 12))
            channels = np.column_stack((np.zeros(8), np.arange(8) * 80.))
            for name, values in [('session_index', sessions), ('locations', locations),
                                 ('waveform_all', waveforms), ('channel_locations', channels)]:
                np.save(root / (name + '.npy'), values)
            settings = dict(path_to_data=folder, output_folder=folder, n_jobs=1,
                            centering_waveforms=False,
                            motionEstimation=dict(features=[['Waveform']], max_iter=len(counts),
                                                  repeat_last_feature_set=True, compute_ci=False),
                            waveformCorrection=dict(path_to_motion='', linear_correction=linear, n_templates=2),
                            clustering=dict(features=['Waveform'], max_distance=15))
            if stop is not None:
                settings['motionEstimation']['stop_early'] = stop
            initial = Motion(num_sessions=2)
            if initial_kind != 'zero':
                initial.Constant = np.array([-3., 3.])
            if initial_kind == 'manual':
                np.save(root / 'manual.npy', initial.Constant)
                settings['waveformCorrection']['path_to_motion'] = str(root / 'manual.npy')
            shared = initial if initial_kind == 'shared' else None
            pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
            fitted = []
            round_number = 0

            def cluster(*args):
                nonlocal round_number
                selected = pairs[:counts[round_number]]
                round_number += 1
                matrix = np.eye(8, dtype=bool)
                matrix[selected[:, 0], selected[:, 1]] = True
                matrix[selected[:, 1], selected[:, 0]] = True
                for name, values in [('ClusterMatrix', matrix), ('SimilarityMatrix', matrix.astype(float)),
                                     ('SimilarityPairs', selected), ('SimilarityThreshold', np.array(.5)),
                                     ('SimilarityWeights', np.array([1.]))]:
                    np.save(root / (name + '.npy'), values)

            compute_motion = motion_module.computeMotion

            def fit(settings):
                result = compute_motion(settings)
                fitted.append(copy.deepcopy(result))
                return result

            random_state = np.random.get_state()
            try:
                np.random.seed(42)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                        patch.object(motion_module, 'iterativeClustering', side_effect=cluster), \
                        patch.object(motion_module, 'computeMotion', side_effect=fit):
                    retained, history = motion_module.motionEstimation(settings, initial_motion=shared)
                    saved_waveforms = np.load(root / 'waveforms_corrected.npy')
                    expected_waveforms = motion_module.computeWaveformFeatures(settings, waveforms, retained)
                expected = initial if retained_fit == 0 else fitted[retained_fit - 1]
                disk = Motion.load(folder)
                for name in ['Linear', 'Constant', 'LinearScale']:
                    np.testing.assert_array_equal(getattr(retained, name), getattr(expected, name))
                    np.testing.assert_array_equal(getattr(disk, name), getattr(retained, name))
                np.testing.assert_array_equal(saved_waveforms, expected_waveforms)
                self.assertEqual(len(history), retained_fit)
                with np.load(root / 'MotionIterations.npz', allow_pickle=True) as saved:
                    self.assertEqual(len(saved['result_iter']), retained_fit)
                    if retained_fit:
                        np.testing.assert_array_equal(saved['result_iter'][-1]['motion'].Constant, retained.Constant)
                # Exercise the real finalClustering loader, intercepting only HDBSCAN/LDA.
                with patch.object(clustering_module, 'iterativeClustering') as final_cluster:
                    clustering_module.finalClustering(settings)
                loaded_waveforms, loaded_motion = final_cluster.call_args.args[2:4]
                np.testing.assert_array_equal(loaded_waveforms, expected_waveforms)
                for name in ['Linear', 'Constant', 'LinearScale']:
                    np.testing.assert_array_equal(getattr(loaded_motion, name), getattr(retained, name))
                expected_pairs = clustering_module.getNearbyPairs(15, sessions, locations, retained)[0]
                actual_pairs = clustering_module.getNearbyPairs(15, sessions, locations, loaded_motion)[0]
                np.testing.assert_array_equal(actual_pairs, expected_pairs)
                if retained_fit == 0 and initial_kind == 'zero' and not linear:
                    rejected_pairs = clustering_module.getNearbyPairs(15, sessions, locations, fitted[-1])[0]
                    self.assertFalse(np.array_equal(expected_pairs, rejected_pairs))
            finally:
                np.random.set_state(random_state)
                motion_module.plt.close('all')

    def test_early_and_late_rollback(self):
        for linear in [False, True]:
            for counts, retained_fit in [([2, 2], 0), ([2, 1], 0), ([2, 3, 3], 1), ([2, 3, 2], 1)]:
                with self.subTest(linear=linear, counts=counts):
                    self.check_run(counts, retained_fit, linear=linear)

    def test_manual_and_shared_initialization(self):
        for initial_kind in ['manual', 'shared']:
            with self.subTest(initial_kind=initial_kind):
                self.check_run([2, 2], 0, initial_kind=initial_kind)

    def test_no_rollback(self):
        for stop in [False, None]:
            with self.subTest(stop=stop):
                self.check_run([2, 2], 2, stop=stop)
        self.check_run([2, 3], 2)


if __name__ == '__main__':
    unittest.main()
