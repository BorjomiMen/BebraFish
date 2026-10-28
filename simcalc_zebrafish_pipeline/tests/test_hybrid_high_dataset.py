from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import generate_hybrid_high_dataset as exporter
from hybrid_high_model import simulate_hybrid_high
from prepare_hybrid_high_template import SEGMENTATION_PARAMETERS, sha256
from test_hybrid_high import fixture


def save_fixture(path):
    template, _ = fixture()
    # Small synthetic file fixture for the prepared-template serialization contract.
    metadata = {
        'model': 'Synthetic test fixture, not a real recording',
        'preparation_code_sha256': sha256(exporter.ROOT/'prepare_hybrid_high_template.py'),
        'statistics': {'floor': 100., 'segmentation_parameters': SEGMENTATION_PARAMETERS},
        'training_source_indices': [123],
    }
    profiles = {'percentiles': np.array([0., 50., 100.]),
                'highSNR_quantiles': np.array([[100., 1800., 5000.]])}
    np.savez_compressed(path, **template, **profiles, metadata_json=np.asarray(json.dumps(metadata)))
    return template, profiles


class HybridDatasetTests(unittest.TestCase):
    def settings(self, root):
        return {'template_path': root/'template.npz', 'output_dir': root/'output',
                'image_count': 3, 'random_seed': 42, 'checkpoint_size': 1,
                'cell_parameters': {'density_per_10000_pixels': 250.}}

    def test_streamed_npz_matches_generator_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory); template, profiles = save_fixture(root/'template.npz')
            settings = self.settings(root)
            with patch.object(exporter, 'save_previews'):
                path = exporter.export_dataset(**settings)
            with np.load(path, allow_pickle=False) as z:
                expected = simulate_hybrid_high(template, profiles, 42, settings['cell_parameters'])
                np.testing.assert_array_equal(z['synthetic_images'][0], np.rint(expected['highSNR']).astype(np.uint16))
                np.testing.assert_array_equal(z['synthetic_labels'][0], expected['labels'])
                np.testing.assert_array_equal(z['synthetic_seeds'], [42, 43, 44])
                self.assertEqual(z['synthetic_images'].shape, (3, 110, 140))
                self.assertEqual(z['synthetic_images'].dtype, np.uint16)
                self.assertFalse(any('lowSNR' in key or key.startswith('real_') for key in z.files))
                self.assertFalse(np.any((z['synthetic_labels']>0)&~z['synthetic_tissue_masks']))
                metadata = json.loads(z['metadata_json'].item())
                self.assertEqual(metadata['warp_parameters']['elastic_rms_px'], 3.)
                self.assertEqual(metadata['image_count'], 3)
            before = sha256(path)
            with patch.object(exporter, 'simulate_hybrid_high', side_effect=AssertionError('Completed run must not regenerate')):
                self.assertEqual(exporter.export_dataset(**settings), path)
            self.assertEqual(sha256(path), before)
            with self.assertRaises(ValueError):
                exporter.export_dataset(**(settings | {'random_seed': 50}))
            self.assertEqual(sha256(path), before)

    def test_interrupted_run_reuses_committed_samples(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory); save_fixture(root/'template.npz')
            settings = self.settings(root); calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('Simulated interruption')
                return simulate_hybrid_high(*args, **kwargs)
            with patch.object(exporter, 'simulate_hybrid_high', side_effect=interrupted), \
                 self.assertRaisesRegex(RuntimeError, 'Simulated interruption'):
                exporter.export_dataset(**settings)
            checkpoint = root/'output/_parts/part_000000_000001.npz'
            before = sha256(checkpoint)
            self.assertFalse((root/'output/hybrid_highSNR_3.npz').exists())
            with patch.object(exporter, 'simulate_hybrid_high', wraps=simulate_hybrid_high) as render, \
                 patch.object(exporter, 'save_previews'):
                path = exporter.export_dataset(**settings)
                self.assertEqual(render.call_count, 2)
            self.assertEqual(sha256(checkpoint), before)
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['synthetic_seeds'], [42, 43, 44])
            state = json.loads((root/'output/progress.json').read_text())
            self.assertEqual(state['phase'], 'complete')
            self.assertEqual(state['completed_images'], 3)


if __name__ == '__main__':
    unittest.main()
