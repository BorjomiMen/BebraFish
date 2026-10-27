from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from zebrafish_diffusion.checkpoints import latest_diffusion

class CheckpointSelectionTests(unittest.TestCase):
    def test_only_7500_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            expected=Path(folder)/'step_007500.pt'
            expected.touch()
            self.assertEqual(latest_diffusion(folder),expected)

    def test_numeric_step_and_empty_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(latest_diffusion(folder))
            for name in ('step_7500.pt','step_10000.pt','step_backup.pt'):
                (Path(folder)/name).touch()
            self.assertEqual(latest_diffusion(folder).name,'step_10000.pt')

if __name__=='__main__':
    unittest.main()
