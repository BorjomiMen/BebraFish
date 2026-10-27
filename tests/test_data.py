import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from zebrafish_diffusion.data import prepare_diffusion,load_prepared,prepare_segmentation,segmentation_records

class DataTests(unittest.TestCase):
    def test_folder_normalization_and_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'source'; source.mkdir()
            for i in range(10):
                a=np.arange(32*40,dtype=np.uint16).reshape(32,40)+i*100
                Image.fromarray(a).save(source/f'{i}.png')
            out=prepare_diffusion(source,root/'prepared')
            data=load_prepared(out)
            self.assertEqual(data['images'].shape,(10,32,40))
            self.assertEqual(data['images'][0].dtype,np.uint16)
            self.assertEqual(set(data['split']),{0,1,2})
            data['images'].close()
            edited=np.load(out/'images.npy',mmap_mode='r+')
            edited[0,0,0]+=1; edited.flush(); del edited
            with self.assertRaises(ValueError):
                load_prepared(out)

    def test_simcalc_instance_to_binary_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            images=np.arange(6*24*32,dtype=np.uint16).reshape(6,24,32)
            labels=np.zeros_like(images); labels[:,4:10,5:15]=8; labels[:,12:17,16:20]=43
            np.savez(root/'simcalc.npz',synthetic_images=images,synthetic_labels=labels,
                     synthetic_seeds=np.arange(6,dtype=np.uint32))
            out=prepare_segmentation(root/'simcalc.npz',root/'seg')
            records,report=segmentation_records(out)
            self.assertEqual(report['pairs'],6)
            self.assertEqual(set(np.unique(np.array(Image.open(records[0]['mask'])))),{0,255})
            self.assertEqual(np.count_nonzero(np.array(Image.open(records[0]['mask']))),np.count_nonzero(labels[0]))

    def test_float_npy_temporal_split(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            frames=np.random.default_rng(5).normal(size=(100,24,32)).astype(np.float32)
            np.save(root/'frames.npy',frames)
            out=prepare_diffusion(root/'frames.npy',root/'prepared',gap=1)
            data=load_prepared(out)
            self.assertEqual(data['images'].raw.dtype,np.float32)
            self.assertTrue(np.any(data['split']==-1))
            self.assertTrue(all(np.any(data['split']==code) for code in (0,1,2)))
            data['images'].close()

if __name__=='__main__':
    unittest.main()
