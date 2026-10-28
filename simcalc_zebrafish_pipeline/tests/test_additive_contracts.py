import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset_io import ConstantFrames,prepare_main,load_prepared
from diffusion_sampling import cosine_schedule,q_sample,ddim_step,sdedit
from diffusion_unet import DiffusionUNet
from simcalc_addon import compose,mask_qa,export_main
from simcalc_data import load_simcalc,load_domains
from train_simcalc_diffusion import FORMAT,session_end_step,read_checkpoint


ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class Contracts(unittest.TestCase):
    def setUp(self):
        temp_root=ROOT/'work'/'additive_tests'
        temp_root.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='test_',dir=temp_root)
        self.folder=Path(self.temporary.name).resolve()
        assert self.folder.is_relative_to(temp_root.resolve())
    def tearDown(self):
        self.temporary.cleanup()

    def prepared(self,name='synthetic',role='synthetic',count=3,labels=False):
        images=(np.arange(count*13*17).reshape(count,13,17)%1000+18000).astype(np.uint16)
        source=self.folder/(name+'.npz')
        arrays=dict(images=images,metadata_json=np.asarray('{}'))
        if labels:
            lab=np.zeros_like(images,dtype=np.uint64)
            lab[:,6,8]=2**40+7
            arrays['labels']=lab
        np.savez_compressed(source,**arrays)
        out=self.folder/name
        prepare_main(['--input',str(source),'--output',str(out),'--role',role,'--intensity','adu'])
        return out,images

    def test_schedule_oracle_and_identity(self):
        alphas=cosine_schedule()
        self.assertTrue(torch.all(alphas[1:]<alphas[:-1]))
        x0=torch.linspace(-.4,.4,32*40).reshape(1,1,32,40)
        noise=torch.full_like(x0,.2)
        xt=q_sample(x0,350,noise,alphas)
        actual=ddim_step(xt,noise,350,100,alphas)
        torch.testing.assert_close(actual,q_sample(x0,100,noise,alphas),atol=1e-6,rtol=1e-5)
        self.assertTrue(torch.equal(sdedit(None,x0,alphas,strength=0),x0))

    def test_architecture_and_native_geometry(self):
        default=DiffusionUNet()
        self.assertEqual(sum(p.numel() for p in default.parameters()),3762241)
        model=DiffusionUNet(channels=8,attention=False).eval()
        with torch.no_grad():
            output=model(torch.zeros(1,1,400,485),torch.tensor([200]))
        self.assertEqual(output.shape,(1,1,400,485))

    def test_residual_identity_and_mask_coefficient(self):
        raw=np.full((8,8),20000,dtype=np.uint16)
        source=torch.zeros(1,1,8,8)
        edited=source+.2
        norm=dict(black=15000,white=25000)
        labels=np.ones((8,8),dtype=np.uint16)
        output,residual=compose(raw,source,edited,labels,norm)
        self.assertTrue(np.all(residual==600))
        self.assertTrue(np.array_equal(raw.astype(np.int32)+residual,output.astype(np.int32)))
        extreme=np.array([[0,65535],[32767,123]],dtype=np.uint16)
        same=torch.zeros(1,1,2,2)
        output,residual=compose(extreme,same,same,np.zeros((2,2),dtype=np.uint16),norm)
        self.assertTrue(np.array_equal(output,extreme))
        self.assertFalse(residual.any())

    def test_sparse_instance_ids_and_missing_masks(self):
        raw=np.full((13,17),20000,dtype=np.uint16)
        raw[6,8]=22000
        labels=np.zeros(raw.shape,dtype=np.uint64)
        labels[6,8]=2**40+7
        result=mask_qa(raw,raw,labels,np.ones(raw.shape,dtype=bool),dict(black=15000,white=25000))
        self.assertEqual(result['instance_count'],1)
        self.assertEqual(result['median_instance_contrast_relative_change'],0)
        self.assertFalse(result['review_required'])
        missing=mask_qa(raw,raw,labels,np.ones(raw.shape,dtype=bool),dict(black=15000,white=25000),labels_available=False)
        self.assertIsNone(missing['instance_count'])
        self.assertTrue(missing['review_required'])

    def test_preparation_calibration_and_checksum(self):
        image=np.linspace(0,1,2*13*17,dtype=np.float32).reshape(2,13,17)
        source=self.folder/'unit.npy'
        np.save(source,image)
        out=self.folder/'prepared'
        prepare_main(['--input',str(source),'--output',str(out),'--intensity','unit',
                      '--adu-black','17498','--adu-white','32767'])
        data=load_simcalc(out,self.folder/'cache')
        self.assertEqual(data['synthetic_images'].shape,(2,13,17))
        self.assertEqual(int(data['synthetic_images'][0,0,0]),17498)
        self.assertEqual(int(data['synthetic_images'][-1,-1,-1]),32767)
        self.assertFalse(data['labels_available'])
        with (out/'synthetic_images.npy').open('r+b') as f:
            f.seek(-1,2)
            last=f.read(1)
            f.seek(-1,2)
            f.write(bytes([last[0]^1]))
        with self.assertRaises(ValueError):
            load_prepared(out)

    def test_instance_ids_preserved_by_preparation(self):
        out,_=self.prepared(labels=True)
        data=load_simcalc(out,self.folder/'cache')
        self.assertEqual(int(data['synthetic_labels'][0,6,8]),2**40+7)
        self.assertTrue(data['labels_available'])

    def test_custom_real_loader_and_shape_mismatch(self):
        synthetic,_=self.prepared()
        real,_=self.prepared('real',role='real',count=30)
        args=SimpleNamespace(synthetic_npz=str(synthetic),synthetic_cache=str(self.folder/'cache'),
              real_input=str(real),temporal_gap=0,norm_samples=10,real_split_file=None)
        real_data,synth=load_domains(args)
        self.assertEqual(real_data['images'].shape,(30,13,17))
        self.assertEqual(int(np.sum(real_data['split']==1)),3)
        self.assertEqual(synth['images'].shape,(3,13,17))
        bad=self.folder/'wrong_geometry.npy'
        np.save(bad,np.zeros((30,14,17),dtype=np.uint16))
        args.real_input=str(bad)
        with self.assertRaises(ValueError):
            load_domains(args)

    def test_real_group_leakage_is_rejected(self):
        synthetic,_=self.prepared()
        real,_=self.prepared('real_groups',role='real',count=30)
        # Supply explicit split + overlapping recording groups through preparation.
        source=self.folder/'real_groups.npz'
        split=np.r_[np.zeros(20),np.ones(5),np.full(5,2)].astype(np.int8)
        split_path=self.folder/'split.npy'
        groups_path=self.folder/'groups.npy'
        np.save(split_path,split);np.save(groups_path,np.zeros(30,dtype=np.int64))
        grouped=self.folder/'grouped'
        prepare_main(['--input',str(source),'--output',str(grouped),'--role','real','--intensity','adu',
                      '--split',str(split_path),'--groups',str(groups_path)])
        args=SimpleNamespace(synthetic_npz=str(synthetic),synthetic_cache=str(self.folder/'cache'),
            real_input=str(grouped),temporal_gap=0,norm_samples=10,real_split_file=None)
        with self.assertRaises(ValueError):
            load_domains(args)


    def test_export_identity_new_dataset_and_masks_absence(self):
        synthetic,images=self.prepared()
        model=DiffusionUNet(channels=8,attention=False)
        checkpoint=self.folder/'identity.pt'
        torch.save(dict(format=FORMAT,step=0,model_config=model.config,ema=model.state_dict(),
            args=dict(timesteps=1000),domain_metadata=dict(normalization=dict(black=15000,white=30000)),
            data_sha256=dict(synthetic_npz_sha256='another-input')),checkpoint)
        out=self.folder/'export'
        export_main(['--checkpoint',str(checkpoint),'--synthetic-npz',str(synthetic),'--output',str(out),
                     '--strength','0','--limit','2','--device','cpu','--allow-new-synthetic'])
        shard=next(out.glob('addon_*.npz'))
        with np.load(shard,allow_pickle=False) as z:
            self.assertTrue(np.array_equal(z['refined_images'],images[:2]))
            self.assertFalse(z['additive_residual_adu'].any())
            self.assertNotIn('synthetic_labels',z.files)
            self.assertTrue(z['review_required'].all())
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['labels_available'])
        self.assertIn('No instance masks',manifest['mask_status'])
        self.assertTrue(json.loads((out/'export_progress.json').read_text(encoding='utf-8'))['complete'])
        self.assertEqual(session_end_step(4000,10000,1000),5000)

    def test_legacy_allowed_only_for_initialization(self):
        path=self.folder/'legacy.pt'
        torch.save(dict(format='zebrafish-real-ddpm-sdedit-v1',step=2000),path)
        with self.assertRaises(ValueError):
            read_checkpoint(path)
        self.assertEqual(read_checkpoint(path,allow_legacy=True)['step'],2000)


if __name__=='__main__':
    unittest.main()
