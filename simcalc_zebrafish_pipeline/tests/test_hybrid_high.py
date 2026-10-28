from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from hybrid_high_model import deform_template,simulate_hybrid_high
from prepare_hybrid_high_template import extract_geometry


def fixture():
    h,w=110,140;yy,xx=np.mgrid[:h,:w]
    head=(yy>=8)&(yy<=101)
    eyes=((yy-28)**2+(xx-24)**2<12**2)|((yy-81)**2+(xx-24)**2<12**2)
    channels=((abs(xx-66)<=2)&head)|((abs(yy-54)<=1)&(xx>=66)&head)
    # A one-pixel branch is joined to the main dark channel. It must survive
    # extraction without being specified as an edge/landmark to the algorithm.
    channels|=(yy==76)&(xx>=66)&(xx<=96)
    image=np.full((h,w),100.,dtype=float)
    image[head&~eyes&~channels]=2100.
    settings={'eye_opening_iterations':2,'eye_min_area_px':100,'eye_min_radius_px':7.}
    arrays,stats,_=extract_geometry(image,settings)
    return arrays,stats


class HybridHighTests(unittest.TestCase):
    def test_raster_extraction_preserves_unprescribed_small_branch(self):
        arrays,stats=fixture()
        self.assertEqual(len(stats['eye_components']),2)
        self.assertTrue(arrays['eye_mask'][28,24])
        self.assertTrue(arrays['eye_mask'][81,24])
        self.assertGreater(np.mean(arrays['channel_mask'][76,71:94]),.8)
        self.assertTrue(arrays['channel_mask'][54,100])
        self.assertFalse(arrays['tissue_mask'][76,84])
        self.assertNotIn('mean',arrays)
        self.assertEqual(arrays['emission_grid'].shape,(17,21))
        self.assertEqual(arrays['density_grid'].shape,(13,17))

    def test_zero_warp_preserves_geometry_and_new_seed_changes_cells(self):
        template,_=fixture()
        deformed=deform_template(template,42,{'strength':0.})
        for key in ('head_mask','eye_mask','channel_mask','tissue_mask'):
            np.testing.assert_array_equal(deformed[key],template[key])
        profiles={'percentiles':np.array([0.,50.,100.]),
                  'highSNR_quantiles':np.array([[100.,1800.,5000.]])}
        cells={'density_per_10000_pixels':250.,'sigma_px':.8}
        first=simulate_hybrid_high(template,profiles,42,cells,{'strength':0.})
        second=simulate_hybrid_high(template,profiles,43,cells,{'strength':0.})
        np.testing.assert_array_equal(first['tissue_mask'],second['tissue_mask'])
        self.assertFalse(np.array_equal(first['centres_yx'],second['centres_yx']))
        self.assertFalse(np.array_equal(first['highSNR'],second['highSNR']))

    def test_high_only_no_file_io_and_masks_follow_all_template_fields(self):
        template,_=fixture()
        profiles={'percentiles':np.array([0.,50.,100.]),
                  'highSNR_quantiles':np.array([[100.,1800.,5000.]])}
        with patch('builtins.open',side_effect=AssertionError('Unexpected image read')), \
             patch.object(Path,'read_text',side_effect=AssertionError('Unexpected config read')), \
             patch.object(np,'load',side_effect=AssertionError('Unexpected template read')) :
            a=simulate_hybrid_high(template,profiles,42,{'density_per_10000_pixels':250.})
            b=simulate_hybrid_high(template,profiles,42,{'density_per_10000_pixels':250.})
        for key in ('highSNR','labels','head_mask','eye_mask','channel_mask','tissue'):
            np.testing.assert_array_equal(a[key],b[key])
            self.assertTrue(np.isfinite(a[key]).all())
        self.assertFalse(any('lowSNR' in key for key in a))
        forbidden=(~a['head_mask'])|a['eye_mask']|a['channel_mask']
        self.assertFalse(np.any((a['labels']>0)&forbidden))
        self.assertFalse(np.any((a['labels']>0)&(a['tissue']<.5)))
        self.assertGreaterEqual(a['warp_metadata']['inverse_jacobian_min'],.45)
        with self.assertRaises(ValueError):
            simulate_hybrid_high({**template,'mean_image':np.ones((110,140))},profiles,42)


if __name__=='__main__':
    unittest.main()
