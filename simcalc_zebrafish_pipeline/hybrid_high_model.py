"""New SimCalc neurons inside a jointly deformed highSNR raster template.

HighSNR-only renderer. The training mean is never an input or a background
image. Transferred raster masks make this explicitly hybrid synthesis.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from simcalc_core import SimCalc, _build_scene, numpy_seed, sample_head_centres, inverse_warp_diagnostics

CELL_DEFAULTS={
    'density_per_10000_pixels':1100., 'sigma_px':.9,'size_cv':.15,
    'eccentricity':.35,'min_spacing_px':1.4,'amplitude_cv':.3,
    'ring_fraction':0.,'blur_px':.45,'background':.2,
    'background_texture':.3,'neuropil_filters':12,
}
WARP_DEFAULTS={'strength':1.,'elastic_rms_px':3.,'rotation_sd_degrees':.8,
               'scale_x_sd':.02,'scale_y_sd':.025,'translation_sd_px':2.}
DETECTOR_DEFAULTS={'photons':500.,'read_noise':.1,'dark_read_noise':.003}
TEMPLATE_KEYS=('head_mask','eye_mask','channel_mask','tissue_mask','tissue_field',
               'rim_mask','rim_field','emission_grid','density_grid','rim_strength_grid')


def _settings(defaults,overrides):
    overrides=overrides or {}
    if set(overrides)-set(defaults):
        raise ValueError(f'Unsupported numeric settings: {set(overrides)-set(defaults)}')
    p=deepcopy(defaults)|overrides
    if any(not isinstance(v,(int,float,np.number)) or not np.isfinite(v) or v<0 for v in p.values()):
        raise ValueError('Settings must be finite nonnegative numbers')
    return p


def _grid_field(grid,coordinates,shape):
    return map_coordinates(np.asarray(grid),
        [coordinates[0]/(shape[0]-1)*(len(grid)-1),
         coordinates[1]/(shape[1]-1)*(np.asarray(grid).shape[1]-1)],
        order=1,mode='nearest').astype(np.float32)


def deform_template(template,seed,overrides=None):
    w=_settings(WARP_DEFAULTS,overrides)
    if w['strength']>1.5:
        raise ValueError('Warp strength must be <=1.5')
    h,width=template['head_mask'].shape
    shape=(h,width);yy,xx=np.mgrid[:h,:width].astype(float)
    pixel_scale=min(h/400,width/485)  # Pixel settings refer to the native recording.
    rng=np.random.default_rng(seed+8101)
    grid_shape=(49,61)
    gy,gx=np.meshgrid(np.linspace(0,h-1,grid_shape[0]),np.linspace(0,width-1,grid_shape[1]),indexing='ij')
    for attempt in range(30):
        strength=w['strength']
        theta=float(np.clip(rng.normal(),-2,2)*w['rotation_sd_degrees']*strength)
        sy,sx=np.exp(np.clip(rng.normal(size=2),-2,2)*np.array([w['scale_y_sd'],w['scale_x_sd']])*strength)
        dy,dx=np.clip(rng.normal(size=2),-2,2)*w['translation_sd_px']*strength*pixel_scale
        angle=np.deg2rad(theta);py=gy-(h-1)/2-dy;px=gx-(width-1)/2-dx
        inverse=np.stack(((h-1)/2+(np.cos(angle)*py-np.sin(angle)*px)/sy,
                          (width-1)/2+(np.sin(angle)*py+np.cos(angle)*px)/sx))
        for axis in range(2):
            coarse=rng.normal(size=(6,8));coarse-=coarse.mean();coarse/=max(coarse.std(),1e-8)
            cy,cx=np.meshgrid(np.linspace(0,5,grid_shape[0]),np.linspace(0,7,grid_shape[1]),indexing='ij')
            field=gaussian_filter(map_coordinates(coarse,[cy,cx],order=3,mode='nearest'),.8)
            field-=field.mean();field/=max(field.std(),1e-8)
            inverse[axis]+=field*w['elastic_rms_px']*strength*pixel_scale
        displacement=inverse-np.stack((gy,gx))
        lo,hi=inverse_warp_diagnostics(displacement,(h-1,width-1))
        if lo>=.45 and hi<=2.2:break
    else:
        raise ValueError('No nonfolding template deformation found; reduce strength')
    if not strength:
        coordinates=np.stack((yy,xx))
    else:
        cy,cx=yy/(h-1)*(grid_shape[0]-1),xx/(width-1)*(grid_shape[1]-1)
        coordinates=np.stack((yy,xx))+np.stack([
            map_coordinates(field,[cy,cx],order=1,mode='nearest') for field in displacement])
    def warp(key,order=1):
        return map_coordinates(template[key].astype(float),coordinates,order=order,mode='constant',cval=0.)
    head=warp('head_mask',0)>.5;eyes=warp('eye_mask',0)>.5;channels=warp('channel_mask',0)>.5
    valid=head&~eyes&~channels
    tissue=warp('tissue_field');tissue[~valid]=0
    rim=warp('rim_field')*head*(~eyes)
    emission=_grid_field(template['emission_grid'],coordinates,shape)
    density=_grid_field(template['density_grid'],coordinates,shape)
    rim_strength=_grid_field(template['rim_strength_grid'],coordinates,shape)
    return {'head_mask':head,'eye_mask':eyes,'channel_mask':channels,'tissue_mask':valid,
            'tissue':tissue.astype(np.float32),'rim':rim.astype(np.float32),
            'emission_gain':emission,'density_weights':density,
            'surface':(rim*rim_strength).astype(np.float32),
            'inverse_coordinates_yx':coordinates.astype(np.float32),
            'warp_metadata':{'seed':int(seed),'settings':w,'rotation_degrees':theta,
                             'scale_yx':[float(sy),float(sx)],'translation_yx_px':[float(dy),float(dx)],
                             'attempts':attempt+1,'inverse_jacobian_min':lo,'inverse_jacobian_max':hi}}


def simulate_hybrid_high(template,profiles,seed,cell_parameters=None,warp_parameters=None,detector_parameters=None):
    if set(template)!=set(TEMPLATE_KEYS):
        raise ValueError('Only the declared raster geometry/coarse template fields are accepted')
    p=_settings(CELL_DEFAULTS,cell_parameters)
    if p['sigma_px']<=0 or p['min_spacing_px']<=0 or p['eccentricity']>=1 or p['ring_fraction']>1:
        raise ValueError('Invalid kernel size, spacing or shape')
    if int(p['neuropil_filters'])!=p['neuropil_filters']:
        raise ValueError('neuropil_filters must be an integer')
    detector=_settings(DETECTOR_DEFAULTS,detector_parameters)
    if detector['photons']<=0:
        raise ValueError('photons must be positive')
    fields=deform_template(template,seed,warp_parameters)
    shape=fields['tissue'].shape;p['shape_yx']=list(shape)
    support=distance_transform_edt(fields['tissue']>=.6)>max(1.5,1.3*p['sigma_px'])
    count=int(round(support.sum()*p['density_per_10000_pixels']/10000))
    if not 0<count<65535:
        raise ValueError('Need between 1 and 65534 cells; adjust tissue/density')
    centres=sample_head_centres(support,count,p['min_spacing_px'],np.random.default_rng(seed+8201),fields['density_weights'])
    scene=_build_scene(p,seed,centres=centres,tissue=fields['tissue'],
                       emission_gain=fields['emission_gain'],surface=fields['surface'])
    h,width=shape;side=max(shape);photons=detector['photons']
    summed=np.zeros((side,side));summed[:h,:width]=scene['signal']
    model=SimCalc(h=side,N=1,T=1,StimCycles=1,dt=1)
    model.spat_kernels=[summed];model.F=np.full((1,2),photons)
    model.npil=np.broadcast_to(scene['background_square']*photons,(2,side,side)).copy()
    with numpy_seed(seed+8301),redirect_stdout(io.StringIO()):
        model.gen_data()
    raw=model.video[0,:h,:width]/photons
    rng=np.random.default_rng(seed+8401)
    noise=detector['dark_read_noise']+detector['read_noise']*fields['tissue']
    raw+=rng.normal(size=shape)*noise
    index=int(np.random.default_rng(seed+8501).integers(len(profiles['highSNR_quantiles'])))
    target=profiles['highSNR_quantiles'][index]
    source,indices=np.unique(np.percentile(raw,profiles['percentiles']),return_index=True)
    if len(source)<2:
        raise ValueError('Degenerate synthetic intensity distribution')
    mapped=np.interp(raw,source,target[indices])
    scene.update(fields)
    scene.update(highSNR=np.clip(mapped,0,32767).astype(np.float32),raw_highSNR=raw.astype(np.float32),
                 profile_index=index,detector_parameters=detector,cell_parameters=p)
    return scene
