"""Extract a raster tissue template from training highSNR frames only.

No eye ellipses, branch guides, hand-picked landmarks or v5 geometry enter
extraction. Masks are intensity-based proxies, not expert anatomical labels.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (binary_closing, binary_fill_holes, binary_opening,
    binary_propagation, distance_transform_edt, gaussian_filter, gaussian_filter1d,
    label, map_coordinates, maximum_filter, median_filter)
import tifffile

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/'data/zenodo_6293696'
TEMPLATE_PATH=ROOT/'configs/hybrid_highSNR_template_v1.npz'
DIAGNOSTIC_DIR=ROOT/'reports/hybrid_highSNR_template_v1'
SEGMENTATION_PARAMETERS={
    'smooth_sigma_px':.9, 'head_threshold_above_floor':400.,
    'eye_threshold_above_floor':350., 'eye_opening_iterations':4,
    'eye_min_area_px':400, 'eye_min_radius_px':11.,
    'channel_strong_ratio':.35, 'channel_weak_ratio':.66,
    'channel_min_component_area_px':15, 'channel_background_sigma_px':8.,
    'channel_junction_radius_px':8.,
    'tissue_edge_sigma_px':.6,
}
PERCENTILES=np.unique(np.r_[0,.1,.5,1,2,5,np.arange(10,100,5),98,99,99.5,99.9,100])


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _remove_small(mask,area):
    components,_=label(mask)
    counts=np.bincount(components.ravel())
    return (components>0)&(counts[components]>=area)


def extract_geometry(mean_high,parameters=None):
    p={**SEGMENTATION_PARAMETERS,**(parameters or {})}
    image=np.asarray(mean_high,dtype=float)
    if image.ndim!=2 or min(image.shape)<32 or not np.isfinite(image).all():
        raise ValueError('Expected a finite 2D highSNR mean, dimensions >=32')
    floor=float(np.percentile(image,1))
    signal=np.maximum(image-floor,0)
    smooth=gaussian_filter(signal,p['smooth_sigma_px'])
    # The recording has a single head intersecting each image column. Keep
    # its full-resolution vertical envelope, including the cropped eye areas.
    support=binary_closing(np.pad(smooth>p['head_threshold_above_floor'],4,mode='edge'),iterations=2)[4:-4,4:-4]
    support=_remove_small(support,50)
    h,w=image.shape
    top,bottom=np.full(w,np.nan),np.full(w,np.nan)
    for x in range(w):
        rows=np.flatnonzero(support[:,x])
        if len(rows):top[x],bottom[x]=rows[0],rows[-1]
    valid=np.isfinite(top)
    if valid.sum()<w*.5:
        raise ValueError('Insufficient outer-head coverage; inspect segmentation settings')
    xs=np.arange(w)
    top=gaussian_filter1d(median_filter(np.interp(xs,xs[valid],top[valid]),size=5),.7)
    bottom=gaussian_filter1d(median_filter(np.interp(xs,xs[valid],bottom[valid]),size=5),.7)
    yy,xx=np.mgrid[:h,:w]
    head=(yy>=top[None])&(yy<=bottom[None])
    absolute=(smooth<p['eye_threshold_above_floor'])&head
    opened=binary_opening(absolute,iterations=p['eye_opening_iterations'])
    components,count=label(opened)
    seeds=np.zeros_like(head)
    eye_components=[]
    for i in range(1,count+1):
        part=components==i
        area=int(part.sum());radius=float(distance_transform_edt(part).max())
        if area>=p['eye_min_area_px'] and radius>=p['eye_min_radius_px']:
            seeds|=part
            eye_components.append({'area_px':area,'inscribed_radius_px':radius})
    if not eye_components:
        raise ValueError('No large compact dark eye regions found; inspect the template diagnostic')
    eyes=absolute&(distance_transform_edt(~seeds)<p['eye_opening_iterations']+1)
    eyes=binary_fill_holes(binary_closing(np.pad(eyes,3),iterations=2)[3:-3,3:-3])&head
    valid=head&~eyes
    sigma=p['channel_background_sigma_px']
    denominator=gaussian_filter(valid.astype(float),sigma)
    local=gaussian_filter(smooth*valid,sigma)/np.maximum(denominator,1e-6)
    ratio=smooth/np.maximum(local,300.)
    margin=distance_transform_edt(head)
    strong=(ratio<p['channel_strong_ratio'])&valid&(margin>2)
    # A wide dark trunk depresses its local mean, which can disconnect a thin
    # branch at its root. Estimate the nearby bright envelope only in a small
    # neighbourhood of a measured strong valley; do not invent a graph edge.
    envelope=gaussian_filter(maximum_filter(smooth,size=9),2.)
    near_trunk=distance_transform_edt(~strong)<=p['channel_junction_radius_px']
    weak_background=np.where(near_trunk,np.maximum(local,envelope),local)
    weak=(smooth/np.maximum(weak_background,300.)<p['channel_weak_ratio'])&valid&(margin>1)
    # Hysteresis retains weak small branches attached to a strong dark valley.
    # It does not simplify the result to a prescribed graph or skeleton.
    channels=_remove_small(binary_propagation(strong,mask=weak),p['channel_min_component_area_px'])
    tissue_mask=head&~eyes&~channels
    tissue=gaussian_filter(tissue_mask.astype(float),p['tissue_edge_sigma_px'])
    tissue[~tissue_mask]=0
    interior=distance_transform_edt(tissue_mask)>5
    if interior.sum()<100:
        raise ValueError('Extracted tissue is too small')
    clipped=np.minimum(signal,np.percentile(signal[interior],95))
    emission=gaussian_filter(clipped*interior,14)/np.maximum(gaussian_filter(interior.astype(float),14),1e-8)
    nearest=distance_transform_edt(~interior,return_distances=False,return_indices=True)
    emission[~interior]=emission[tuple(nearest[:,~interior])]
    emission/=max(float(emission[interior].mean()),1.)
    response=gaussian_filter(signal,.65)-gaussian_filter(signal,2.5)
    peaks=(response==maximum_filter(response,size=3))&(response>np.percentile(response[interior],65))&interior
    density=gaussian_filter(peaks.astype(float),25)/np.maximum(gaussian_filter(interior.astype(float),25),1e-8)
    density[~interior]=density[tuple(nearest[:,~interior])]
    density/=max(float(density[interior].mean()),1e-9)
    def coarse(field,shape,low,high):
        gy,gx=np.meshgrid(np.linspace(0,h-1,shape[0]),np.linspace(0,w-1,shape[1]),indexing='ij')
        return np.clip(map_coordinates(field,[gy,gx],order=1),low,high).astype(np.float32)
    # Keep the actual bright peripheral band, including its interruptions.
    # Store its binary shape and a heavily smoothed intensity estimate only.
    edge_distance=np.abs(distance_transform_edt(head)-distance_transform_edt(~head))
    rim=(gaussian_filter(signal,1.)>np.percentile(signal[interior],95))&(edge_distance<7)&~eyes
    rim=_remove_small(rim,12)
    source_scale=max(float(np.mean(signal[interior])),1.)
    rim_strength=gaussian_filter(signal*rim,8)/np.maximum(gaussian_filter(rim.astype(float),8),1e-8)/source_scale
    rim_strength=np.clip(rim_strength,0,8)
    arrays={
        'head_mask':head,'eye_mask':eyes,'channel_mask':channels,
        'tissue_mask':tissue_mask,'tissue_field':tissue.astype(np.float32),
        'rim_mask':rim,'rim_field':gaussian_filter(rim.astype(np.float32),.6),
        'emission_grid':coarse(emission,(17,21),.2,2.5),
        'density_grid':coarse(density,(13,17),.2,2.3),
        'rim_strength_grid':coarse(rim_strength,(17,21),0,8),
    }
    stats={'floor':floor,'eye_components':eye_components,
           'head_area_fraction':float(head.mean()),'eye_area_fraction':float(eyes.mean()),
           'channel_area_fraction':float(channels.mean()),'tissue_area_fraction':float(tissue_mask.mean()),
           'peak_density_proxy_count':int(peaks.sum()),'segmentation_parameters':p}
    return arrays,stats,ratio


def save_template_diagnostic(path,mean,arrays,stats,ratio):
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    panels=[(mean,'Среднее обучающих highSNR',stats['floor'],32767),
            (arrays['head_mask'],'Внешний контур · маска целиком',0,1),
            (arrays['eye_mask'],'Глаза · реальная форма масок',0,1),
            (arrays['channel_mask'],'Тёмные разделения · без графа',0,1),
            (arrays['tissue_field'],'Ткань для размещения новых клеток',0,1)]
    for ax,(data,title,lo,hi) in zip(axes.flat,panels):
        ax.imshow(data,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest');ax.set_title(title,fontsize=11);ax.axis('off')
    ax=axes[1,2];ax.imshow(mean,cmap='gray',vmin=stats['floor'],vmax=32767)
    for key,color in (('head_mask','cyan'),('eye_mask','lime'),('channel_mask','tomato')):
        ax.contour(arrays[key],levels=[.5],colors=[color],linewidths=.45)
    ax.set_title('Наложение: контур, глаза, разделения',fontsize=11);ax.axis('off')
    fig.suptitle('Автоматический растровый шаблон из highSNR · требует визуальной проверки\n'
                 'Это приближённое выделение тёмных структур, не экспертная анатомическая разметка')
    fig.savefig(path,dpi=150);plt.close(fig)
    h,w=mean.shape;crop=(slice(int(.45*h),int(.98*h)),slice(int(.23*w),int(.62*w)))
    fig,axes=plt.subplots(1,3,figsize=(13,5),constrained_layout=True)
    for ax,data,title in zip(axes,[mean[crop],arrays['channel_mask'][crop],arrays['tissue_mask'][crop]],
                             ['Реальный фрагмент','Извлечённые тёмные ответвления','Оставшаяся ткань']):
        ax.imshow(data,cmap='gray',vmin=stats['floor'] if title=='Реальный фрагмент' else 0,
                  vmax=32767 if title=='Реальный фрагмент' else 1,interpolation='nearest')
        ax.set_title(title);ax.axis('off')
    fig.savefig(path.with_name('template_branch_detail.png'),dpi=160);plt.close(fig)


def prepare_template(data_dir=DATA_DIR,template_path=TEMPLATE_PATH,diagnostic_dir=DIAGNOSTIC_DIR,
                     parameters=None):
    manifest=json.loads((data_dir/'manifest.json').read_text())
    sample=manifest['representative_sample'];info=manifest['files']['highSNR']
    path=data_dir/info['sampled_file']
    if sha256(path)!=info['sampled_sha256']:
        raise ValueError('highSNR TIFF checksum mismatch')
    if sample['frame_count']!=300:
        raise ValueError('Expected the existing 25-window representative sample')
    train=[w*12+o for w in range(25) if w%5!=4 for o in range(12)]
    validation=[w*12+o for w in range(25) if w%5==4 for o in range(12)]
    average=None;quantiles=[]
    with tifffile.TiffFile(path) as tif:
        for local in train:
            frame=tif.pages[local].asarray().astype(float)
            if average is None:average=np.zeros_like(frame)
            average+=frame
            quantiles.append(np.percentile(frame,PERCENTILES))
    average=(average/len(train)).astype(np.float32)
    arrays,stats,ratio=extract_geometry(average,parameters)
    ids=sample['frame_indices_zero_based']
    metadata={'model':'HighSNR raster tissue template from one recording',
              'source_dataset':'https://zenodo.org/records/6293696','license':'CC BY 4.0',
              'source_highSNR_path':str(path),'source_highSNR_sha256':sha256(path),
              'shape_yx':list(average.shape),'training_local_indices':train,'validation_local_indices':validation,
              'training_source_indices':[ids[i] for i in train],
              'validation_source_indices':[ids[i] for i in validation],
              'mean_frame_count':len(train),'statistics':stats,
              'mean_registration':'No registration; temporal mean in the recorded pixel coordinates',
              'transferred':'Raster head, eye and dark-gap masks; rim support; coarse emission/density/rim strength; scalar intensity profiles',
              'not_transferred':'Reference neuron intensity texture is not used by the renderer',
              'limitations':['Automatic intensity segmentation can miss faint branches and include intercellular gaps',
                             'Vertical head-envelope assumption fits this FOV, not arbitrary orientation/recordings',
                             'Temporal split is from one animal/recording, not independent anatomy',
                             'Fine mask boundaries can retain reference cell-scale structure; this is hybrid synthesis'],
              'preparation_code_sha256':sha256(Path(__file__))}
    template_path.parent.mkdir(parents=True,exist_ok=True);diagnostic_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(template_path,**arrays,percentiles=PERCENTILES,
                        highSNR_quantiles=np.asarray(quantiles,dtype=np.float32),
                        metadata_json=np.asarray(json.dumps(metadata,ensure_ascii=False)))
    # The mean is a diagnostic artifact; it is absent from the generation template.
    tifffile.imwrite(diagnostic_dir/'training_mean_highSNR.tif',average,photometric='minisblack')
    save_template_diagnostic(diagnostic_dir/'template_extraction.png',average,arrays,stats,ratio)
    (diagnostic_dir/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(f'Prepared highSNR template: {template_path}; {len(train)} training frames; '
          f'{len(stats["eye_components"])} eye candidates',flush=True)
    return template_path


if __name__=='__main__':
    prepare_template()
