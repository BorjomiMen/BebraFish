% Isolated accelerated zebrafish NAOMi. Does not write to the original run.
% Run from any directory:
% NAOMI_CELL_COUNT=4000 NAOMI_NUC_RAD=1.4 octave --quiet --eval ...
%
% The prior is an image-derived placement probability, not verified neuron labels.
% Set NAOMI_CELL_COUNT to try another count without overwriting the
% 900-cell baseline. For example: NAOMI_CELL_COUNT=4000 octave --quiet ...
% Set NAOMI_NUC_RAD to change the visible nuclear radius in um without
% overwriting the earlier anatomy or loading a cache with the old radius.
% Set NAOMI_SEED to generate a new, reproducible anatomy without
% overwriting the earlier seed's anatomy cache or images.

code_dir = fileparts(mfilename('fullpath'));
out_root = getenv('NAOMI_OUT_ROOT');
assert(~isempty(out_root), 'NAOMI_OUT_ROOT is required');
cell_count_env = getenv('NAOMI_CELL_COUNT');
nucleus_radius_env = getenv('NAOMI_NUC_RAD');
seed_env = getenv('NAOMI_SEED');
depth_env = getenv('NAOMI_DEPTH_UM');
exc_na_env = getenv('NAOMI_EXC_NA');
obj_na_env = getenv('NAOMI_OBJ_NA');
halo_env = getenv('NAOMI_HALO_SCALE');
bg_file = getenv('NAOMI_DIFFUSE_BG_PNG');
bg_level_env = getenv('NAOMI_DIFFUSE_BG_LEVEL');
local_bg_level_env = getenv('NAOMI_LOCAL_BG_LEVEL');
depth_um = 12;
exc_na = 0.6;
obj_na = 0.8;
halo_scale = 1.0;
if ~isempty(depth_env); depth_um = str2double(depth_env); end
if ~isempty(exc_na_env); exc_na = str2double(exc_na_env); end
if ~isempty(obj_na_env); obj_na = str2double(obj_na_env); end
if ~isempty(halo_env); halo_scale = str2double(halo_env); end
assert(isfinite(depth_um) && depth_um >= 8 && depth_um == floor(depth_um), ...
       'NAOMI_DEPTH_UM must be an integer >= 8');
assert(isfinite(exc_na) && exc_na > 0 && exc_na <= obj_na, ...
       'NAOMI_EXC_NA must be positive and <= NAOMI_OBJ_NA');
assert(isfinite(obj_na) && obj_na > 0 && obj_na <= 1.35, ...
       'NAOMI_OBJ_NA must be between 0 and 1.35');
assert(isfinite(halo_scale) && halo_scale >= 0 && halo_scale <= 1, ...
       'NAOMI_HALO_SCALE must be between 0 and 1');
if isempty(seed_env)
    run_seed = 20261024;
else
    run_seed = str2double(seed_env);
    assert(isfinite(run_seed) && run_seed >= 0 && ...
           run_seed == floor(run_seed) && run_seed <= 4294967295, ...
           'NAOMI_SEED must be an integer between 0 and 4294967295');
end
if isempty(cell_count_env)
    cell_count = 900;
else
    cell_count = str2double(cell_count_env);
    assert(isfinite(cell_count) && cell_count >= 1 && ...
           cell_count == floor(cell_count), ...
           'NAOMI_CELL_COUNT must be a positive integer');
end
if isempty(nucleus_radius_env)
    nucleus_radius = 2.0;
else
    nucleus_radius = str2double(nucleus_radius_env);
    assert(isfinite(nucleus_radius) && nucleus_radius > 0 && ...
           nucleus_radius < 3.8, ...
           'NAOMI_NUC_RAD must be between 0 and the 3.8 um soma radius');
end
if abs(nucleus_radius - 2.0) < 1e-9 && cell_count == 900
    out_dir = fullfile(out_root, 'naomi_density_image');
elseif abs(nucleus_radius - 2.0) < 1e-9
    out_dir = fullfile(out_root, sprintf('naomi_density_image_%d', cell_count));
else
    radius_tag = strrep(sprintf('%.2f', nucleus_radius), '.', 'p');
    out_dir = fullfile(out_root, ...
        sprintf('naomi_density_image_%d_nuc%s', cell_count, radius_tag));
end
if ~isempty(seed_env)
    out_dir = sprintf('%s_seed%.0f', out_dir, run_seed);
else
    % Reuse a cache created earlier with the explicit default seed, while
    % preserving older unsuffixed caches from previous runs.
    seeded_dir = sprintf('%s_seed%.0f', out_dir, run_seed);
    if exist(fullfile(seeded_dir, 'anatomy_cache.mat'), 'file') == 2
        out_dir = seeded_dir;
    end
end
if ~isempty(depth_env) || ~isempty(exc_na_env) || ...
   ~isempty(obj_na_env) || ~isempty(halo_env)
    out_dir = sprintf('%s_z%d_exc%.2f_obj%.2f_h%.2f', ...
        out_dir, depth_um, exc_na, obj_na, halo_scale);
end
if ~isempty(bg_file)
    assert(~isempty(bg_level_env), ...
           'NAOMI_DIFFUSE_BG_LEVEL is required with NAOMI_DIFFUSE_BG_PNG');
    bg_level = str2double(bg_level_env);
    assert(isfinite(bg_level) && bg_level >= 0 && bg_level <= 100, ...
           'NAOMI_DIFFUSE_BG_LEVEL must be between 0 and 100');
    out_dir = sprintf('%s_bg%.2f', out_dir, bg_level);
end
if ~isempty(local_bg_level_env)
    local_bg_level = str2double(local_bg_level_env);
    assert(isfinite(local_bg_level) && local_bg_level >= 0 && local_bg_level <= 100, ...
           'NAOMI_LOCAL_BG_LEVEL must be between 0 and 100');
    assert(isempty(bg_file), ...
           'Use either NAOMI_LOCAL_BG_LEVEL or NAOMI_DIFFUSE_BG_PNG, not both');
    out_dir = sprintf('%s_localbg%.2f', out_dir, local_bg_level);
end
forced_run_dir = getenv('NAOMI_RUN_DIR');
if ~isempty(forced_run_dir); out_dir = forced_run_dir; end
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
addpath(genpath(code_dir));
pkg load image;
pkg load statistics;
pkg load signal;
rng(run_seed);

prior_file = getenv('NAOMI_PRIOR_PNG');
assert(~isempty(prior_file) && exist(prior_file, 'file') == 2, ...
       'NAOMI_PRIOR_PNG must point to a density PNG');
density_yx = single(imread(prior_file)) / 65535;
assert(ndims(density_yx) == 2, 'Density PNG must be grayscale');
target_hw = size(density_yx);
pixel_size_um = str2double(getenv('NAOMI_PIXEL_SIZE_UM'));
assert(isfinite(pixel_size_um) && pixel_size_um > 0 && pixel_size_um <= 2, ...
       'NAOMI_PIXEL_SIZE_UM must be in (0, 2]');
% The original Zenodo profile used exactly 350 x 423 native voxels.
if isequal(target_hw, [400 485]) && abs(pixel_size_um - 0.875) < 1e-9
    native_hw = [350 423];
else
    native_hw = max([8 8], round(target_hw * pixel_size_um));
end
% Trial count, not a measured biological count.
prior_native = imresize(density_yx, native_hw, 'bilinear');
prior_native(prior_native < 0.002) = 0;

vol_params = check_vol_params(struct( ...
    'vol_sz', [native_hw depth_um], 'vol_depth', 8, 'vres', 1, ...
    'N_neur', cell_count, 'N_den', 0, 'N_bg', 0, ...
    'min_dist', 3, 'verbose', 0));
vol_params.neuron_xy_density = prior_native;
vol_params.compiled_exact_sampler = true;
vol_params.nuclear_only_fast = true;
neur_params = check_neur_params(struct( ...
    'n_samps', 40, 'avg_rad', 3.8, 'nuc_rad', nucleus_radius, ...
    'nuc_fluorsc', 1));
vasc_params = check_vasc_params(struct('flag', 0));
dend_params = check_dend_params(struct( ...
    'dtParams', [1 40 20 1 0], 'atParams', [1 3 3 3 0]));
dend_params.nuclear_only_smoke = true;
axon_params = check_axon_params(struct('flag', 0));
bg_params = check_bg_params(struct('flag', 0));
psf_params = check_psf_params(struct( ...
    'NA', exc_na, 'objNA', obj_na, 'n', 1.35, ...
    'lambda', 0.92, 'psf_sz', [10 10 10], 'fastmask', false));
scan_params = check_scan_params(struct( ...
    'nuc_label', 1, 'motion', false, 'scan_buff', 0, ...
    'scan_avg', 2, 'sfrac', 1, 'verbose', 1));
if ~isempty(bg_file)
    assert(exist(bg_file, 'file') == 2, 'Diffuse background PNG is missing');
    bg_yx = single(imread(bg_file)) / 65535;
    assert(isequal(size(bg_yx), target_hw), ...
           'Diffuse background PNG must match target shape');
    scan_params.diffuse_background_native = ...
        bg_level * imresize(bg_yx, native_hw, 'bilinear');
end
if ~isempty(local_bg_level_env)
    scan_params.local_background_level = local_bg_level;
end
noise_params = check_noise_params([]);
tpm_params = check_tpm_params(struct('pavg', 40));
spike_opts = check_spike_opts(struct( ...
    'nt', 1, 'dt', 1/15, 'rate', 0, 'prot', 'GCaMP6s', ...
    'axonflag', 0, 'dendflag', 0, 'smod_flag', 'independent'));

fprintf('NAOMi anatomy: %d requested cells, nucleus radius %.2f um, %s um.\n', ...
        cell_count, nucleus_radius, mat2str(vol_params.vol_sz));
cache_file = fullfile(out_dir, 'anatomy_cache.mat');
use_cache = false;
if exist(cache_file, 'file') == 2
    cache_meta = load(cache_file, 'cached_cell_count', 'cached_native_hw', ...
                      'cached_nucleus_radius', 'cached_seed', 'vol_params');
    radius_matches = (isfield(cache_meta, 'cached_nucleus_radius') && ...
                      isequal(cache_meta.cached_nucleus_radius, nucleus_radius)) || ...
                     (~isfield(cache_meta, 'cached_nucleus_radius') && ...
                      abs(nucleus_radius - 2.0) < 1e-9);
    seed_matches = (isfield(cache_meta, 'cached_seed') && ...
                    isequal(cache_meta.cached_seed, run_seed)) || ...
                   (~isfield(cache_meta, 'cached_seed') && ...
                    isempty(seed_env) && run_seed == 20261024);
    use_cache = isequal(cache_meta.cached_cell_count, cell_count) && ...
                isequal(cache_meta.cached_native_hw, native_hw) && ...
                radius_matches && seed_matches && ...
                isfield(cache_meta, 'vol_params') && ...
                isequal(cache_meta.vol_params.vol_sz, [native_hw depth_um]);
end
if use_cache
    load(cache_file, 'vol_out', 'vol_params');
    fprintf('Loaded %d NAOMi cells from cache.\n', size(vol_out.locs, 1));
else
    tic;
    [vol_out, vol_params] = simulate_neural_volume( ...
        vol_params, neur_params, vasc_params, dend_params, ...
        bg_params, axon_params, psf_params);
    fprintf('Anatomy finished in %.1f s; placed %d cells.\n', ...
            toc, size(vol_out.locs, 1));
    cached_cell_count = cell_count;
    cached_native_hw = native_hw;
    cached_nucleus_radius = nucleus_radius;
    cached_seed = run_seed;
    save('-mat7-binary', cache_file, 'vol_out', 'vol_params', ...
         'cached_cell_count', 'cached_native_hw', ...
         'cached_nucleus_radius', 'cached_seed');
end

tic;
% Keep each seed's optical cache beside its anatomy. Independent dataset
% workers must never read a partially written shared cache file.
optics_cache = fullfile(out_dir, 'optics_cache.mat');
use_optics_cache = false;
if exist(optics_cache,'file') == 2 && ~any(vol_out.neur_ves_all(:))
    saved_optics = load(optics_cache);
    use_optics_cache = isequal(saved_optics.cached_vol_sz,vol_params.vol_sz) && ...
        isequal(saved_optics.cached_vres,vol_params.vres) && ...
        isequal(saved_optics.cached_depth,vol_params.vol_depth) && ...
        isequal(saved_optics.cached_psf_params,psf_params);
end
if use_optics_cache
    PSF_struct = saved_optics.PSF_struct;
else
    PSF_struct = simulate_optical_propagation(vol_params, psf_params, vol_out);
    if ~any(vol_out.neur_ves_all(:))
        cached_vol_sz = vol_params.vol_sz;
        cached_vres = vol_params.vres;
        cached_depth = vol_params.vol_depth;
        cached_psf_params = psf_params;
        save('-mat7-binary',optics_cache,'PSF_struct','cached_vol_sz', ...
            'cached_vres','cached_depth','cached_psf_params');
    end
end
fprintf('Optics finished in %.1f s; cache=%d.\n',toc,use_optics_cache);
if halo_scale < 1
    PSF_struct.psfT.weight = PSF_struct.psfT.weight * halo_scale;
    PSF_struct.psfB.weight = PSF_struct.psfB.weight * halo_scale;
end

% Static fluorescence: one baseline value per simulated neuron.
spike_opts.K = size(vol_out.gp_vals, 1);
neur_act = struct('soma', ones(spike_opts.K, 1, 'single'));
% A separate noise seed keeps the noisy frame reproducible whether anatomy
% and optics were recomputed or loaded from cache.
rng(mod(run_seed + 1000003, 4294967296));
tic;
[raw_noisy, raw_clean] = scan_volume( ...
    vol_out, PSF_struct, neur_act, scan_params, ...
    noise_params, spike_opts, tpm_params);
fprintf('Scanning finished in %.1f s; raw shape %s.\n', ...
        toc, mat2str(size(raw_clean)));

assert(isequal(size(raw_clean), native_hw), 'Unexpected NAOMi frame size');
clean = imresize(single(raw_clean), target_hw, 'bilinear');
noisy = imresize(single(raw_noisy), target_hw, 'bilinear');
assert(isequal(size(clean), target_hw));

% A few nonfinite edge pixels arise from NAOMi's optical grid interpolation.
% Replace them only in the resized output; raw arrays remain saved below.
clean(~isfinite(clean)) = 0;
noisy(~isfinite(noisy)) = 0;
% PNG previews use independent display contrast; compare raw values in MAT.
hi_clean = prctile(double(clean(:)), 99.8);
if ~isfinite(hi_clean) || hi_clean <= 0; hi_clean = 1; end
lo_noisy = prctile(double(noisy(:)), 1);
hi_noisy = prctile(double(noisy(:)), 99.8);
if ~isfinite(hi_noisy) || hi_noisy <= lo_noisy
    lo_noisy = 0; hi_noisy = 1;
end
clean_png = uint16(65535 * min(max(clean / hi_clean, 0), 1));
noisy_png = uint16(65535 * min(max((noisy - lo_noisy) / ...
                                     (hi_noisy - lo_noisy), 0), 1));
imwrite(clean_png, fullfile(out_dir, 'naomi_clean.png'));
imwrite(noisy_png, fullfile(out_dir, 'naomi_noisy.png'));

if strcmp(getenv('NAOMI_EXPORT_DATASET'), '1')
    fid = fopen(fullfile(out_dir, 'naomi_noisy_float32_yx.bin'), 'wb');
    assert(fid >= 0, 'Cannot open noisy float output');
    fwrite(fid, noisy.', 'single');
    fclose(fid);
    fid = fopen(fullfile(out_dir, 'naomi_clean_float32_yx.bin'), 'wb');
    assert(fid >= 0, 'Cannot open clean float output');
    fwrite(fid, clean.', 'single');
    fclose(fid);
    % Every row is (cell_id, native_y, native_x). Projected footprints may
    % overlap, so this sparse representation preserves every cell separately.
    fid = fopen(fullfile(out_dir, 'nucleus_footprints_native_u16.bin'), 'wb');
    assert(fid >= 0, 'Cannot open nucleus footprint output');
    for kk = 1:numel(vol_out.gp_nuc(:,1))
        idx = double(vol_out.gp_nuc{kk,1});
        if isempty(idx); continue; end
        [yy, xx, ~] = ind2sub(vol_params.vol_sz, idx);
        xy = unique([yy(:), xx(:)], 'rows');
        records = uint16([repmat(kk, size(xy,1), 1), xy]);
        fwrite(fid, records.', 'uint16');
    end
    fclose(fid);
end

% NAOMi centers are X=row, Y=column in the native simulation volume.
centers_yx_px = [vol_out.locs(:,1) * target_hw(1) / native_hw(1), ...
                 vol_out.locs(:,2) * target_hw(2) / native_hw(2)];
csvwrite(fullfile(out_dir, 'naomi_centers_yx_px.csv'), centers_yx_px);
fprintf('Saved one %d x %d image and masks source in %s\n', ...
        target_hw(1), target_hw(2), out_dir);
