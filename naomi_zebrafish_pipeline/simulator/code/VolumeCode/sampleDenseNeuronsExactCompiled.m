function [neur_locs,Vcell,Vnuc,Tri,rotAng] = sampleDenseNeuronsExactCompiled( ...
    neur_params,vol_params,neur_ves)
% Keep NAOMi's alternating placement mask and density weighting, but run
% the repeated 3D location search in a compiled MEX function.

vres = vol_params.vres;
vol_sz = vol_params.vol_sz;
[x,y,z] = ndgrid(-ceil(vol_params.min_dist/2):ceil(vol_params.min_dist/2));
se = strel(sqrt(x.^2+y.^2+z.^2) <= vol_params.min_dist/2);
ves_blocked = imdilate(neur_ves,se);
depth_px = vol_params.vol_depth*vres;
allowed = ~ves_blocked(:,:,1+depth_px:depth_px+vol_sz(3)*vres);

if isfield(vol_params,'neuron_xy_density') && ~isempty(vol_params.neuron_xy_density)
    density = double(vol_params.neuron_xy_density);
    assert(isequal(size(density),vol_sz(1:2)*vres));
    density(~isfinite(density) | density < 0) = 0;
else
    density = ones(vol_sz(1:2)*vres);
end

[sphere,Tri] = SpiralSampleSphere(neur_params.n_samps,false);
neur_params.S_samp = sphere;
neur_params.Tri = Tri;
N = vol_params.N_neur;
Vcell = zeros(neur_params.n_samps,3,N);
Vnuc = zeros(neur_params.n_samps,3,N);
rotAng = zeros(N,3);
draws = zeros(N,1);
for kk = 1:N
    [body,nucleus,~,angle] = generateNeuralBody(neur_params);
    Vcell(:,:,kk) = body;
    Vnuc(:,:,kk) = nucleus;
    rotAng(kk,:) = angle;
    if N > 1
        draws(kk) = rand;
    end
end
if N == 1
    locations = vol_sz/2;
    placed = 1;
else
    [locations,placed] = sample_dense_locations_exact( ...
        allowed,density,double(vol_sz),double(vol_params.min_dist),draws);
end
neur_locs = single(locations(1:placed,:));
Vcell = Vcell(:,:,1:placed);
Vnuc = Vnuc(:,:,1:placed);
rotAng = rotAng(1:placed,:);
Vcell = single(bsxfun(@plus,Vcell, ...
    reshape(neur_locs.',[1,3,placed])));
Vnuc = single(bsxfun(@plus,Vnuc, ...
    reshape(neur_locs.',[1,3,placed])));
if placed < vol_params.N_neur
    warning('Compiled sampler placed %d of %d requested cells', ...
        placed,vol_params.N_neur);
end
end
