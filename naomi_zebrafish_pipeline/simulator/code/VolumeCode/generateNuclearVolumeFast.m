function [neur_soma,neur_vol,gp_nuc,gp_soma] = generateNuclearVolumeFast( ...
    neur_params,vol_params,neur_locs,Vcell,Vnuc,Tri)
% Reuse the exact NAOMi nucleus voxel test, while omitting soma voxelization.

vol_sz = vol_params.vol_sz;
vres = vol_params.vres;
K = size(Vnuc,3);
neur_soma = zeros(vol_sz*vres,'uint16');
neur_vol = zeros(vol_sz*vres,'single');
gp_nuc = cell(K,2);
gp_soma = cell(K,1);

for kk = 1:K
    max_ext = ceil(max(sqrt(sum(bsxfun(@plus,Vcell(:,:,kk), ...
                                               -neur_locs(kk,:)).^2,2))));
    mExtRes = ceil(max_ext*vres);
    idx_pos = [round(vres*neur_locs(kk,1)), ...
               round(vres*neur_locs(kk,2)), ...
               round(vres*neur_locs(kk,3))];
    idxX = max(1,idx_pos(1)-mExtRes):min(idx_pos(1)+mExtRes,vol_sz(1)*vres);
    idxY = max(1,idx_pos(2)-mExtRes):min(idx_pos(2)+mExtRes,vol_sz(2)*vres);
    idxZ = max(1,idx_pos(3)-mExtRes):min(idx_pos(3)+mExtRes,vol_sz(3)*vres);
    [mesh_x,mesh_y,mesh_z] = meshgrid( ...
        single((idxX-idx_pos(1))/vres), ...
        single((idxY-idx_pos(2))/vres), ...
        single((idxZ-idx_pos(3))/vres));
    mesh_x = permute(mesh_x,[2 1 3]);
    mesh_y = permute(mesh_y,[2 1 3]);
    mesh_z = permute(mesh_z,[2 1 3]);
    idx_to_test = sqrt(mesh_x.^2+mesh_y.^2+mesh_z.^2) <= max_ext;
    idx_tri = [mesh_x(idx_to_test)+idx_pos(1)/vres+1/vres/2, ...
               mesh_y(idx_to_test)+idx_pos(2)/vres+1/vres/2, ...
               mesh_z(idx_to_test)+idx_pos(3)/vres+1/vres/2];
    inside = intriangulation(Vnuc(:,:,kk),Tri,idx_tri);
    mask = false(size(idx_to_test));
    mask(idx_to_test) = inside;
    [iX,iY,iZ] = ind2sub(size(mask),find(mask));
    iX = iX+idxX(1)-1;
    iY = iY+idxY(1)-1;
    iZ = iZ+idxZ(1)-1;
    indices = sub2ind(vol_sz*vres,iX,iY,iZ);
    gp_nuc{kk,1} = int32(indices);
    gp_nuc{kk,2} = neur_params.nuc_fluorsc;
    neur_vol(indices) = neur_params.nuc_fluorsc;
end
end
