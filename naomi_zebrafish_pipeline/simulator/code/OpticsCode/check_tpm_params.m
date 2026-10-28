function tpm_params = check_tpm_params(tpm_params)

% tpm_params = check_tpm_params(tpm_params)
%  
% This function checks the elements of the struct vol_params to ensure
% that all fields are set. Fields not set are set to a list of default
% parameters. The struct checked is:
% 
%   - tpm_params - Struct containing parameters for estimating photon flux
%     .nidx   = 1.33       - refractive index [], water
%     .nac    = 0.8        - objective NA []
%     .phi    = 0.8*SA*0.4 - 80 percent transmission and 0.8NA solid angle
%                            with average 40 perfect PMT QE []
%     .eta    = 0.6        - eGFP quantum yield, estimate for 2P QY []
%     .conc   = 10         - fluorophore concentration, average of  
%                            literature (Huber et al 2012, Zariwala et al
%                            2012), [uM] 
%     .delta  = 35         - two-photon abs. cross section, estimate from
%                            Harris lab (Janelia), saturated GCaMP [GM]
%     .delta  = 2          - estimate of 2pCS at resting calcium levels
%                            [GM] 
%     .gp     = 0.588      - pulse-shape temporal coherence [], sech 
%     .f      = 80         - Ti:S laser rep rate [MHz]
%     .tau    = 150        - Ti:S laser pulse-width [fs]
%     .pavg   = 40         - laser power [mW]
%     .lambda = 0.92       - wavelength used for GCaMP excitation
% 
% 2017 - Alex Song

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run the checks

if isempty(tpm_params)                                                     % Make sure that tpm_params is a struct
    clear tpm_params
    tpm_params = struct;
end

dParams.nidx   = 1.33;                                                     % Index of refraction defaults to that of water
dParams.nac    = 0.8;                                                      % Default objective NA
dParams.phi    = [];                                                       % if this is empty, it will default to 80 percent transmission and 0.8NA light all on detector (no strong scattering) with average 40 perfect PMT QE
dParams.eta    = 0.6;                                                      % eGFP quantum yield
dParams.conc   = 10;                                                       % Average of literature measurements (Huber et al 2012, Zariwala et al 2012), uM
dParams.delta  = 2;                                                        % Estimate from Harris lab (Janelia), saturated GCaMP is 35, estimate of F0 at resting calcium levels
dParams.gp     = 0.588;                                                    % sech pulse shape (temporal) 
dParams.f      = 80;                                                       % Ti:S laser
dParams.tau    = 150;                                                      % Ti:S laser
dParams.pavg   = 40;                                                       % Default 40mW power for L2/3 imaging
dParams.lambda = 0.92;                                                     % 0.92um is the default for excitation wavelength

tpm_params = setParams(dParams, tpm_params);
if isempty(tpm_params.phi)
    tpm_params.phi = 0.8*((1-sqrt(1-(tpm_params.nac/tpm_params.nidx)^2))/2)*0.4;   
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
