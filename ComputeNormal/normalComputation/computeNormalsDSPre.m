function [Nx,Ny,Nz,valid] = computeNormalsDSPre(D, fx_rgb, fy_rgb, cx_rgb, cy_rgb)
    sampleFactor = 4;
    bilateralSigmaSpatial = 24;
    bilateralSigmaIntensity = 0.2;
    %camera_params;
    params = [fx_rgb, fy_rgb, cx_rgb, cy_rgb]; 

    %downsample to compute at a higher resolution
    Dd = imresize(D, 1.0 / sampleFactor);
    [Xd,Yd,Zd] = backproject(Dd, params / sampleFactor);
    [Nxd, Nyd, Nzd] = surfnorm(Xd,Yd,Zd);

    %convert to [0,1]
    NxU = (Nxd + 1) / 2; NyU = (Nyd + 1) / 2; NzU = (Nzd + 1) / 2;
    %bilateral filter
    NxU = bfilter2(NxU, bilateralSigmaSpatial*3, [bilateralSigmaSpatial,bilateralSigmaIntensity]);
    NyU = bfilter2(NyU, bilateralSigmaSpatial*3, [bilateralSigmaSpatial,bilateralSigmaIntensity]);
    NzU = bfilter2(NzU, bilateralSigmaSpatial*3, [bilateralSigmaSpatial,bilateralSigmaIntensity]);
    %reconvert to normals, and then upsample
    Nx = imresize(2*NxU-1, sampleFactor,'method','bilinear');
    Ny = imresize(2*NyU-1, sampleFactor,'method','bilinear');
    Nz = imresize(2*NzU-1, sampleFactor,'method','bilinear');

    %scale to units
    N = Nx.^2 + Ny.^2 + Nz.^2;
    N = N.^0.5;
    Nx = Nx ./ N; Ny = Ny ./ N; Nz = Nz ./ N;
    
    %compute valid map
    %valid = imerode(imdilate(D > 0,ones(3)), ones(sampleFactor*6));
    valid = D>0;


end
