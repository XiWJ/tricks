function [fx_rgb, fy_rgb, cx_rgb, cy_rgb] = par_load(cam_txt)
    %packaging everything in a parfor is a pain
    load(cam_txt);
    fx_rgb = cam(1,1);
    fy_rgb = cam(2,2);
    cx_rgb = cam(1,3);
    cy_rgb = cam(2,3);
end
