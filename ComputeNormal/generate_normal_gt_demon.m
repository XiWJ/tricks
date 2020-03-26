%% Compute Surface Normal version must be MATLAB 2017a +
clc;clear all;
addpath('./npy-matlab/npy-matlab/');
addpath('./normalComputation/');
%% path set
Source = string('/opt/xiwj/demon/dpsnet/train/');
error_path = '/opt/xiwj/github/y-mvsnet/tmp/';
train_txt = 'train.txt';
test_txt = 'test.txt';
%% list file paths
fpn = fopen([Source+'/'+train_txt], 'rt');
files = [];
while ~feof(fpn)
    file = string(fgetl(fpn)); %% using string version must be MATLAB 2017a +
    files = [files; file];
    %fprintf('%s\n',file);
end
fclose(fpn);
[numfiles col] = size(files);
%% parrall for loop compute surface normal
fprintf('par for starting\n');
error_files = [];
parfor i=1:numfiles
    fprintf('%s\n',files(i)); %% string files, version must be MATLAB 2017a +
    depthPath = [Source+'/'+files(i)];
    cam_txt = load([depthPath+'/cam.txt']);
    fx_rgb = cam_txt(1,1);
    fy_rgb = cam_txt(2,2);
    cx_rgb = cam_txt(1,3);
    cy_rgb = cam_txt(2,3);
    names = dir(char([depthPath+'/0*.npy']));
    for nameI=1:numel(names)
        name = names(nameI).name;
        depthSource = [depthPath+'/'+name];
        normalTarget = [depthPath+'/normal_'+name];
        %if exist(normalTarget)
        %    continue;
        %end
        try
            depthData = readNPY(depthSource);
        catch
            error_files = [error_files; depthSource];
            fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');
            fprintf('%s is error!!!!\n',depthSource);
            fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');
            continue;
        end
        [nx, ny, nz, depthValid] = computeNormalsDSPre(depthData, fx_rgb, fy_rgb, cx_rgb, cy_rgb);
        normal_vector = cat(3, nx, ny, nz);
        saveNormals(normalTarget, normal_vector, depthValid, '.npy');
    end
end 
%% save error mat
save([error_path '/error.mat'], 'error_files')
