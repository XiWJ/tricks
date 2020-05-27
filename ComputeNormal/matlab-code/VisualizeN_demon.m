addpath('/home/xiweijie/github/npy-matlab/npy-matlab/');
addpath('normalComputation/');
Source = '/media/xiweijie/xiwj_drive0/projects/data/demon/dpsnet/tmp/';
train_txt = 'train.txt';
test_txt = 'test.txt';
fpn = fopen([Source train_txt], 'rt');
files = [];
while ~feof(fpn)
    file = fgetl(fpn);
    files = [files; file];
    %fprintf('%s\n',file);
end
fclose(fpn);
[numfiles col] = size(files);

%if ~exist(normalTarget,'file')
%    mkdir(normalTarget);
%end

for i=1:numfiles
    fprintf('%s\n',files(i,:));
    depthPath = [Source '/' files(i,:)];
    %cam_txt = [depthPath '/cam.txt'];
    %load(cam_txt);
    %fx_rgb = cam(1,1);
    %fy_rgb = cam(2,2);
    %cx_rgb = cam(1,3);
    %cy_rgb = cam(2,3);
    names = dir([depthPath '/00*.npy']);
    for nameI=1:numel(names)
        name = names(nameI).name;
        depthSource = [depthPath '/' name];
        %depthData = readNPY([depthPath '/' name]);
        %[nx, ny, nz, depthValid] = computeNormalsDSPre(depthData, fx_rgb, fy_rgb, cx_rgb, cy_rgb);
        %normal_vector = cat(3, nx, ny, nz);
        normalTarget = [depthPath '/normal_' name];
        %saveNormals(normalTarget, normal_vector, depthValid, '.npy');
        normal_vector = readNPY(normalTarget);
        depthValidTarget = strrep(normalTarget, 'normal', 'normal_mask');
        depthValid = readNPY(depthValidTarget);
        visDense = visualizeNormal(normal_vector, depthValid);
        saveNormal = strrep(normalTarget, '.npy', '.png');
        imwrite(visDense, saveNormal);
    end
end 

