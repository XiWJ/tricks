addpath('normalComputation/');
globals;

%in principle, you can change this 
reducedSize = 4;
seedLocation = 'rngSeed.mat';

%%%%%%%%%%%%%%%%%%%%%
%download the raw data
%%%%%%%%%%%%%%%%%%%%%
if ~exist(NYUDatasetLocal,'file');
    fprintf('Downloading NYU Dataset\n');
    system(sprintf('wget %s -O %s',NYUDatasetRemote,NYUDatasetLocal));
end

%if ~exist(NYUSplitLocal,'file')
%    fprintf('Downloading train/test splits\n');
%    system(sprintf('wget %s -O %s',NYUSplitRemote,NYUSplitLocal));
%end

%%%%%%%%%%%%%%%%%%%%%
%make the dataset folder if it doesn't exist
%%%%%%%%%%%%%%%%%%%%%
if ~exist(datasetLocation)
    mkdir(datasetLocation);
end

if exist(seedLocation,'file')
    load(seedLocation);
    rng(s);
else
    s = rng();
    save(seedLocation,'s');
end

ds = matfile(NYUDatasetLocal);
load(NYUSplitLocal);

imageTarget = [datasetLocation 'canonicalSplits/'];
depthTarget = [datasetLocation 'depth/'];
normalTarget = [datasetLocation 'normals/'];
normalReducedTarget = [datasetLocation 'normalsR4/'];

if ~exist(imageTarget,'file')
    mkdir(imageTarget);
end

if ~exist(depthTarget,'file');
    mkdir(depthTarget);
end

if ~exist(normalTarget,'file')
    mkdir(normalTarget);
end

if ~exist(normalReducedTarget,'file')
    mkdir(normalReducedTarget);
end


%%%%%%%%%%%%%%%%%%%%%
%split up the training data
%%%%%%%%%%%%%%%%%%%%%

train = trainNdxs;
test = testNdxs;

trainShuffle = train(randperm(numel(train)));
splitPoint = fix(numel(train)*0.8);

trainTrain = trainShuffle(1:splitPoint);
trainHN = trainShuffle(splitPoint+1:end);

targets = {trainTrain,trainHN,test};
targetNames = {'trainTrain/','trainHN/','test/'};
targetDesc = {'Training','Hard Negatives','Testing'};

%%%%%%%%%%%%%%%%%%%%%
%dump the images first
%%%%%%%%%%%%%%%%%%%%%
for i=1:numel(targetNames)
    subtarget = [imageTarget '/' targetNames{i}];
    if ~exist(subtarget)
        mkdir(subtarget);
    end
    for ii=targets{i}'
        output = sprintf('%s/rgb_%06d.jpg',subtarget,ii);
        if ~exist(output,'file')
            fprintf('Dumping images for %s: image %d\n',targetDesc{i},ii);
            imwrite(ds.images(:,:,:,ii),output);
        end
    end
end

numImages = size(ds.images,4);

%%%%%%%%%%%%%%%%%%%%%
%dump the depth next
%%%%%%%%%%%%%%%%%%%%%
for i=1:numImages
    output = sprintf('%s/depth_%06d.mat',depthTarget,i);
    if ~exist(output,'file')
        depth = ds.rawDepths(:,:,i);
        fprintf('Dumping depths, image %d / %d\n',i,numImages);
        save(output,'depth');
    end
end

%%%%%%%%%%%%%%%%%%%%%
%make normals
%%%%%%%%%%%%%%%%%%%%%
fprintf('Computing surface normals. This will take a while...\n');

parfor i=1:numImages
    fprintf('Computing Normals, normal %d/%d\n',i,numImages);
    source = sprintf('%s/depth_%06d.mat',depthTarget,i); 
    dest = sprintf('%s/nm_%06d.mat',normalTarget,i);

    if exist(dest,'file')
        continue;
    end

    depthData = load(source);
    [nx, ny, nz, depthValid] = computeNormalsDSPre(depthData.depth);
    saveNormals(dest,nx,ny,nz,depthValid);
end 


%%%%%%%%%%%%%%%%%%%%%
%make resized normals
%%%%%%%%%%%%%%%%%%%%%

parfor i=1:numImages
    fprintf('Resizing normals, normal %d/%d\n',i,numImages);
    source = sprintf('%s/nm_%06d.mat',normalTarget,i);
    dest = sprintf('%s/nm_%06d.mat',normalReducedTarget,i);

    if exist(dest,'file')
        continue;
    end

    fullNormal = load(source);
    nx = imresize(fullNormal.nx,1.0/reducedSize,'bilinear');
    ny = imresize(fullNormal.ny,1.0/reducedSize,'bilinear');
    nz = imresize(fullNormal.nz,1.0/reducedSize,'bilinear');
    renorm = (nx.^2+ny.^2+nz.^2).^0.5;
    nx = nx ./ renorm; ny = ny ./ renorm; nz = nz ./ renorm;
    depthValid = imresize(fullNormal.depthValid,1.0/reducedSize) > 0; 

    saveNormals(dest,nx,ny,nz,depthValid);
end 


