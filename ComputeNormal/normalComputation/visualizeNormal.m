function vis = visualizeNormal(normalMap, validMap, backingImage)
    %David Fouhey, Abhinav Gupta, Martial Hebert
    %Data-Driven 3D Primitives For Single Image Understanding 
    %ICCV 2013
    %
    %Inference-only code

    if nargin < 3
        backingImage = zeros(size(normalMap,1),size(normalMap,2),3,'uint8');
    else
        backingImage = uint8(backingImage/2+128); 
    end


    vis = uint8( 255 * (max(min(normalMap,1),-1)+1) / 2);
    if nargin > 1
        for c=1:3
            vc = vis(:,:,c); bic = backingImage(:,:,c);
            vc(~validMap) = bic(~validMap);
            vis(:,:,c) = vc;
        end 
    end
end
