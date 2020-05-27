function [X,Y,Z] = backproject(Z, calibration)
    fu = calibration(1); fv = calibration(2);
    cu = calibration(3); cv = calibration(4);
  
    X = zeros(size(Z)); Y = zeros(size(Z));
    %backproject to make [X,Y,Z]
    for v=1:size(Z,1)
        for u=1:size(Z,2)
            X(v,u) = (Z(v,u)/fu)*(u-cu);
            Y(v,u) = (Z(v,u)/fv)*(v-cv);
        end
    end

end
