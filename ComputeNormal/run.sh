#!/bin/bash
echo "**********************************************************************"
echo "compute surface normal for deMoN dataset!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "**********************************************************************"
sudo docker run -it --mount type=bind,source=/data4T2/xiwj/,target=/opt/xiwj/ sitonholy/matlab2017:v1.0 /bin/bash
cd /opt/xiwj/github/y-mvsnet/ComputeNormal/
matlab -nodesktop -nosplash -r generate_normal_gt_demon
