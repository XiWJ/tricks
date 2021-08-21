# triangle normal computation
## per faces normal
- per-faces-normal也就是triangle每个面上的normal, 是constant的<br>
![](pics/per-face.jpeg)
- 计算公式:<br>
![](http://latex.codecogs.com/svg.latex?\mathbf{n} = e_1 \times e_2 = (v_1 - v_2) \times (v_1 - v_3))
- C++ code
```c++
void per_face_normal(const Eigen::MatrixXd & V,
                     const Eigen::MatrixXi & F,
                     Eigen::MatrixXd & N_faces){
    N_faces.resize(F.rows(), 3);
    for(int i=0; i<F.rows(); i++) {
        Eigen::Vector3d v1 = V.row(F(i, 0));
        Eigen::Vector3d v2 = V.row(F(i, 1));
        Eigen::Vector3d v3 = V.row(F(i, 2));

        Eigen::Vector3d e1 = v1 - v2; /// 边 e1
        Eigen::Vector3d e2 = v1 - v3; /// 边 e2
        N_faces.row(i) = e1.cross(e2); /// 叉乘
    }
}
```