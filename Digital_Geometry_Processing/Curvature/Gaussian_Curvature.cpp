#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_corner_normals.h>
#include <iostream>

void internal_angles(const Eigen::MatrixXd & V,
                     const Eigen::MatrixXi & F,
                     Eigen::MatrixXd & A) {
    int F_nums = F.rows();
    Eigen::MatrixXd L_sq;
    A.resize(F_nums, 3);
    L_sq.resize(F_nums, 3);

    /// 计算每条边长度的平方 a^2, b^2, c^2
    for(int i=0; i<F_nums; i++) {
        Eigen::Vector3d v1 = V.row(F(i, 0));
        Eigen::Vector3d v2 = V.row(F(i, 1));
        Eigen::Vector3d v3 = V.row(F(i, 2));

        Eigen::Vector3d e_a = v2 - v3; /// v1的对边
        Eigen::Vector3d e_b = v3 - v1; /// v2的对边
        Eigen::Vector3d e_c = v1 - v2; /// v3的对边

        L_sq.row(i) << e_a.squaredNorm(), e_b.squaredNorm(), e_c.squaredNorm();

        /// a^2 = b^2 + c^2 - 2*b*c*Cos(A)
        /// https://en.wikipedia.org/wiki/Triangle#Sine,_cosine_and_tangent_rules
        for(int d=0; d<3; d++) {
            double s1 = L_sq(i, d);
            double s2 = L_sq(i, (d+1) % 3);
            double s3 = L_sq(i, (d+2) % 3);

            A(i, d) = acos((s3 + s2 - s1) / (2. * sqrt(s3 * s2)));
        }
    }
}

void gaussian_curvature(const Eigen::MatrixXd & V,
                        const Eigen::MatrixXi & F,
                        Eigen::MatrixXd & K) {
    /// 计算每个conner的角度
    Eigen::MatrixXd A;
    internal_angles(V, F, A);
    K.setConstant(V.rows(), 1, 2*igl::PI);

    /// K_G(x_i) = (2π - ∑θj)
    for(int i=0; i<F.rows(); i++) {
        for(int j=0; j<3; j++) {
            K(F(i, j), 0) -= A(i, j);
        }
    }
}

int main(int argc, char *argv[])
{
    using namespace Eigen;
    using namespace std;

    MatrixXd V;
    MatrixXi F;

    // Load a mesh in OFF format
    igl::read_triangle_mesh(
            argc>1?argv[1]: "/home/xiweijie/geoprocessing/libigl/tutorial/data/bumpy.off",V,F);

    MatrixXd K;
    gaussian_curvature(V, F, K);

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().show_lines = true;
    viewer.data().set_mesh(V, F);
    viewer.data().set_data(K);
    viewer.launch();
}

