#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_corner_normals.h>
#include <iostream>

Eigen::MatrixXd V;
Eigen::MatrixXi F;

Eigen::MatrixXd N_vertices;
Eigen::MatrixXd N_faces;
Eigen::MatrixXd N_corners;


// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    switch(key)
    {
        case '1':
            viewer.data().set_normals(N_faces);
            return true;
            case '2':
                viewer.data().set_normals(N_vertices);
                return true;
                case '3':
                    viewer.data().set_normals(N_corners);
                    return true;
                    default: break;
    }
    return false;
}

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

void doublearea(const Eigen::MatrixXd & V,
                const Eigen::MatrixXi & F,
                Eigen::Matrix<double, Eigen::Dynamic, 1> & A){
    A.setZero(F.rows(), 1);
    /// 计算triangle面积, 采用shoelace formula, 具体参考
    /// https://en.wikipedia.org/wiki/Triangle#Using_coordinates
    for(int i=0; i<F.rows(); i++) {
        for(int j=0; j<3; j++) {
            int x = j;
            int y = (j+1) % 3;
            double rx = V(F(i, 0), x) - V(F(i, 2), x); /// X_A - X_C
            double sy = V(F(i, 1), y) - V(F(i, 0), y); /// Y_B - Y_A
            double ry = V(F(i, 0), x) - V(F(i, 1), x); /// X_A - X_B
            double sx = V(F(i, 2), y) - V(F(i, 0), y); /// Y_C - Y_A
            double square_sum = pow((rx * sy - ry * sx),2); /// ||(X_A - X_C) * (Y_B - Y_A) - (X_A - X_B) * (Y_C - Y_A)||2
            A(i, 0) += square_sum;
        }
        A(i, 0) = sqrt(A(i, 0));
        /// 正常来说, 计算triangle面积需要乘 1/2, 但这里返回的是2倍的面积
        //A(i, 0) /= 2.0;
    }
}

void per_vertex_normals(const Eigen::MatrixXd & V,
                        const Eigen::MatrixXi & F,
                        Eigen::MatrixXd & N_vertices){
    Eigen::MatrixXd N_faces;
    per_face_normal(V, F, N_faces);

    N_vertices.setZero(V.rows(), 3);
    Eigen::Matrix<double, Eigen::Dynamic, 1> A;
    Eigen::Matrix<double, Eigen::Dynamic, 1> A2;
    doublearea(V,F,A);
    Eigen::MatrixXd W = A.replicate(1, 3);

    for(int f=0; f<F.rows(); f++) {
        /// 将面上的normal分配给面上的三个vertex
        for(int i=0; i<3; i++) {
            N_vertices.row(F(f, i)) += W(f, i) * N_faces.row(f);
        }
    }
    N_vertices.rowwise().normalize(); /// 归一化
}

int main(int argc, char *argv[])
{
    // Load a mesh in OFF format
    igl::read_triangle_mesh(
            argc>1?argv[1]: "/home/xiweijie/geoprocessing/libigl/tutorial/data/fandisk.off",V,F);

    // Compute per-face normals
//    igl::per_face_normals(V,F,N_faces);
    per_face_normal(V, F, N_faces);

    // Compute per-vertex normals
//    igl::per_vertex_normals(V, F, N_vertices);
    per_vertex_normals(V,F,N_vertices);

    // Compute per-corner normals, |dihedral angle| > 20 degrees --> crease
    igl::per_corner_normals(V,F,20,N_corners);

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.callback_key_down = &key_down;
    viewer.data().show_lines = false;
    viewer.data().set_mesh(V, F);
    viewer.data().set_normals(N_faces);
    std::cout<<
    "Press '1' for per-face normals."<<std::endl<<
    "Press '2' for per-vertex normals."<<std::endl<<
    "Press '3' for per-corner normals."<<std::endl;
    viewer.launch();
}

