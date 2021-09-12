#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/// Compute area of each face
/// \param V vertex
/// \param F faces
/// \param A area
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

/// Compute the square of length of each edge
/// \param V vertex
/// \param F faces
/// \param L the square of length
void squared_edge_lengths(
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & F,
        Eigen::Matrix<double, Eigen::Dynamic, 3> & L)
{
    using namespace std;
    const int m = F.rows();

    L.resize(m,3);
    // loop over faces
    for(int i=0; i<m; i++)
    {
        L(i,0) = (V.row(F(i,1))-V.row(F(i,2))).squaredNorm();
        L(i,1) = (V.row(F(i,2))-V.row(F(i,0))).squaredNorm();
        L(i,2) = (V.row(F(i,0))-V.row(F(i,1))).squaredNorm();
    }
}

/// Compute cotangent between each edge ij
/// \param V vertex
/// \param F faces
/// \param C cotangent
void cotmatrix_entries(const Eigen::MatrixXd & V,
               const Eigen::MatrixXi & F,
               Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> & C)
{
    using namespace std;
    using namespace Eigen;
    // Number of elements
    int m = F.rows();
    // Triangles
    //Compute Squared Edge lengths
    Matrix<double, Dynamic, 3> l2;
    squared_edge_lengths(V,F,l2);
    //Compute Edge lengths
    Matrix<double, Dynamic, 3> l;
    l = l2.array().sqrt();

    // double area
    Matrix<double, Dynamic, 1> dblA;
    doublearea(V, F, dblA);

    // cotangents and diagonal entries for element matrices
    // correctly divided by 4 (alec 2010)
    C.resize(m,3);
    for(int i = 0;i<m;i++)
    {
        // Alec: I'm doubtful that using l2 here is actually improving numerics.
        C(i,0) = (l2(i,1) + l2(i,2) - l2(i,0))/dblA(i)/4.0;
        C(i,1) = (l2(i,2) + l2(i,0) - l2(i,1))/dblA(i)/4.0;
        C(i,2) = (l2(i,0) + l2(i,1) - l2(i,2))/dblA(i)/4.0;
    }
}

/// Cotangent laplacian weight matrix
/// \param V vertex
/// \param F faces
/// \param L cotangent weight matrix
void cotmatrix(const Eigen::MatrixXd & V,
               const Eigen::MatrixXi & F,
               Eigen::SparseMatrix<double> & L)
{
    using namespace Eigen;
    using namespace std;

    L.resize(V.rows(),V.rows());
    Matrix<int,Dynamic,2> edges;
    // This is important! it could decrease the comptuation time by a factor of 2
    // Laplacian for a closed 2d manifold mesh will have on average 7 entries per
    // row
    L.reserve(10*V.rows());
    edges.resize(3,2);
    edges <<
        1,2,
        2,0,
        0,1;

    // Gather cotangents
    Matrix<double, Dynamic, Dynamic> C;
    cotmatrix_entries(V, F,C);

    vector<Triplet<double> > IJV;
    IJV.reserve(F.rows()*edges.rows()*4);
    // Loop over triangles
    for(int i = 0; i < F.rows(); i++)
    {
        // loop over edges of element
        for(int e = 0; e<edges.rows(); e++)
        {
            int source = F(i,edges(e,0));
            int dest = F(i,edges(e,1));
            IJV.push_back(Triplet<double>(source, dest, C(i,e)));
            IJV.push_back(Triplet<double>(dest, source, C(i,e)));
            IJV.push_back(Triplet<double>(source, source, -C(i,e)));
            IJV.push_back(Triplet<double>(dest, dest, -C(i,e)));
        }
    }
    L.setFromTriplets(IJV.begin(),IJV.end());
}

int main(int argc, char *argv[])
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    Eigen::SparseMatrix<double> L, L1; /// cotangent weight matrix

    // Load a mesh in OFF format
    igl::read_triangle_mesh(
            argc>1?argv[1]: "/home/xiweijie/geoprocessing/laplacian_surface_editing/3D/box.obj",V,F);
    cotmatrix(V, F, L);
    igl::cotmatrix(V, F, L1);
    std::cout << L << std::endl;
    std::cout << "--------------- Comparison --------------" << std::endl;
    std::cout << L1 << std::endl;
}

