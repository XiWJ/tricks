#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_corner_normals.h>
#include <iostream>

/// 查找顶点 i 的领接顶点 j; 查找vertex i 所在的面集合
/// \param F input: face of triangle mesh
/// \param AV output: one-ring neighborhood vertex
/// \param AF output: the faces of the vertex i belongs to
/// \param AFi output: the index of vertex i at the face in AF
void adjacency_list(const Eigen::MatrixXi & F,
                    std::vector<std::vector<int>> & AV,
                    std::vector<std::vector<int>> & AF,
                    std::vector<std::vector<int>> & AFi)
{
    AV.clear();
    AF.clear();
    AFi.clear();

    int n = F.maxCoeff() + 1;
    AV.resize(n);
    AF.resize(n);
    AFi.resize(n);

    for(int f=0; f<F.rows(); f++) {
        for(int m=0; m<F.cols(); m++) {
            int i = F(f, m); // vertex i
            int j = F(f, (m+1) % F.cols()); // vertex j
            AV.at(i).push_back(j); // 将 vertex j 加入 i 的邻接集合中, 会存在重复
            AV.at(j).push_back(i); // 将 vertex i 加入 j 的邻接集合中, 会存在重复
            AF[i].push_back(f);    // 将 vertex i 对应的所在面 f 加入 i 的面的集合中
            AFi[i].push_back(m);   // 将 vertex i 对应的在面 f 中的索引 m
        }
    }

    // 去掉 A[i] 集合中重复元素
    for(int i=0; i<(int)AV.size(); ++i)
    {
        std::sort(AV[i].begin(), AV[i].end());
        AV[i].erase(std::unique(AV[i].begin(), AV[i].end()), AV[i].end());
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

    std::vector<std::vector<int>> AV;
    std::vector<std::vector<int>> AF;
    std::vector<std::vector<int>> AFi;
    adjacency_list(F, AV, AF, AFi);
}

