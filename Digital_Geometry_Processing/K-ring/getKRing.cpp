#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_corner_normals.h>
#include <iostream>

void adjacency_list(const Eigen::MatrixXi & F,
                    std::vector<std::vector<int>> & AV,
                    std::vector<std::vector<int>> & AF,
                    std::vector<std::vector<int>> & AFi){
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

/// 获取 vertex i 的 k-ring 邻接点
/// \param V input: vertex
/// \param start input: 起始点vertex i
/// \param r input: K-ring的k, 也是radius
/// \param AV input: one-ring neighborhood vertex
/// \param vv output: vertex i 的k-ring邻接点
void getKRing(const Eigen::MatrixXd & V,
              const int start, const double r,
              const std::vector<std::vector<int>> & AV,
              std::vector<int>& VKring)
{
    int bufsize=V.rows();
    VKring.reserve(bufsize);
    std::list<std::pair<int,int> > queue; // BFS use queue
    std::vector<bool> visited(bufsize, false);
    queue.push_back(std::pair<int,int>(start,0));
    visited[start]=true;

    // BSF find k-ring neighborhood
    while (!queue.empty())
    {
        int toVisit=queue.front().first;
        int distance=queue.front().second;
        queue.pop_front();
        VKring.push_back(toVisit);
        // if distance < r(k-ring) push in queue
        if (distance<(int)r)
        {
            for (unsigned int i=0; i<AV[toVisit].size(); ++i)
            {
                int neighbor=AV[toVisit][i];
                if (!visited[neighbor]) // non-visited before
                {
                    queue.push_back(std::pair<int,int> (neighbor,distance+1));
                    visited[neighbor]=true;
                }
            }
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
            argc>1?argv[1]: "/home/xiweijie/geoprocessing/libigl/tutorial/data/fertility.off",V,F);

    std::vector<std::vector<int>> AV;
    std::vector<std::vector<int>> AF;
    std::vector<std::vector<int>> AFi;

    adjacency_list(F, AV, AF, AFi);

    unsigned int k = 5; // k-ring
    for(int i=0; i<V.rows(); i++) {
        std::vector<int> VKring;
        getKRing(V, i, k, AV, VKring);
    }
}

