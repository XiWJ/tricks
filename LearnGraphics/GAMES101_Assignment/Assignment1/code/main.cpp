#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.

//    float alpha = rotation_angle / 180 * MY_PI; // aplha from angle to π
//    Eigen::Matrix4f R;
//    R <<   cos(alpha), sin(alpha), 0.0f, 0.0f,
//         - sin(alpha), cos(alpha), 0.0f, 0.0f,
//         0.0f,         0.0f,       1.0f, 0.0f,
//         0.0f,         0.0f,       0.0f, 1.0f;
//
//    model = R * model;

    float alpha = rotation_angle / 180 * MY_PI; // aplha from angle to π
    Eigen::Vector3f nz{0.0f, 0.0f, 1.0f}; // z-axis vector

    Eigen::Matrix3f N, R;
    N <<     0.0f, - nz.z(),   nz.y(),
           nz.z(),     0.0f, - nz.x(),
         - nz.y(),   nz.x(),     0.0f;

    R << cos(alpha) * Eigen::Matrix3f::Identity() + (1 - cos(alpha)) * nz * nz.transpose() + sin(alpha) * N;

    model.block(0, 0, 3, 3) << R;

    return model;
}

Eigen::Matrix4f get_rotation(Vector3f axis, float angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Rotation around any axis by angle

    float alpha = angle / 180 * MY_PI; // aplha from angle to π

    Eigen::Matrix3f N, R;
    N << 0.0f,   - axis.z(),   axis.y(),
         axis.z(),     0.0f, - axis.x(),
       - axis.y(), axis.x(),       0.0f;

    R << cos(alpha) * Eigen::Matrix3f::Identity() + (1 - cos(alpha)) * axis * axis.transpose() + sin(alpha) * N;

    model.block(0, 0, 3, 3) << R;

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.

    Eigen::Matrix4f Mp2o, Mo, Mo1;
    Mp2o << zNear, 0.0f,  0.0f, 0.0f,
            0.0f,  zNear, 0.0f, 0.0f,
            0.0f,  0.0f,  zNear + zFar, - zNear * zFar,
            0.0f,  0.0f,  1.0f, 0.0f;

    float fovY = eye_fov / 180 * MY_PI;
    float t = tan(fovY / 2) * abs(zNear);
    float r = aspect_ratio * t;
    float b = 0.0f - t;
    float l = 0.0f - r;

    Mo << 2.0f / (r - l), 0.0f,           0.0f,                  0.0f,
          0.0f,           2.0f / (t - b), 0.0f,                  0.0f,
          0.0f,           0.0f,           2.0f / (zNear - zFar), 0.0f,
          0.0f,           0.0f,           0.0f,                  1.0f;
    Mo1 << 1.0f, 0.0f, 0.0f, - (r + l) / 2.0f,
           0.0f, 1.0f, 0.0f, - (t + b) / 2.0f,
           0.0f, 0.0f, 1.0f, - (zNear + zFar) / 2.0f,
           0.0f, 0.0f, 0.0f, 1.0f;

    projection = Mo * Mo1 * Mp2o * projection;

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
