/*
 * @Description: ICP SVD lidar odometry
 * @Author: Ge Yao
 * @Date: 2020-10-24 21:46:45
 */

#include <pcl/common/transforms.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "glog/logging.h"

#include "lidar_localization/models/registration/icp_svd_registration.hpp"

namespace lidar_localization {

ICPSVDRegistration::ICPSVDRegistration(
    const YAML::Node& node
) : input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    // parse params:
    float max_corr_dist = node["max_corr_dist"].as<float>();
    float trans_eps = node["trans_eps"].as<float>();
    float euc_fitness_eps = node["euc_fitness_eps"].as<float>();
    int max_iter = node["max_iter"].as<int>();

    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

ICPSVDRegistration::ICPSVDRegistration(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) : input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

bool ICPSVDRegistration::SetRegistrationParam(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) {
    // set params:
    max_corr_dist_ = max_corr_dist;
    trans_eps_ = trans_eps;
    euc_fitness_eps_ = euc_fitness_eps;
    max_iter_ = max_iter;

    LOG(INFO) << "ICP SVD params:" << std::endl
              << "max_corr_dist: " << max_corr_dist_ << ", "
              << "trans_eps: " << trans_eps_ << ", "
              << "euc_fitness_eps: " << euc_fitness_eps_ << ", "
              << "max_iter: " << max_iter_ 
              << std::endl << std::endl;

    return true;
}

bool ICPSVDRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target) {
    input_target_ = input_target;
    input_target_kdtree_->setInputCloud(input_target_);

    return true;
}

bool ICPSVDRegistration::ScanMatch(
    const CloudData::CLOUD_PTR& input_source, 
    const Eigen::Matrix4f& predict_pose, 
    CloudData::CLOUD_PTR& result_cloud_ptr,
    Eigen::Matrix4f& result_pose
) {
    input_source_ = input_source;

    // pre-process input source:
    CloudData::CLOUD_PTR transformed_input_source(new CloudData::CLOUD());
    pcl::transformPointCloud(*input_source_, *transformed_input_source, predict_pose);

    // init estimation:
    // Eigen::Matrix4f t_transform;
    // t_transform.setIdentity();
    // static long scan_count = 0;
    // if(scan_count++==0)
        transformation_.setIdentity();
    
    //
    // TODO: first option -- implement all computing logic on your own
    //
    // do estimation:
    int curr_iter = 0;
    while (curr_iter < max_iter_) {
        // TODO: apply current estimation:
        Eigen::Matrix4f t_current_pose = transformation_ * predict_pose;
        pcl::transformPointCloud(*input_source_, *transformed_input_source, t_current_pose);

        // TODO: get correspondence:
        std::vector<Eigen::Vector3f> xs, ys;
        size_t t_correspondence = GetCorrespondence(transformed_input_source, xs, ys);

        // TODO: do not have enough correspondence -- break:
        if(t_correspondence < 4 || xs.size() != ys.size() || xs.size()<=0)
        {
            std::cout<<"icp_svd not enough correspondence "<<std::endl;
            return false;
        }

        // TODO: update current transform:
        Eigen::Matrix4f t_iter_transform;
        GetTransform(xs, ys, t_iter_transform);

        // calculate whether match fitness epsilon
        float t_avg_correspond_dist_sq = 0.0f;
        for (int i = 0; i < xs.size(); i++)
        {
            float t_correspond_dist_sq = pow(xs[i].x()-ys[i].x(), 2) + 
                                            pow(xs[i].y()-ys[i].y(), 2) + 
                                            pow(xs[i].z()-ys[i].z(), 2);
            t_avg_correspond_dist_sq += t_correspond_dist_sq;
        }
        t_avg_correspond_dist_sq /= xs.size();
        if(t_avg_correspond_dist_sq > euc_fitness_eps_)
        {
            std::cout<<"correspond avg dist sq too large: " << t_avg_correspond_dist_sq <<std::endl;
            return false;
        }

        // TODO: whether the transformation update is significant:
        // break if convergent
        if (!IsSignificant(t_iter_transform, trans_eps_)) {
            // transformation_ = t_iter_transform * transformation_;
            break;
        }

        // TODO: update transformation:
        transformation_ = t_iter_transform * transformation_;

        ++curr_iter;
    }

    if(curr_iter >= max_iter_){
        transformation_.setIdentity();
    }

    // set output:
    result_pose = transformation_ * predict_pose;
    pcl::transformPointCloud(*input_source_, *result_cloud_ptr, result_pose);

    // std::cout<<"quat: "<<Eigen::Quaternionf(transformation_.block<3,3>(0,0)).coeffs().transpose() <<" ---- "<<curr_iter<<std::endl;
    
    return true;
}

size_t ICPSVDRegistration::GetCorrespondence(
    const CloudData::CLOUD_PTR &input_source, 
    std::vector<Eigen::Vector3f> &xs,
    std::vector<Eigen::Vector3f> &ys
) {
    const float MAX_CORR_DIST_SQR = max_corr_dist_ * max_corr_dist_;

    size_t num_corr = 0;

    // TODO: set up point correspondence
    xs.clear();
    ys.clear();
    const int K_size = 1;
    std::vector<int> t_indices;
    std::vector<float> t_sqr_dists;
    t_indices.resize(K_size);
    t_sqr_dists.resize(K_size);
    for (int i = 0; i < input_source->size(); i++)
    {
        int neighbors = input_target_kdtree_->nearestKSearch(input_source->points[i], K_size, t_indices, t_sqr_dists);
        if(neighbors>0 && t_sqr_dists[0] < MAX_CORR_DIST_SQR)
        {
            xs.push_back(Eigen::Vector3f(input_target_->points[t_indices[0]].x,
                                        input_target_->points[t_indices[0]].y,
                                        input_target_->points[t_indices[0]].z));
            ys.push_back(Eigen::Vector3f(input_source->points[i].x,
                                        input_source->points[i].y,
                                        input_source->points[i].z));
            num_corr++;
        }
    }

    return num_corr;
}

void ICPSVDRegistration::GetTransform(
    const std::vector<Eigen::Vector3f> &xs,
    const std::vector<Eigen::Vector3f> &ys,
    Eigen::Matrix4f &transformation
) {
    const size_t N = xs.size();

    // TODO: find centroids of mu_x and mu_y:
    Eigen::Vector3f ux=Eigen::Vector3f::Zero(), uy=Eigen::Vector3f::Zero();
    for (size_t i = 0; i < N; i++)
    {
        ux += xs[i];
        uy += ys[i];
    }
    ux /= N;
    uy /= N;

    // TODO: build H:
    // H = sum((yi-uy) * (xi-ux)_t)
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < N; i++)
    {
        H += (ys[i]-uy) * (xs[i]-ux).transpose();
    }

    // TODO: solve R:
    // H = U * S * V_t
    // R = V * U_t
    Eigen::JacobiSVD<Eigen::MatrixXf> svd_h(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f u = svd_h.matrixU();
    Eigen::Matrix3f v = svd_h.matrixV();
    // v矩阵z轴旋转符号调整
    if(u.determinant() * v.determinant()<0)
    {
        for (size_t i = 0; i < 3; i++)
        {
            v(i,2) *= -1;
        }
    }
    
    Eigen::Matrix3f R = v * u.transpose();
    Eigen::Quaternionf t_quat(R);
    t_quat.normalize();

    // TODO: solve t:
    // e2 = || ux - R * uy -t ||^2
    // t = ux - R * uy
    Eigen::Vector3f t = ux - t_quat.toRotationMatrix() * uy;

    // TODO: set output:
    transformation << t_quat.toRotationMatrix(), t, 0.0f, 0.0f, 0.0f, 1.0f;
}

bool ICPSVDRegistration::IsSignificant(
    const Eigen::Matrix4f &transformation,
    const float trans_eps
) {
    // a. translation magnitude -- norm:
    float translation_magnitude = transformation.block<3, 1>(0, 3).norm();
    // b. rotation magnitude -- angle:
    float rotation_magnitude = fabs(
        acos(
            (transformation.block<3, 3>(0, 0).trace() - 1.0f) / 2.0f
        )
    );

    return (
        (translation_magnitude > trans_eps) || 
        (rotation_magnitude > trans_eps)
    );
}

} // namespace lidar_localization