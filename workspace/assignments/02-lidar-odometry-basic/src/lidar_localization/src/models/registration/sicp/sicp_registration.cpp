/*
 * @Description: ICP SVD lidar odometry
 * @Author: Ge Yao
 * @Date: 2020-10-24 21:46:45
 */

#include <pcl/common/transforms.h>

#include <Eigen/Dense>

#include "glog/logging.h"

#include "lidar_localization/models/registration/sicp/ICP.h"

#include "lidar_localization/models/registration/sicp/scip_registration.hpp"

namespace lidar_localization {
SICP::Parameters g_sicp_params_;

SICPRegistration::SICPRegistration(
    const YAML::Node& node
) {
    // parse params:
    g_sicp_params_.p = node["p"].as<float>();
    g_sicp_params_.mu = node["mu"].as<float>();
    g_sicp_params_.alpha = node["alpha"].as<float>();
    g_sicp_params_.max_mu = node["max_mu"].as<float>();
    g_sicp_params_.max_icp = node["max_icp"].as<int>();
    g_sicp_params_.max_outer = node["max_outer"].as<int>();
    g_sicp_params_.max_inner = node["max_inner"].as<int>();
    g_sicp_params_.stop = node["stop"].as<float>();
    
}

bool SICPRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target) {
    input_target_ = input_target;

    return true;
}

bool SICPRegistration::ScanMatch(
    const CloudData::CLOUD_PTR& input_source, 
    const Eigen::Matrix4f& predict_pose, 
    CloudData::CLOUD_PTR& result_cloud_ptr,
    Eigen::Matrix4f& result_pose
) {
    input_source_ = input_source;

    // pre-process input source:
    CloudData::CLOUD_PTR transformed_input_source(new CloudData::CLOUD());
    pcl::transformPointCloud(*input_source_, *transformed_input_source, predict_pose);

    //
    // TODO: second option -- adapt existing implementation
    //
    // TODO: format inputs for SICP:
    Eigen::Matrix3Xd X ( 3, input_target_->size() ); // source, transformed
    Eigen::Matrix3Xd Y ( 3, transformed_input_source->size() ); // target
    
    // init estimation:
    transformation_.setIdentity();

    for(int i = 0; i < input_target_->size(); i++)
    {
      X(0,i) = input_target_->points[i].x;
      X(1,i) = input_target_->points[i].y;
      X(2,i) = input_target_->points[i].z;
    }
    for(int i = 0; i < transformed_input_source->size(); i++)
    {
      Y(0,i) = transformed_input_source->points[i].x;
      Y(1,i) = transformed_input_source->points[i].y;
      Y(2,i) = transformed_input_source->points[i].z;
    }
    
    // std::cout<<"sicp registration start, source: "<<transformed_input_source->size()<<std::endl;
    // TODO: SICP registration:
    Eigen::Affine3d t_transform = SICP::point_to_point ( Y, X, g_sicp_params_ ); // sparse ICP
    transformation_ = t_transform.matrix().cast<float>();
    // std::cout<<"sicp registration get trans"<<std::endl;

    // set output:
    result_pose = transformation_ * predict_pose;
    pcl::transformPointCloud(*input_source_, *result_cloud_ptr, result_pose);
    
    return true;
}

} // namespace lidar_localization