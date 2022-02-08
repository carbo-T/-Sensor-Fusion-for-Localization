/*
 * @Description: IMU integration activity
 * @Author: Ge Yao
 * @Date: 2020-11-10 14:25:03
 */
#include <cmath>

#include "imu_integration/estimator/activity.hpp"
#include "glog/logging.h"

// #define MID_VALUE
namespace imu_integration {

namespace estimator {

Activity::Activity(void) 
    : private_nh_("~"), 
    initialized_(false),
    // gravity acceleration:
    G_(0, 0, -9.81),
    // angular velocity bias:
    angular_vel_bias_(0.0, 0.0, 0.0),
    // linear acceleration bias:
    linear_acc_bias_(0.0, 0.0, 0.0)
{}

void Activity::Init(void) {
    // parse IMU config:
    private_nh_.param("imu/topic_name", imu_config_.topic_name, std::string("/sim/sensor/imu"));
    imu_sub_ptr_ = std::make_shared<IMUSubscriber>(private_nh_, imu_config_.topic_name, 1000000);

    // a. gravity constant:
    private_nh_.param("imu/gravity/x", imu_config_.gravity.x,  0.0);
    private_nh_.param("imu/gravity/y", imu_config_.gravity.y,  0.0);
    private_nh_.param("imu/gravity/z", imu_config_.gravity.z, -9.81);
    G_.x() = imu_config_.gravity.x;
    G_.y() = imu_config_.gravity.y;
    G_.z() = imu_config_.gravity.z;

    // b. angular velocity bias:
    private_nh_.param("imu/bias/angular_velocity/x", imu_config_.bias.angular_velocity.x,  0.0);
    private_nh_.param("imu/bias/angular_velocity/y", imu_config_.bias.angular_velocity.y,  0.0);
    private_nh_.param("imu/bias/angular_velocity/z", imu_config_.bias.angular_velocity.z,  0.0);
    angular_vel_bias_.x() = imu_config_.bias.angular_velocity.x;
    angular_vel_bias_.y() = imu_config_.bias.angular_velocity.y;
    angular_vel_bias_.z() = imu_config_.bias.angular_velocity.z;

    // c. linear acceleration bias:
    private_nh_.param("imu/bias/linear_acceleration/x", imu_config_.bias.linear_acceleration.x,  0.0);
    private_nh_.param("imu/bias/linear_acceleration/y", imu_config_.bias.linear_acceleration.y,  0.0);
    private_nh_.param("imu/bias/linear_acceleration/z", imu_config_.bias.linear_acceleration.z,  0.0);
    linear_acc_bias_.x() = imu_config_.bias.linear_acceleration.x;
    linear_acc_bias_.y() = imu_config_.bias.linear_acceleration.y;
    linear_acc_bias_.z() = imu_config_.bias.linear_acceleration.z;

    // parse odom config:
    private_nh_.param("pose/frame_id", odom_config_.frame_id, std::string("inertial"));
    private_nh_.param("pose/topic_name/ground_truth", odom_config_.topic_name.ground_truth, std::string("/pose/ground_truth"));
    private_nh_.param("pose/topic_name/estimation", odom_config_.topic_name.estimation, std::string("/pose/estimation"));

    odom_ground_truth_sub_ptr = std::make_shared<OdomSubscriber>(private_nh_, odom_config_.topic_name.ground_truth, 1000000);
    odom_estimation_pub_ = private_nh_.advertise<nav_msgs::Odometry>(odom_config_.topic_name.estimation, 500);

    std::string filename = __FILE__;
    ground_truth.open(filename.substr(0, filename.find_last_of('/'))+"/ground_truth.txt", std::ios::out);
    estimated_traj.open(filename.substr(0, filename.find_last_of('/'))+"/estimated_traj.txt", std::ios::out);
    if(!ground_truth.is_open() || !estimated_traj.is_open())
    {
        std::cout<<"open file to save traj failed!!!"<<std::endl;
    }
}

bool Activity::Run(void) {
    if (!ReadData())
        return false;

    while(HasData()) {
        if (UpdatePose()) {
            PublishPose();
        }
    }

    return true;
}

bool Activity::ReadData(void) {
    // fetch IMU measurements into buffer:
    imu_sub_ptr_->ParseData(imu_data_buff_);

    if (static_cast<size_t>(0) == imu_data_buff_.size())
        return false;

    // changed by yct
        odom_ground_truth_sub_ptr->ParseData(odom_data_buff_);
    if (!initialized_) {

        if (static_cast<size_t>(0) == odom_data_buff_.size())
            return false;
    }

    return true;
}

bool Activity::HasData(void) {
    if (imu_data_buff_.size() < static_cast<size_t>(3))
        return false;

    if (
        !initialized_ && 
        static_cast<size_t>(0) == odom_data_buff_.size()
    ) {
        return false;
    }

    return true;
}

bool Activity::UpdatePose(void) {
    // static size_t odom_index = 0;
    if (!initialized_) {
        // use the latest measurement for initialization:
        // stamped pos & vel
        OdomData &odom_data = odom_data_buff_.back();
        // stamped lin_acc & ang_vel
        IMUData imu_data = imu_data_buff_.back();

        pose_ = odom_data.pose;
        vel_ = odom_data.vel;

        initialized_ = true;

        odom_data_buff_.clear();
        imu_data_buff_.clear();
        // odom_index = 0;

        // keep the latest IMU measurement for mid-value integration:
        imu_data_buff_.push_back(imu_data);
        odom_data_buff_.push_back(odom_data);
    } else {
        static int save_line_count=0;
        //
        // TODO: implement your estimation here
        // 初始化后队列超过三个元素才会到此
        // get deltas:
        Eigen::Vector3d t_angular_delta=Eigen::Vector3d::Zero();
        if(!GetAngularDelta(1, 0, t_angular_delta))
        {
            return false;
        }

        // update orientation:
        Eigen::Matrix3d t_R_curr=Eigen::Matrix3d::Identity(), t_R_prev=Eigen::Matrix3d::Identity();
        // 存在角度差异再更新旋转矩阵
        if(t_angular_delta.norm()>0){
            UpdateOrientation(t_angular_delta, t_R_curr, t_R_prev);
        }

        // get velocity delta:
        double delta_t=0.0;
        Eigen::Vector3d velocity_delta=Eigen::Vector3d::Zero();
        if(!GetVelocityDelta(1, 0, t_R_curr, t_R_prev, delta_t, velocity_delta))
        {
            return false;
        }

        // update position:
        UpdatePosition(delta_t, velocity_delta);

        // estimate difference
        // std::cout<<imu_data_buff_.size()<<", "<<odom_data_buff_.size()<<std::endl;
        if (odom_data_buff_.size() > 0 && ground_truth.is_open() && estimated_traj.is_open())
        {
            if (save_line_count > 250000)
            {
                std::cout << "------------------ 2500 lines saved!!! --------------------" << std::endl;
            }
            else
            {
                // Eigen::Vector3d pos_gt = odom_data_buff_.at(0).pose.block<3, 1>(0, 3);
                // Eigen::Vector3d pos_est = pose_.block<3, 1>(0, 3);

                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 4; ++j)
                    {
                        estimated_traj << pose_(i, j);
                        ground_truth << odom_data_buff_.at(0).pose(i, j);
                        if (i == 2 && j == 3)
                        {
                            ground_truth << std::endl;
                            estimated_traj << std::endl;
                        }
                        else
                        {
                            ground_truth << " ";
                            estimated_traj << " ";
                        }
                    }
                }
                save_line_count++;

                odom_data_buff_.pop_front();
            }
        }

        // move forward -- 
        // NOTE: this is NOT fixed. you should update your buffer according to the method of your choice:
        imu_data_buff_.pop_front();
        // odom_index++;
    }
    
    return true;
}

bool Activity::PublishPose() {
    // a. set header:
    message_odom_.header.stamp = ros::Time::now();
    message_odom_.header.frame_id = odom_config_.frame_id;
    
    // b. set child frame id:
    message_odom_.child_frame_id = odom_config_.frame_id;

    // b. set orientation:
    Eigen::Quaterniond q(pose_.block<3, 3>(0, 0));
    message_odom_.pose.pose.orientation.x = q.x();
    message_odom_.pose.pose.orientation.y = q.y();
    message_odom_.pose.pose.orientation.z = q.z();
    message_odom_.pose.pose.orientation.w = q.w();

    // c. set position:
    Eigen::Vector3d t = pose_.block<3, 1>(0, 3);
    message_odom_.pose.pose.position.x = t.x();
    message_odom_.pose.pose.position.y = t.y();
    message_odom_.pose.pose.position.z = t.z();  

    // d. set velocity:
    message_odom_.twist.twist.linear.x = vel_.x();
    message_odom_.twist.twist.linear.y = vel_.y();
    message_odom_.twist.twist.linear.z = vel_.z(); 

    odom_estimation_pub_.publish(message_odom_);

    return true;
}

/**
 * @brief  get unbiased angular velocity in body frame
 * @param  angular_vel, angular velocity measurement
 * @return unbiased angular velocity in body frame
 */
inline Eigen::Vector3d Activity::GetUnbiasedAngularVel(const Eigen::Vector3d &angular_vel) {
    return angular_vel - angular_vel_bias_;
}

/**
 * @brief  get unbiased linear acceleration in navigation frame
 * @param  linear_acc, linear acceleration measurement
 * @param  R, corresponding orientation of measurement
 * @return unbiased linear acceleration in navigation frame
 */
inline Eigen::Vector3d Activity::GetUnbiasedLinearAcc(
    const Eigen::Vector3d &linear_acc,
    const Eigen::Matrix3d &R
) {
    return R*(linear_acc - linear_acc_bias_) - G_;
}

/**
 * @brief  get angular delta
 * @param  index_curr, current imu measurement buffer index
 * @param  index_prev, previous imu measurement buffer index
 * @param  angular_delta, angular delta output
 * @return true if success false otherwise
 */
bool Activity::GetAngularDelta(
    const size_t index_curr, const size_t index_prev,
    Eigen::Vector3d &angular_delta
) {
    //
    // TODO: this could be a helper routine for your own implementation
    //
    if (
        index_curr <= index_prev ||
        imu_data_buff_.size() <= index_curr
    ) {
        return false;
    }

    const IMUData &imu_data_curr = imu_data_buff_.at(index_curr);
    const IMUData &imu_data_prev = imu_data_buff_.at(index_prev);

    double delta_t = imu_data_curr.time - imu_data_prev.time;

    Eigen::Vector3d angular_vel_curr = GetUnbiasedAngularVel(imu_data_curr.angular_velocity);
    Eigen::Vector3d angular_vel_prev = GetUnbiasedAngularVel(imu_data_prev.angular_velocity);

    #ifdef MID_VALUE
    // 姿态差异中值法
    angular_delta = 0.5*delta_t*(angular_vel_curr + angular_vel_prev);
    #else
    // 姿态差异欧拉法
    angular_delta = delta_t*angular_vel_prev;
    #endif

    return true;
}

/**
 * @brief  get velocity delta
 * @param  index_curr, current imu measurement buffer index
 * @param  index_prev, previous imu measurement buffer index
 * @param  R_curr, corresponding orientation of current imu measurement
 * @param  R_prev, corresponding orientation of previous imu measurement
 * @param  velocity_delta, velocity delta output
 * @return true if success false otherwise
 */
bool Activity::GetVelocityDelta(
    const size_t index_curr, const size_t index_prev,
    const Eigen::Matrix3d &R_curr, const Eigen::Matrix3d &R_prev, 
    double &delta_t, Eigen::Vector3d &velocity_delta
) {
    //
    // TODO: this could be a helper routine for your own implementation
    //
    if (
        index_curr <= index_prev ||
        imu_data_buff_.size() <= index_curr
    ) {
        return false;
    }

    const IMUData &imu_data_curr = imu_data_buff_.at(index_curr);
    const IMUData &imu_data_prev = imu_data_buff_.at(index_prev);

    delta_t = imu_data_curr.time - imu_data_prev.time;

    // std::cout<<(imu_data_curr.linear_acceleration).transpose()<<" ??? \n"<<R_curr<<std::endl;

    Eigen::Vector3d linear_acc_curr = GetUnbiasedLinearAcc(imu_data_curr.linear_acceleration, R_curr);
    Eigen::Vector3d linear_acc_prev = GetUnbiasedLinearAcc(imu_data_prev.linear_acceleration, R_prev);
    
    #ifdef MID_VALUE
    velocity_delta = 0.5*delta_t*(linear_acc_curr + linear_acc_prev);
    #else
    velocity_delta = delta_t * linear_acc_prev;
    #endif

    return true;
}

/**
 * @brief  update orientation with effective rotation angular_delta
 * @param  angular_delta, effective rotation
 * @param  R_curr, current orientation
 * @param  R_prev, previous orientation
 * @return void
 */
void Activity::UpdateOrientation(
    const Eigen::Vector3d &angular_delta,
    Eigen::Matrix3d &R_curr, Eigen::Matrix3d &R_prev
) {
    //
    // TODO: this could be a helper routine for your own implementation
    //
    // magnitude:
    double angular_delta_mag = angular_delta.norm();
    // direction:
    Eigen::Vector3d angular_delta_dir = angular_delta.normalized();

    // build delta q:
    double angular_delta_cos = cos(angular_delta_mag/2.0);
    double angular_delta_sin = sin(angular_delta_mag/2.0);
    Eigen::Quaterniond dq(
        angular_delta_cos, 
        angular_delta_sin*angular_delta_dir.x(), 
        angular_delta_sin*angular_delta_dir.y(), 
        angular_delta_sin*angular_delta_dir.z()
    );
    Eigen::Quaterniond q(pose_.block<3, 3>(0, 0));
    
    // update:
    q = q*dq;
    
    // write back:
    R_prev = pose_.block<3, 3>(0, 0);
    pose_.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    R_curr = pose_.block<3, 3>(0, 0);
}

/**
 * @brief  update orientation with effective velocity change velocity_delta
 * @param  delta_t, timestamp delta 
 * @param  velocity_delta, effective velocity change
 * @return void
 */
void Activity::UpdatePosition(const double &delta_t, const Eigen::Vector3d &velocity_delta) {
    //
    // TODO: this could be a helper routine for your own implementation
    //
    pose_.block<3, 1>(0, 3) += delta_t*vel_ + 0.5*delta_t*velocity_delta;
    vel_ += velocity_delta;
}


} // namespace estimator

} // namespace imu_integration