/*
 * @Description: 数据预处理模块，包括时间同步、点云去畸变等
 * @Author: Ren Qian
 * @Date: 2020-02-10 08:38:42
 */
#include "lidar_localization/data_pretreat/data_pretreat_flow.hpp"

#include "glog/logging.h"
#include "lidar_localization/global_defination/global_defination.h"

#ifndef HK_DATA
#define HK_DATA (1)
#endif

Eigen::Vector3d g_ = Eigen::Vector3d(0,0,9.78772);
double g_last_time=0.0;
Eigen::Vector3f last_enu=Eigen::Vector3f::Identity();

namespace lidar_localization {
DataPretreatFlow::DataPretreatFlow(ros::NodeHandle& nh, std::string cloud_topic) {
    #if HK_DATA==0
    // subscribers:
    // a. velodyne measurement:
    cloud_sub_ptr_ = std::make_shared<CloudSubscriber>(nh, "/kitti/velo/pointcloud", 100000);
    // b. OXTS IMU:
    imu_sub_ptr_ = std::make_shared<IMUSubscriber>(nh, "/kitti/oxts/imu", 1000000);
    // c. OXTS velocity:
    velocity_sub_ptr_ = std::make_shared<VelocitySubscriber>(nh, "/kitti/oxts/gps/vel", 1000000);
    // d. OXTS GNSS:
    gnss_sub_ptr_ = std::make_shared<GNSSSubscriber>(nh, "/kitti/oxts/gps/fix", 1000000);
    lidar_to_imu_ptr_ = std::make_shared<TFListener>(nh, "/imu_link", "/velo_link");
    #else
    cloud_sub_ptr_ = std::make_shared<CloudSubscriber>(nh, "/velodyne_points", 100000);
    imu_sub_ptr_ = std::make_shared<IMUSubscriber>(nh, "/imu/data", 1000000);
    gnss_sub_ptr_ = std::make_shared<GNSSSubscriber>(nh, "/navsat/fix", 1000000);
    lidar_to_imu_ptr_ = std::make_shared<TFListener>(nh, "/imu", "/velo_link");
    #endif

    // publishers:
    cloud_pub_ptr_ = std::make_shared<CloudPublisher>(nh, cloud_topic, "/velo_link", 100);
    imu_pub_ptr_ = std::make_shared<IMUPublisher>(nh, "/synced_imu", "/imu_link", 100);
    pos_vel_pub_ptr_ = std::make_shared<PosVelPublisher>(nh, "/synced_pos_vel", "/map", "/imu_link", 100);
    gnss_pub_ptr_ = std::make_shared<OdometryPublisher>(nh, "/synced_gnss", "/map", "/velo_link", 100);

    // motion compensation for lidar measurement:
    distortion_adjust_ptr_ = std::make_shared<DistortionAdjust>();
}

bool DataPretreatFlow::Run() {
    if (!ReadData())
        return false;
// std::cout<<"000"<<std::endl;
    if (!InitCalibration()) 
        return false;
// std::cout<<"111"<<std::endl;
    if (!InitGNSS())
        return false;
// std::cout<<"222"<<std::endl;
    while(HasData()) {
// std::cout<<"333"<<std::endl;
        if (!ValidData())
            continue;
// std::cout<<"444"<<std::endl;
        TransformData();
        PublishData();
    }

    return true;
}

bool DataPretreatFlow::ReadData() {
    static std::deque<IMUData> unsynced_imu_;
    static std::deque<VelocityData> unsynced_velocity_;
    static std::deque<GNSSData> unsynced_gnss_;

    // fetch lidar measurements from buffer:
    cloud_sub_ptr_->ParseData(cloud_data_buff_);
    imu_sub_ptr_->ParseData(unsynced_imu_);
    #if HK_DATA==0
    velocity_sub_ptr_->ParseData(unsynced_velocity_);
    #else
    static VelocityData last_vel;
    for (size_t i = 0; i < unsynced_imu_.size(); i++)
    {
        VelocityData t_vel;
        t_vel.time = unsynced_imu_.at(i).time;
        t_vel.angular_velocity.x = unsynced_imu_.at(i).angular_velocity.x;
        t_vel.angular_velocity.y = unsynced_imu_.at(i).angular_velocity.y;
        t_vel.angular_velocity.z = unsynced_imu_.at(i).angular_velocity.z;

        // Eigen::Quaterniond q(unsynced_imu_.at(i).orientation.w, unsynced_imu_.at(i).orientation.x, unsynced_imu_.at(i).orientation.y, unsynced_imu_.at(i).orientation.z);
        // Eigen::Matrix3d matrix = q.matrix();
        // Eigen::Vector3d aligned_gravity = matrix * g_;
        // if (last_vel.time > 0.0)
        // {
        //     double dt = t_vel.time - last_vel.time;
        //     t_vel.linear_velocity.x = last_vel.linear_velocity.x + (unsynced_imu_.at(i).linear_acceleration.x-aligned_gravity.x())*dt;
        //     t_vel.linear_velocity.y = last_vel.linear_velocity.y + (unsynced_imu_.at(i).linear_acceleration.y-aligned_gravity.y())*dt;
        //     t_vel.linear_velocity.z = last_vel.linear_velocity.z + (unsynced_imu_.at(i).linear_acceleration.z-aligned_gravity.z())*dt;
        // }
        last_vel = t_vel;
        unsynced_velocity_.push_back(last_vel);
    }
    #endif

    gnss_sub_ptr_->ParseData(unsynced_gnss_);

    if (cloud_data_buff_.size() == 0)
        return false;

    // use timestamp of lidar measurement as reference:
    double cloud_time = cloud_data_buff_.front().time;
    // sync IMU, velocity and GNSS with lidar measurement:
    // find the two closest measurement around lidar measurement time
    // then use linear interpolation to generate synced measurement:
    bool valid_imu = IMUData::SyncData(unsynced_imu_, imu_data_buff_, cloud_time);
    // #if HK_DATA==0
    bool valid_velocity = VelocityData::SyncData(unsynced_velocity_, velocity_data_buff_, cloud_time);
    // #endif
    bool valid_gnss = GNSSData::SyncData(unsynced_gnss_, gnss_data_buff_, cloud_time);

    // only mark lidar as 'inited' when all the three sensors are synced:
    static bool sensor_inited = false;
    if (!sensor_inited) {
        // #if HK_DATA==0
        if (!valid_imu || !valid_velocity || !valid_gnss) {
        // #else
        // // gps 数据频率变化，kitti 10Hz frameid imu_link，hk_data 1Hz frameid gps，暂时取消gps合法性校验
        // if (!valid_imu || !valid_gnss) {
        // #endif
            cloud_data_buff_.pop_front();
            // LOG(INFO)<<(valid_imu?"imu valid ":"imu invalid ")<<(valid_velocity?"vel valid ":"vel invalid ")<<(valid_gnss?"gnss valid ":"gnss invalid ")<<unsynced_gnss_.size();
            return false;
        }
        sensor_inited = true;
    }

    return true;
}

bool DataPretreatFlow::InitCalibration() {
    // lookup imu pose in lidar frame:
    static bool calibration_received = false;
    if (!calibration_received) {
        if (lidar_to_imu_ptr_->LookupData(lidar_to_imu_)) {
            calibration_received = true;
        }
    }

    return calibration_received;
}

bool DataPretreatFlow::InitGNSS() {
    static bool gnss_inited = false;
    if (!gnss_inited) {
        GNSSData gnss_data = gnss_data_buff_.front();
        gnss_data.InitOriginPosition();
        gnss_inited = true;
    }

    return gnss_inited;
}

bool DataPretreatFlow::HasData() {
    if (cloud_data_buff_.size() == 0)
        return false;
    if (imu_data_buff_.size() == 0)
        return false;
    // #if HK_DATA==0
    if (velocity_data_buff_.size() == 0)
        return false;
    // #endif
    if (gnss_data_buff_.size() == 0)
        return false;

    return true;
}

bool DataPretreatFlow::ValidData() {
    current_cloud_data_ = cloud_data_buff_.front();
    current_imu_data_ = imu_data_buff_.front();
    // #if HK_DATA==0
    current_velocity_data_ = velocity_data_buff_.front();
    // #endif
    current_gnss_data_ = gnss_data_buff_.front();

    double diff_imu_time = current_cloud_data_.time - current_imu_data_.time;
    double diff_gnss_time = current_cloud_data_.time - current_gnss_data_.time;
    //
    // this check assumes the frequency of lidar is 10Hz:
    //
    //  #if HK_DATA==0
    double diff_velocity_time = current_cloud_data_.time - current_velocity_data_.time;
    if (diff_imu_time < -0.05 || diff_velocity_time < -0.05 || diff_gnss_time < -0.05) {
    // #else
    // // LOG(WARNING)<<"cloud imu gnss t: "<<current_cloud_data_.time-1.55645e9<<", "<<current_imu_data_.time-1.55645e9<<", "<<current_gnss_data_.time-1.55645e9;
    // if (diff_imu_time < -0.05 || diff_gnss_time < -0.05) {
    // #endif
        cloud_data_buff_.pop_front();
        return false;
    }

    if (diff_imu_time > 0.05) {
        imu_data_buff_.pop_front();
        return false;
    }

    // #if HK_DATA==0
    if (diff_velocity_time > 0.05) {
        velocity_data_buff_.pop_front();
        return false;
    }
    // #endif

    if (diff_gnss_time > 0.05) {
        gnss_data_buff_.pop_front();
        return false;
    }

    cloud_data_buff_.pop_front();
    imu_data_buff_.pop_front();
    // #if HK_DATA==0
    velocity_data_buff_.pop_front();
    // #endif
    gnss_data_buff_.pop_front();

    return true;
}

bool DataPretreatFlow::TransformData() {
    // a. get reference pose:
    gnss_pose_ = Eigen::Matrix4f::Identity();
    // get position from GNSS
    current_gnss_data_.UpdateXYZ();
    gnss_pose_(0,3) = current_gnss_data_.local_E;
    gnss_pose_(1,3) = current_gnss_data_.local_N;
    gnss_pose_(2,3) = current_gnss_data_.local_U;
    // get orientation from IMU:
    gnss_pose_.block<3,3>(0,0) = current_imu_data_.GetOrientationMatrix();
    // this is lidar pose in GNSS/map frame:
    gnss_pose_ *= lidar_to_imu_;

    float dt = 1.0;
    if (g_last_time <= 0.0)
    {
        g_last_time = current_gnss_data_.time;
    }
    else
    {
        dt = current_gnss_data_.time - g_last_time;
        Eigen::Vector3f local_vel = current_imu_data_.GetOrientationMatrix().transpose() * (gnss_pose_.block<3, 1>(0, 3) - last_enu) / dt;
        current_velocity_data_.linear_velocity.x = local_vel.x();
        current_velocity_data_.linear_velocity.y = local_vel.y();
        current_velocity_data_.linear_velocity.z = local_vel.z();
    }
    char buf[255];
    sprintf(buf, "time: %.2f, pos: %.7f %.7f %.7f\nang:%.3f %.3f %.3f, lin:%.3f %.3f %.3f\n", dt, last_enu.x(), last_enu.y(), last_enu.z(),
            current_velocity_data_.angular_velocity.x, current_velocity_data_.angular_velocity.y, current_velocity_data_.angular_velocity.z,
            current_velocity_data_.linear_velocity.x, current_velocity_data_.linear_velocity.y, current_velocity_data_.linear_velocity.z);
    std::cout << buf << std::endl;
    last_enu << current_gnss_data_.local_E, current_gnss_data_.local_N, current_gnss_data_.local_U;
    g_last_time = current_gnss_data_.time;

    // b. set synced pos vel
    pos_vel_.pos.x() = current_gnss_data_.local_E;
    pos_vel_.pos.y() = current_gnss_data_.local_N;
    pos_vel_.pos.z() = current_gnss_data_.local_U;

    // #if HK_DATA==0
    pos_vel_.vel.x() = current_velocity_data_.linear_velocity.x;
    pos_vel_.vel.y() = current_velocity_data_.linear_velocity.y;
    pos_vel_.vel.z() = current_velocity_data_.linear_velocity.z;

    // c. motion compensation for lidar measurements:
    current_velocity_data_.TransformCoordinate(lidar_to_imu_);
    distortion_adjust_ptr_->SetMotionInfo(0.1, current_velocity_data_);
    distortion_adjust_ptr_->AdjustCloud(current_cloud_data_.cloud_ptr, current_cloud_data_.cloud_ptr);
    // #endif
    return true;
}

bool DataPretreatFlow::PublishData() {
    cloud_pub_ptr_->Publish(current_cloud_data_.cloud_ptr, current_cloud_data_.time);
    imu_pub_ptr_->Publish(current_imu_data_, current_cloud_data_.time);

    pos_vel_pub_ptr_->Publish(pos_vel_, current_cloud_data_.time);
    
    //
    // this synced odometry has the following info:
    //
    // a. lidar frame's pose in map
    // b. lidar frame's velocity
    // gnss_pose_.block<3, 1>(0, 3) += Eigen::Vector3f(15.0f, 18.0f, 0.7f);
    gnss_pub_ptr_->Publish(gnss_pose_, current_velocity_data_, current_cloud_data_.time);

    
    return true;
}
}