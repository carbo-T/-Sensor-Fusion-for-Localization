/*
 * @Description: ceres residual block for LIO IMU pre-integration measurement
 * @Author: Ge Yao
 * @Date: 2020-11-29 15:47:49
 */
#ifndef LIDAR_LOCALIZATION_MODELS_SLIDING_WINDOW_FACTOR_PRVAG_IMU_PRE_INTEGRATION_HPP_
#define LIDAR_LOCALIZATION_MODELS_SLIDING_WINDOW_FACTOR_PRVAG_IMU_PRE_INTEGRATION_HPP_

#include <ceres/ceres.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <sophus/so3.hpp>

#include "glog/logging.h"

namespace sliding_window {

class FactorPRVAGIMUPreIntegration : public ceres::SizedCostFunction<15, 15, 15> {
public:
	static const int INDEX_P = 0;
	static const int INDEX_R = 3;
	static const int INDEX_V = 6;
	static const int INDEX_A = 9;
	static const int INDEX_G = 12;

  FactorPRVAGIMUPreIntegration(void) {};

	void SetT(const double &T) {
		T_ = T;
	}

	void SetGravitiy(const Eigen::Vector3d &g) {
		g_ = g;
	}

  void SetMeasurement(const Eigen::VectorXd &m) {
		m_ = m;
	}

  void SetInformation(const Eigen::MatrixXd &I) {
    I_ = I;
  }

	void SetJacobian(const Eigen::MatrixXd &J) {
		J_ = J;
	}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    //
    // parse parameters:
    //
    // a. pose i
    Eigen::Map<const Eigen::Vector3d>     pos_i(&parameters[0][INDEX_P]);
    Eigen::Map<const Eigen::Vector3d> log_ori_i(&parameters[0][INDEX_R]);
    const Sophus::SO3d                    ori_i = Sophus::SO3d::exp(log_ori_i);
		Eigen::Map<const Eigen::Vector3d>     vel_i(&parameters[0][INDEX_V]);
		Eigen::Map<const Eigen::Vector3d>     b_a_i(&parameters[0][INDEX_A]);
		Eigen::Map<const Eigen::Vector3d>     b_g_i(&parameters[0][INDEX_G]);

    // b. pose j
    Eigen::Map<const Eigen::Vector3d>     pos_j(&parameters[1][INDEX_P]);
    Eigen::Map<const Eigen::Vector3d> log_ori_j(&parameters[1][INDEX_R]);
    const Sophus::SO3d                    ori_j = Sophus::SO3d::exp(log_ori_j);
		Eigen::Map<const Eigen::Vector3d>     vel_j(&parameters[1][INDEX_V]);
		Eigen::Map<const Eigen::Vector3d>     b_a_j(&parameters[1][INDEX_A]);
		Eigen::Map<const Eigen::Vector3d>     b_g_j(&parameters[1][INDEX_G]);

    //
    // parse measurement:
    // 
		const Eigen::Vector3d &alpha_ij = m_.block<3, 1>(INDEX_P, 0);
		const Eigen::Vector3d &theta_ij = m_.block<3, 1>(INDEX_R, 0);
		const Eigen::Vector3d  &beta_ij = m_.block<3, 1>(INDEX_V, 0);

    //
    // TODO: get square root of information matrix:
    // from vins-mono, llt分解获得下三角矩阵L与转置Lt
    Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(I_).matrixL().transpose();

    //
    // TODO: compute residual:第九章定义预积分残差计算
    //
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual_(residuals);
    residual_.block<3, 1>(INDEX_P, 0) = ori_i.inverse() * (pos_j - pos_i - vel_i * T_ + 0.5 * g_ * T_ * T_) - alpha_ij;
    residual_.block<3, 1>(INDEX_R, 0) = (Sophus::SO3d::exp(theta_ij).inverse()*(ori_i.inverse()*ori_j)).log();
    residual_.block<3, 1>(INDEX_V, 0) = ori_i.inverse()*(vel_j-vel_i+g_*T_)-beta_ij;
    residual_.block<3, 1>(INDEX_A, 0) = b_a_j - b_a_i;
    residual_.block<3, 1>(INDEX_G, 0) = b_g_j - b_g_i;

    //
    // TODO: compute jacobians:预积分雅可比
    //
    if ( jacobians ) {
      // compute shared intermediate results:提取部分偏导简化书写
      Eigen::Matrix3d dp_dba = J_.block<3,3>(INDEX_P, INDEX_A);
      Eigen::Matrix3d dp_dbg = J_.block<3,3>(INDEX_P, INDEX_G);

      Eigen::Matrix3d dq_dbg = J_.block<3,3>(INDEX_R, INDEX_G);
      
      Eigen::Matrix3d dv_dba = J_.block<3,3>(INDEX_V, INDEX_A);
      Eigen::Matrix3d dv_dbg = J_.block<3,3>(INDEX_V, INDEX_G);

      Eigen::Matrix3d jr_inv = JacobianRInv( residual_.block<3, 1>(INDEX_R, 0));

      if ( jacobians[0] ) {
        Eigen::Map<Eigen::Matrix<double, 15, 15, Eigen::RowMajor>> J_PRVAG_i(jacobians[0]);
        J_PRVAG_i.setZero();

        // a. residual, position:dp_dpi, dp_dri, dp_dvi, dp_dbai, dp_dbgi
        J_PRVAG_i.block<3,3>(INDEX_P, INDEX_P) = - ori_i.inverse().matrix();
        J_PRVAG_i.block<3,3>(INDEX_P, INDEX_R) = Sophus::SO3d::hat(ori_i.inverse() *(pos_j-pos_i-vel_i*T_+0.5*g_*T_*T_));
        J_PRVAG_i.block<3,3>(INDEX_P, INDEX_V) = - ori_i.inverse().matrix()*T_;
        J_PRVAG_i.block<3,3>(INDEX_P, INDEX_A) = - dp_dba;
        J_PRVAG_i.block<3,3>(INDEX_P, INDEX_G) = - dp_dbg;

        // b. residual, orientation:
        J_PRVAG_i.block<3,3>(INDEX_R, INDEX_R) = -jr_inv * ori_j.inverse().matrix()*ori_i.matrix();
        J_PRVAG_i.block<3,3>(INDEX_R, INDEX_G) = -jr_inv * Sophus::SO3d::exp(residual_.block<3, 1>(INDEX_R, 0)).matrix().inverse() * dq_dbg;

        // c. residual, velocity:
        J_PRVAG_i.block<3,3>(INDEX_V, INDEX_R) = Sophus::SO3d::hat(ori_i.inverse().matrix()*(vel_j-vel_i+g_*T_));
        J_PRVAG_i.block<3,3>(INDEX_V, INDEX_V) = - ori_i.inverse().matrix();
        J_PRVAG_i.block<3,3>(INDEX_V, INDEX_A) = - dv_dba;
        J_PRVAG_i.block<3,3>(INDEX_V, INDEX_G) = - dv_dbg;

        // d. residual, bias accel:
        J_PRVAG_i.block<3,3>(INDEX_A, INDEX_A) = - Eigen::Matrix3d::Identity();

        // e. residual, bias gyro:
        J_PRVAG_i.block<3,3>(INDEX_G, INDEX_G) = - Eigen::Matrix3d::Identity();

        // J_PRVAG_i = sqrt_info * J_PRVAG_i;
      }

      if ( jacobians[1] ) {
        Eigen::Map<Eigen::Matrix<double, 15, 15, Eigen::RowMajor>> J_PRVAG_j(jacobians[1]);
        J_PRVAG_j.setZero();

        // a. residual, position:dp_dpi, dp_dri, dp_dvi, dp_dbai, dp_dbgi
        J_PRVAG_j.block<3,3>(INDEX_P, INDEX_P) = ori_i.inverse().matrix();

        // b. residual, orientation:
        J_PRVAG_j.block<3,3>(INDEX_R, INDEX_R) = jr_inv;

        // c. residual, velocity:
        J_PRVAG_j.block<3,3>(INDEX_V, INDEX_V) = ori_i.inverse().matrix();

        // d. residual, bias accel:
        J_PRVAG_j.block<3,3>(INDEX_A, INDEX_A) = Eigen::Matrix3d::Identity();

        // e. residual, bias gyro:
        J_PRVAG_j.block<3,3>(INDEX_G, INDEX_G) = Eigen::Matrix3d::Identity();

        // J_PRVAG_j = sqrt_info * J_PRVAG_j;
      }
    }
    //
    // TODO: correct residual by square root of information matrix:
    //
    residual_ = sqrt_info * residual_;
    
    return true;
  }

private:
  static Eigen::Matrix3d JacobianRInv(const Eigen::Vector3d &w) {
      Eigen::Matrix3d J_r_inv = Eigen::Matrix3d::Identity();

      double theta = w.norm();

      if ( theta > 1e-5 ) {
          Eigen::Vector3d k = w.normalized();
          Eigen::Matrix3d K = Sophus::SO3d::hat(k);
          double theta_half = 0.5 * theta;
          double cot_theta = 1.0 / tan(theta_half);

          // J_r_inv = theta_half * cot_theta * J_r_inv + (1.0 - theta_half * cot_theta) * k * k.transpose() + theta_half * K;

          J_r_inv = J_r_inv 
                    + 0.5 * K
                    + (1.0 - (1.0 + std::cos(theta)) * theta / (2.0 * std::sin(theta))) * K * K;
      }

      return J_r_inv;
  }

   static Eigen::Matrix3d JacobianR(const Eigen::Vector3d &w) {
      Eigen::Matrix3d J_r = Eigen::Matrix3d::Identity();

      double theta = w.norm();

      if ( theta > 1e-5 ) {
          Eigen::Vector3d k = w.normalized();
          Eigen::Matrix3d K = Sophus::SO3d::hat(k);
          
          J_r = sin(theta) / theta * Eigen::Matrix3d::Identity() * (1.0-sin(theta) / theta) * k * k.transpose() - (1.0 - cos(theta)) / theta * K;
      }

      return J_r;
  }
  
	double T_ = 0.0;

	Eigen::Vector3d g_ = Eigen::Vector3d::Zero();

  Eigen::VectorXd m_;
  Eigen::MatrixXd I_;

	Eigen::MatrixXd J_;
};

} // namespace sliding_window

#endif // LIDAR_LOCALIZATION_MODELS_SLIDING_WINDOW_FACTOR_PRVAG_IMU_PRE_INTEGRATION_HPP_
