// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

//
// TODO: implement analytic Jacobians for LOAM residuals in this file
// 
#ifndef ALOAM_FACTOR_HPP
#define ALOAM_FACTOR_HPP

#include <eigen3/Eigen/Dense>

//
// TODO: Sophus is ready to use if you have a good undestanding of Lie algebra.
// 
#include <sophus/so3.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

class PoseSO3Parameterization : public ceres::LocalParameterization {
public:
	
    PoseSO3Parameterization() {}
    virtual ~PoseSO3Parameterization() {}
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const
	{
		Eigen::Quaterniond delta_q;
		getTransformFromSo3(Eigen::Map<const Eigen::Matrix<double,3,1>>(delta), delta_q);
		Eigen::Map<const Eigen::Quaterniond> quater(x);
		Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta);

		quater_plus = delta_q * quater;

		return true;
	}
    virtual bool ComputeJacobian(const double* x, double* jacobian) const
	{
		Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
		(j.topRows(3)).setIdentity();
    	(j.bottomRows(1)).setZero();

		return true;
	}
    virtual int GlobalSize() const { return 4; }
    virtual int LocalSize() const { return 3; }

	void getTransformFromSo3(const Eigen::Matrix<double, 3, 1> &so3, Eigen::Quaterniond &q) const
	{
		Eigen::Vector3d omega(so3.data());
		Eigen::Matrix3d Omega = Sophus::SO3d::exp(so3).matrix();

		double theta = omega.norm();
		double half_theta = 0.5 * theta;

		double imag_factor;
		double real_factor = cos(half_theta);
		if (theta < 1e-10)
		{
			double theta_sq = theta * theta;
			double theta_po4 = theta_sq * theta_sq;
			imag_factor = 0.5 - 0.0208333 * theta_sq + 0.000260417 * theta_po4;
		}
		else
		{
			double sin_half_theta = sin(half_theta);
			imag_factor = sin_half_theta / theta;
		}

		q = Eigen::Quaterniond(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z());
	}
};

struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		residual[0] = nu.x() / de.norm();
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 3, 4, 3>(
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};

// added by yct, ceres analytic edge factor.
// one residual block, 7 parameters ----- quat(xyzw) t(xyz)
class LidarEdgeAnalyticFactor : public ceres::SizedCostFunction<1, 4, 3>
{
	public:
	LidarEdgeAnalyticFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
							Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}
	~LidarEdgeAnalyticFactor() {}

	// parameteres quat(xyzw) t(xyz)
	virtual bool Evaluate(double const *const *parameters,
						  double *residuals,
						  double **jacobians) const
	{
		// handle params
		Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};
		// Eigen::Quaterniond q_last_curr{parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]};
		Eigen::Quaterniond q_identity{1,0,0,0};
		q_last_curr = q_identity.slerp(s, q_last_curr);
		q_last_curr.normalize();
		Eigen::Matrix<double, 3, 1> t_last_curr{s * parameters[1][0], s * parameters[1][1], s * parameters[1][2]};

		// Sophus::SO3d SO3_R(q_last_curr);
		// Eigen::Vector3d so3_r = SO3_R.log();

		// Eigen::Matrix3d rot = Sophus::SO3d::exp(so3_r).matrix();
		// Eigen::Vector3d t{parameters[0][0], parameters[0][1], parameters[0][2]};

		// calculate residuals with so3 and t interpolation
		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpa{last_point_a.x(), last_point_a.y(), last_point_a.z()};
		Eigen::Matrix<double, 3, 1> lpb{last_point_b.x(), last_point_b.y(), last_point_b.z()};

		Eigen::Matrix<double, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<double, 3, 1> de = lpa - lpb;

		double nu_norm = nu.norm()+1e-11;
		double de_norm = de.norm()+1e-11;
		residuals[0] = nu_norm / de_norm;

		// compute jacobians
		if (jacobians != NULL && jacobians[0] != NULL && jacobians[1] != NULL)
		{
			Eigen::Matrix<double,1,3> partial_dabse_de = (nu / (nu_norm)).transpose();
			Eigen::Matrix3d partial_de_p = Sophus::SO3d::hat(de) / de_norm;
			// Eigen::Vector3d partial_de_p_abs;
			// partial_de_p_abs.x() = sqrt(pow(partial_de_p(0, 0), 2) + pow(partial_de_p(1, 0), 2) + pow(partial_de_p(2, 0), 2));
			// partial_de_p_abs.y() = sqrt(pow(partial_de_p(0, 1), 2) + pow(partial_de_p(1, 1), 2) + pow(partial_de_p(2, 1), 2));
			// partial_de_p_abs.z() = sqrt(pow(partial_de_p(0, 2), 2) + pow(partial_de_p(1, 2), 2) + pow(partial_de_p(2, 2), 2));
			Eigen::Matrix3d partial_dp_t = -Sophus::SO3d::hat(q_last_curr * cp);
			Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> J_so3(jacobians[0]);
			J_so3.setZero();
			J_so3.block<1, 3>(0, 0) = partial_dabse_de*partial_de_p * partial_dp_t;
			Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> J_t(jacobians[1]);
			J_t.setZero();
			J_t.block<1, 3>(0, 0) = partial_dabse_de*partial_de_p;
		}
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new LidarEdgeAnalyticFactor(curr_point_, last_point_a_, last_point_b_, s_));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};

struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm);

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};

// added by yct, ceres analytic plane factor
// one residual block, 7 parameters ----- quat(xyzw) t(xyz)
class LidarPlaneAnalyticFactor : public ceres::SizedCostFunction<1, 4, 3>
{
	public:
	LidarPlaneAnalyticFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	~LidarPlaneAnalyticFactor() {}

	// parameteres quat(xyzw) t(xyz)
	virtual bool Evaluate(double const *const *parameters,
						  double *residuals,
						  double **jacobians) const
	{
		// handle params
		Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};
		// Eigen::Quaterniond q_last_curr{parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]};
		Eigen::Quaterniond q_identity{1,0,0,0};
		q_last_curr = q_identity.slerp(s, q_last_curr);
		q_last_curr.normalize();
		Eigen::Matrix<double, 3, 1> t_last_curr{s * parameters[1][0], s * parameters[1][1], s * parameters[1][2]};

		// Sophus::SO3d SO3_R(q_last_curr);
		// Eigen::Vector3d so3_r = SO3_R.log();

		// calculate residuals with so3 and t interpolation
		// current point
		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		// last point j, link with current point
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		// normal of the plane, normalized
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

		Eigen::Matrix<double, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;
		// std::cout<<"so3 t: "<<curr_so3_r.transpose()<<", "<<curr_translation.transpose()<<std::endl;
		double phi1 = (lp - lpj).dot(ljm);
		residuals[0] = std::fabs(phi1);

		// compute jacobians
		if (jacobians != NULL && jacobians[0] != NULL && jacobians[1] != NULL)
		{
			double partial_dabsh_dh = phi1;
			if(residuals[0]!=0)
			{
				partial_dabsh_dh = phi1 / residuals[0];
			}
			Eigen::Matrix<double,1,3> partial_dh_p = ljm.transpose();
			Eigen::Matrix3d partial_dp_t = -Sophus::SO3d::hat(q_last_curr * cp);
			Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> J_so3(jacobians[0]);
			J_so3.setZero();
			J_so3.block<1, 3>(0, 0) = partial_dabsh_dh*partial_dh_p * partial_dp_t;
			Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> J_t(jacobians[1]);
			J_t.setZero();
			J_t.block<1, 3>(0, 0) = partial_dabsh_dh*partial_dh_p;
			// std::cout<<"jso3 :"<<J_so3<<", \njt"<<J_t<<"\nres: "<<phi1<<std::endl;
		}
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new LidarPlaneAnalyticFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};

struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};

// added by yct, ceres analytic plane normal factor
// one residual block, 6 parameters ----- quat(xyzw) t(xyz)
class LidarPlaneNormAnalyticFactor : public ceres::SizedCostFunction<1, 4, 3>
{
	public:
	LidarPlaneNormAnalyticFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	~LidarPlaneNormAnalyticFactor() {}

	// parameteres quat(xyzw) t(xyz)
	virtual bool Evaluate(double const *const *parameters,
						  double *residuals,
						  double **jacobians) const
	{
		// handle params
		Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};
		q_last_curr.normalize();
		// Eigen::Quaterniond q_last_curr{parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]};
		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		// calculate residuals with so3 and t interpolation
		// current point
		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		Eigen::Vector3d norm(plane_unit_norm.x(), plane_unit_norm.y(), plane_unit_norm.z());
		residuals[0] = norm.dot(lp) + negative_OA_dot_norm;

		// compute jacobians
		if (jacobians != NULL && jacobians[0] != NULL && jacobians[1] != NULL)
		{
			double partial_dabsh_dh = residuals[0] / (fabs(residuals[0])+1e-11);
			Eigen::Matrix<double,1,3> partial_dh_p = plane_unit_norm.transpose();
			Eigen::Matrix3d partial_dp_t = -Sophus::SO3d::hat(q_last_curr * cp);
			Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> J_so3(jacobians[0]);
			J_so3.setZero();
			J_so3.block<1, 3>(0, 0) = partial_dabsh_dh*partial_dh_p * partial_dp_t;
			Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> J_t(jacobians[1]);
			J_t.setZero();
			J_t.block<1, 3>(0, 0) = partial_dabsh_dh*partial_dh_p;
			// std::cout<<J_transform<<std::endl;
		}
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new LidarPlaneNormAnalyticFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};


struct LidarDistanceFactor
{

	LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactor, 3, 4, 3>(
			new LidarDistanceFactor(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};

#endif // !ALOAM_FACTOR_HPP