#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  
  	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

double SNormalizeAngle(double phi)
{
  const double Max = M_PI;
  const double Min = -M_PI;

  return phi < Min
    ? Max + std::fmod(phi - Min, Max - Min)
    : std::fmod(phi - Min, Max - Min) + Min;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

//   MatrixXd h_;
   
//   h_ << 0, 0, 0, 0,
//		 0, 0, 0, 0,
//		 0, 0, 0, 0;
//    MatrixXd h_ = tools::CalculateJacobian(x_);
    Tools tools;
    MatrixXd h_ = tools.CalculateJacobian(x_);

    double ro_pred = pow(x_[0]*x_[0] + x_[1]*x_[1], 0.5);
    double theta_pred = 0.0;
    if (fabs(x_[0]) > 0.0001) {
        theta_pred = atan2(x_[1], x_[0]);
    }

    double rodot_pred = 0.0;
    if (fabs(ro_pred) > 0.0001) {
        rodot_pred = (x_[0] * x_[2] + x_[1] * x_[3]) / ro_pred;
    }

    VectorXd z_pred(3);
    z_pred << ro_pred, theta_pred, rodot_pred;

    VectorXd y = z - z_pred;
	
	y[1] = SNormalizeAngle(y[1]);
	
    MatrixXd ht = h_.transpose();
    MatrixXd S = h_ * P_ * ht + R_radar_;
    MatrixXd Si = S.inverse();
    MatrixXd Pht = P_ * ht;
    MatrixXd K = Pht * Si;

    // new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * h_) * P_;
}
