#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	// Initially the UKF is not initialized
	is_initialized_ = false;

	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);
	x_.fill(0.0);

	// initial covariance matrix
	P_ = MatrixXd(5, 5);
	P_ << 1, 0, 0, 0,0,
	0, 1, 0, 0,0,
	0, 0, 1, 0,0,
	0, 0, 0, 1,0,
	0, 0, 0, 0,1;

	// Number of rows in our state vector
	n_x_ = x_.rows();

	// Number of rows in our state vector + 2 rows for the noise processes
	n_aug_ = n_x_ + 2;

	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
	Xsig_pred_.fill(0.0);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 2.0; //0.5

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.5; //0.55

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	lambda_ = 3 - n_aug_;

	weights_ = VectorXd(2*n_aug_ + 1);

	weights_(0) = lambda_ / float(lambda_ + n_aug_);
	double common_weight = 1 / float(2 *(lambda_ + n_aug_));

	for(int i=1; i<weights_.size(); ++i) {
	  weights_(i) = common_weight;
	}

	NIS_radar_ = 0;
	NIS_laser_ = 0;

	R_laser_ = MatrixXd(2, 2);
	R_laser_ << pow(std_laspx_, 2), 0,
				0, pow(std_laspy_, 2);

	R_radar_ = MatrixXd(3, 3);
	R_radar_ << pow(std_radr_, 2), 0, 0,
				0, pow(std_radphi_, 2), 0,
				0, 0, pow(std_radrd_, 2);

	previous_timestamp_ = 0;

	H_laser_ = MatrixXd(2, n_x_);
	H_laser_ << 1, 0, 0, 0, 0,
				0, 1, 0, 0, 0;
}

UKF::~UKF() {
}

void UKF::init(MeasurementPackage meas_package) {
	// Initialize state x based on whether RADAR or LIDAR
	if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		double rho = meas_package.raw_measurements_[0];
		double phi = meas_package.raw_measurements_[1];
		double rho_dot = meas_package.raw_measurements_[2];
		x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
	} else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
		x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
	}
	if(x_(0) == 0 && x_(1) == 0) {
		x_(0) = 0.01;
		x_(1) = 0.01;
	}
	previous_measurement_ = meas_package;
	previous_timestamp_ = meas_package.timestamp_;
	is_initialized_ = true;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	// If not initialized, initialize
	if(!is_initialized_) {
		init(meas_package);
		return;
	}

	double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = meas_package.timestamp_;
	try{
		Prediction(dt);
	} catch(std::range_error e) {
		init(previous_measurement_);
		P_ << 1, 0, 0, 0,0,
						0, 1, 0, 0,0,
						0, 0, 1, 0,0,
						0, 0, 0, 1,0,
						0, 0, 0, 0,1;
	}

	if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		UpdateRadar(meas_package);
	} else {
		UpdateLidar(meas_package);
	}
	previous_measurement_ = meas_package;
}



/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	// Create Augmented state mean vector x_aug and augmented state covariance matrix P_aug
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.fill(0.0);
	x_aug.segment(0, n_x_) = x_;

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(n_x_, n_x_) = pow(std_a_, 2);
	P_aug(n_x_ + 1, n_x_ + 1) = pow(std_yawdd_, 2);

	//create square root matrix
	MatrixXd A = P_aug.llt().matrixL();
	if (P_aug.llt().info() == Eigen::NumericalIssue) {
	    // if decomposition fails, we have numerical issues
	    std::cout << "LLT failed!" << std::endl;
	    throw std::range_error("LLT failed");
	}
	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	MatrixXd term = sqrt(lambda_ + n_aug_) * A;
	for (int i = 0; i < n_aug_; ++i) {
		Xsig_aug.col(i + 1) = x_aug + term.col(i);
		Xsig_aug.col(i + n_aug_ + 1) = x_aug - term.col(i);
	}

	
	 //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
	}
	
	// Use these values to compute the mean and co-variance for the state predicted at time k+1
	
	VectorXd x(n_x_);
	x.fill(0.0);
	//predict state mean
	for(int i=0;i < Xsig_pred_.cols(); ++i) {
	  x += (weights_(i) * Xsig_pred_.col(i));
	}
	x_<<x;

	//P_ << compute_covariance();
	
	  //predicted state covariance matrix
	  MatrixXd P(n_x_, n_x_);
	  P.fill(0.0);
	  for (int i = 0; i < 2 * n_aug_+ 1; i++) {  //iterate over sigma points

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

		P = P + weights_(i) * x_diff * x_diff.transpose() ;
	  }
	  
	  	P_ << P;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	VectorXd z_pred = H_laser_ * x_;

	VectorXd z = meas_package.raw_measurements_;

	VectorXd y = z - z_pred;

	MatrixXd Ht = H_laser_.transpose();
	MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.rows();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_laser_) * P_;

	NIS_laser_ = y.transpose() * Si * y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	// Map the predicted state x_ to the the measurement space

	int n_z_ = 3; // Number of dimensions in the measurement space for RADAR
	MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
	//transform sigma points into measurement space
	for(int i=0; i<Xsig_pred_.cols(); ++i) {
    // extract values for better readibility
		double p_x = Xsig_pred_(0,i);
		double p_y = Xsig_pred_(1,i);
		double v  = Xsig_pred_(2,i);
		double yaw = Xsig_pred_(3,i);

		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;

		// measurement model
		Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
		Zsig(1,i) = atan2(p_y,p_x);                                 //phi
		Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
	}

	// Calculate the mean z_pred and the covariance S of the predicted points
	VectorXd z_pred = VectorXd(n_z_);
	z_pred.fill(0.0);

	for(int i=0; i<Zsig.cols(); ++i) {
		z_pred += (weights_(i) * Zsig.col(i));
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z_, n_z_);
	S = R_radar_;
	for(int i=0; i<Zsig.cols(); ++i) {
		VectorXd diff = Zsig.col(i) - z_pred;
		while (diff(1)> M_PI) diff(1)-=2.*M_PI;
		while (diff(1)<-M_PI) diff(1)+=2.*M_PI;

		S += (weights_(i) * (diff * diff.transpose()));
	}

	// Calculate the cross-correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, n_z_);
	Tc.fill(0.0);
	for(int i=0;i<2 * n_aug_ + 1; ++i) {
	VectorXd diff_x = (Xsig_pred_.col(i) - x_);
		while (diff_x(3)> M_PI) diff_x(3)-=2.*M_PI;
		while (diff_x(3)<-M_PI) diff_x(3)+=2.*M_PI;

		VectorXd diff_z = Zsig.col(i) - z_pred;

		while (diff_z(1)> M_PI) diff_z(1)-=2.*M_PI;
		while (diff_z(1)<-M_PI) diff_z(1)+=2.*M_PI;

		Tc += (weights_(i) * (diff_x * diff_z.transpose()));
	}
	// Use these to update the state x_ and P_ using the Kalman Gain
	MatrixXd K(n_x_, n_z_);
	K = Tc * S.inverse();

	//residual
	VectorXd z = meas_package.raw_measurements_;
	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
	//update state mean and covariance matrix
	x_ = x_ + K* (z_diff);
	P_ = P_ - K * S * K.transpose();

	NIS_radar_ = z_diff.transpose() * S.transpose() * z_diff;
}
