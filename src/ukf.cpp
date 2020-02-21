#include "ukf.h"
#include "Eigen/Dense"
#include<iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;
  
  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  // Augmented state dimension
  n_aug_ = n_x_ + 2;
  
  // augmented mean vector
  x_aug = VectorXd(n_aug_);
  
  // augmented state covariance
  P_aug = MatrixXd(n_aug_, n_aug_);
  
  // sigma point matrix
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  // sigma point prediction
   Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
   	  
  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;
  
  // time when the state is true, in us
  time_us_ = 0;
  
  // Intialize the weights
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  double weight = 0.5/(lambda_+n_aug_);
  weights_(0) = weight_0;

  for (int i=1; i<2*n_aug_+1; ++i) {  
    weights_(i) = weight;
  }
  
  // the current NIS for radar
  NIS_radar_ = 0.0;

  // the current NIS for laser
  NIS_laser_ = 0.0;

  // measurement matrix for RADAR Sensor
  H_ = MatrixXd(2, 5);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

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
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
   
   // measurement noise covairance matix
   R_lidar_ = MatrixXd(2,2);
   R_lidar_ << std_laspx_*std_laspx_, 0,
               0, std_laspy_*std_laspy_;
     
   R_radar_ = MatrixXd(3, 3);
   R_radar_ << std_radr_*std_radr_, 0, 0,
               0, std_radphi_*std_radphi_, 0,
               0, 0, std_radrd_*std_radrd_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
   //switching between lidar and radar
   
   if (!is_initialized_) {
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rhodot = meas_package.raw_measurements_(2);
      double x = rho * cos(phi);
      double y = rho * sin(phi);
      double vx = rhodot * cos(phi);
      double vy = rhodot * sin(phi);
      double v = sqrt(vx * vx + vy * vy);
      x_ << x, y, v, rho, rhodot;
      
      //state covariance matrix
      //***** values can be tuned *****
      P_ << std_radr_*std_radr_, 0, 0, 0, 0,
            0, std_radr_*std_radr_, 0, 0, 0,
            0, 0, std_radrd_*std_radrd_, 0, 0,
            0, 0, 0, std_radphi_, 0,
            0, 0, 0, 0, std_radphi_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.
      // ***** Last three values below can be tuned *****
      x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0.0;
      
      //state covariance matrix
      //***** values can be tuned *****
      P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
            0, std_laspy_*std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }
    
    // done initializing, no need to predict or update
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
       
  }
  
  // Calculate delta_t, store current time for future
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;

  time_us_ = meas_package.timestamp_;

  // Predict
  Prediction(delta_t);

  // Measurement updates
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }
      
}

void UKF::Prediction(double delta_t) {
  /**
   * Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
   
  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
   
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  
  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();
  
  // create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; ++i) {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
  
  // Sigma Point prediction
  for(int i=0; i<2*n_aug_+1; i++){
 
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double vk = Xsig_aug(2, i);
    double yawk = Xsig_aug(3, i);
    double yawdotk = Xsig_aug(4, i);
    double miua = Xsig_aug(5, i);
    double miuyaw = Xsig_aug(6, i);

    if(fabs(yawdotk) <0.001){
      Xsig_pred_(0, i) = px + (vk * cos(yawk) * delta_t) + (0.5 * delta_t*delta_t*cos(yawk)*miua);  
      Xsig_pred_(1, i) = py + (vk * sin(yawk) * delta_t) + (0.5 * delta_t*delta_t*sin(yawk)*miua);  
    }else{
      Xsig_pred_(0, i) = px + vk/yawdotk * ( sin (yawk + yawdotk*delta_t) - sin(yawk)) + (0.5 * delta_t*delta_t*cos(yawk)*miua);  
      Xsig_pred_(1, i) = py + vk/yawdotk * ( cos(yawk) - cos(yawk+yawdotk*delta_t) ) + (0.5 * delta_t*delta_t*sin(yawk)*miua);   
    }
    Xsig_pred_(2, i) = vk + delta_t * miua;
    Xsig_pred_(3, i) = yawk + 0.5 * delta_t * delta_t * miuyaw;
    Xsig_pred_(4, i) = yawdotk + delta_t * miuyaw;  
  }

  // Predict Mean state and Covariance Matrix
  x_.fill(0);
  for(int i=0; i<2 * n_aug_ +1; i++){
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  
  // predict state covariance matrix
  P_.fill(0);
  for(int i=0; i<2 * n_aug_ +1; i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
  
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

   /**** Note ****
   * the meaturement function hasn't non-linear effect, so we can use the standard KF
   */
   
  int n_z = 2;
  VectorXd z =  meas_package.raw_measurements_;
  
  // The residual
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  
  // system uncertainity
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_lidar_;
  
  // Kalman gain
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
   
  // state update
  x_ = x_ + (K * y);
  
  // covariance update
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K * H_) * P_;
   
  //calculate NIS
  NIS_laser_ = y.transpose() * S.inverse() * y;
  
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
   
   int n_z = 3;
   VectorXd z = meas_package.raw_measurements_;
   
   // create matrix for sigma points in measurement space
   MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
   
   // transform sigma points into measurement space
  for (int i = 0; i < 2*n_aug_+1; i++)
  {
    double px = Xsig_pred_(0, i); 
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw  = Xsig_pred_(3, i);
    Zsig(0, i) = sqrt(px*px + py*py);
    Zsig(1, i) = atan2(py,px);
    Zsig(2, i) = (px*cos(yaw)*v + py*sin(yaw)*v)/sqrt(px*px + py*py);;
  }
  
  // calculate mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  for(int i=0; i<2 * n_aug_ +1; i++){
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  
  // The residual
  VectorXd y = z - z_pred;
  while (y(1)> M_PI) y(1)-=2.*M_PI;
  while (y(1)<-M_PI) y(1)+=2.*M_PI;
   
  // system uncertainity
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0);
  for(int i=0; i<2 * n_aug_ +1; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose() ;
  }
  S = S + R_radar_;
  
  // create matrix for cross correlation T
  MatrixXd T = MatrixXd(n_x_, n_z);
  // calculate cross correlation matrix
  T.fill(0.0);
  for(int i=0; i<2 * n_aug_ +1; i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    T = T + weights_(i) * x_diff * z_diff.transpose() ;
  }
  
  // Kalman gain
  MatrixXd K = T * S.inverse();
  
  // state update
  x_ = x_ + K * y;
  
  // covariance update
  P_ = P_ - K*S*K.transpose();
    
  //calculate NIS
  NIS_radar_ = y.transpose() * S.inverse() * y;
}