#include "kalman_filter.h"

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
  predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  // variance update
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
      update the state by using Kalman Filter equations
  */
  VectorXd zp = H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  // Kalman gain
  MatrixXd K = P_ * Ht * S.inverse();
  // update
  x_ = x_ + K * (z - zp);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // transform state in  cardesian to polar coordinate system
  float ro = pow(pow(x_[0],2)+pow(x_[1],2),0.5);
  float phi = 0.0;
  // two threshold can be tuned
  if (fabs(x_[0]) > 0.05) {
    phi  = atan2(x_[1],x_[0]);
  }
  float rodot = 0.0;
  if (fabs(ro) > 0.1) {
     rodot = (x_[0]*x_[2] + x_[1]*x_[3]) / ro;
  }

  VectorXd zp(3);
  zp << ro, phi, rodot;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  // using error in polar system

  x_ = x_ + K * (z - zp);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
