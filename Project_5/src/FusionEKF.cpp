#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

#ifdef __APPLE__
#   define sincos __sincos
#endif

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.H_ = H_laser_;

  // set the acceleration noise components
  noise_ax = 9;
  noise_ay = 9;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {

    // first measurement
    cout << "EKF: " << endl;

    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates
      //         and initialize state.
      // set the state with the initial location and zero velocity

      double sin_phi, cos_phi;
      sincos(measurement_pack.raw_measurements_[1], &sin_phi, &cos_phi);

      ekf_.x_ << measurement_pack.raw_measurements_[0] * cos_phi,
                 measurement_pack.raw_measurements_[0] * sin_phi,
                 0,
                 0;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state "as-is" dur to LiDAR measurements nature
      ekf_.x_ << measurement_pack.raw_measurements_[0],
                 measurement_pack.raw_measurements_[1],
                 0,
                 0;
    }

    // state covariance matrix P
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1000, 0,
               0, 0, 0, 1000;

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  /**
   * Prediction
   */

  // Update the state transition matrix F according to the new elapsed time.
  // Time is measured in seconds.
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Precalculations to reduce redundant computations
  const float dt2 = dt * dt;
  const float dt3 = dt2 * dt;
  const float dt4 = dt2 * dt2;

  // Update the process noise covariance matrix using noise_ax and noise_ay
  ekf_.Q_ << dt4/4*noise_ax, 0,              dt3/2*noise_ax, 0,
             0,              dt4/4*noise_ay, 0,              dt3/2*noise_ay,
             dt3/2*noise_ax, 0,              dt2*noise_ax,   0,
             0,              dt3/2*noise_ay, 0,              dt2*noise_ay;


  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Each sensor has it's own measurement covariance matrix, so need to update accordingly
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
