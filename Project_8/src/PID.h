#ifndef PID_H
#define PID_H

#include <vector>

class PID {
 public:
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();

  double FindNewTarget();

  bool getOptimize_params() const;
  void setOptimize_params(bool value);

  int getN_sec() const;
  void setN_sec(int value);

  std::vector<double> getP() const;
  void setP(const std::vector<double> &value);

  std::vector<double> getDp() const;
  void setDp(const std::vector<double> &value);

  double getP_error() const;

  double getI_error() const;

  double getD_error() const;

private:
  /**
   * PID Errors
   */
  double p_error;
  double i_error;
  double d_error;

  /**
   * PID Coefficients
   */ 
  double Kp;
  double Ki;
  double Kd;

  /**
   * Additional variables
   */
  bool cte_initialized = 0;
  double prev_time;

  /**
   * Optimization.
   *
   * Algorithm is following: each n_sec (e.g. 60) seconds try to vary parameter and calculate total error.
   *  If it was better than n_sec before - leave this parameter and go to the next one. We'll use
   *  "online" twiddle tuning parameter, as it will take huge amount of time to optimize by restarting simulator
   *  manually hundrets of times.
   */
  bool optimize_params = false;
  double n_sec = 60;
  double n_sec_recovery = 5;
  double curr_n;
  bool recover_phase = false; // after bad params it's too early to immediately change params, let use previous params for 1 cycle
  double curr_cte_sq_sum; // current quadratic error
  double max_cte = 1.9;
  double best_cte_sum = 10000000; // drop optimization if results are too bad already. Last best attempt value.
  size_t pi = 0; // parameter id for optimization
  bool positive_phase = true; // switch for twiddle
  std::vector<double> p  = {0.5, 0, 0.21}; // best parameters so far
  std::vector<double> dp = {0.1, 0.1, 0.1}; // changes for twiddle
};

#endif  // PID_H
