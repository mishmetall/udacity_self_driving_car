#include "PID.h"

#include <iostream>
#include <chrono>

using std::cout;
using std::endl;

double c_get_realtime_s(void)
{
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return (double) (t.tv_sec + t.tv_nsec / 1e9);
}

inline double abs(double v) {
  if ( v < 0 )
    return -v;
  return v;
}

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  Kp = std::max(Kp_, 0.0);
  Ki = std::max(Ki_, 0.0);
  Kd = std::max(Kd_, 0.0);

  p_error = 0;
  i_error = 0;
  d_error = 0;

  cte_initialized = 0;

  curr_n = 0;
  curr_cte_sq_sum = 0;

  if ( optimize_params ) {
     cout << "Current best: " << p[0] << " " << p[1] << " " << p[2] << endl;
  }

  for ( double & d_val : dp ) {
    d_val = std::max(d_val, 0.001); // minimal step
  }
}

void PID::UpdateError(double cte) {

  double dt = 0;
  if ( !cte_initialized ) {
    d_error = 0;
    cte_initialized = true;
    prev_time = c_get_realtime_s();
  } else {
    double now = c_get_realtime_s();
    dt = now - prev_time;
    d_error = (cte - p_error) / dt;
    i_error += (cte + p_error) / 2.0 * dt;

    prev_time = now;

    curr_cte_sq_sum += (cte + p_error) * (cte + p_error) / 2.0 * dt;
  }
  p_error = cte;

  if ( optimize_params ) {
    curr_n += dt;
    if ( recover_phase ) {
      if ( curr_n > n_sec_recovery ) {
        recover_phase = false;
        Init(p[0], p[1], p[2]);
        cout << "End of recovery phase!" << endl;
      }
      else {
        return;
      }
    }

    cout << "pi: " << pi << " Kp " << Kp <<  " Ki " << Ki << " Kd " << Kd << endl;
    if ( best_cte_sum <= curr_cte_sq_sum || abs(cte) >= max_cte ) {
      // already too bad results, drop attempt
      cout << "Already too bad. Step " << curr_n << ", error: " << curr_cte_sq_sum << ". Trying another params" << endl;
      if ( abs(cte) >= max_cte ) {
        cout << "Too high cte " << cte << endl;
      }

      if ( positive_phase ) {
        // Try to subtract
        p[pi] -= dp[pi];
        Init(p[0], p[1], p[2]); // use prev parameters to recover
        p[pi] -= dp[pi]; // negative phase after recovery
        positive_phase = false;
        cout << "negative phase!" << endl;
      }
      else {
        p[pi] += dp[pi];
        Init(p[0], p[1], p[2]); // use prev parameters to recover
        dp[pi] *= 0.9;
        positive_phase = true;

        pi = ( pi + 1 ) % 3;
        p[pi] += dp[pi];
        cout << "positive phase!" << endl;
      }

      recover_phase = true;
      cout << "Recovery phase!" << endl;
      return;
    }
    else if ( curr_n > n_sec ) {
      // great! have new best
      best_cte_sum = curr_cte_sq_sum;
      cout << "New best! " << best_cte_sum << endl;

      positive_phase = true;
      dp[pi] *= 1.1;

      pi = ( pi + 1 ) % 3;
      p[pi] += dp[pi];

      Init(p[0], p[1], p[2]);
    }
  }
}

double PID::TotalError() {
  return i_error;
}

double PID::FindNewTarget() {
  return -( Kp * p_error + Kd * d_error + Ki * i_error );
}

bool PID::getOptimize_params() const
{
  return optimize_params;
}

void PID::setOptimize_params(bool value)
{
  optimize_params = value;
}

int PID::getN_sec() const
{
  return n_sec;
}

void PID::setN_sec(int value)
{
  n_sec = value;
}

std::vector<double> PID::getP() const
{
  return p;
}

void PID::setP(const std::vector<double> &value)
{
  p = value;
}

std::vector<double> PID::getDp() const
{
  return dp;
}

void PID::setDp(const std::vector<double> &value)
{
  dp = value;
}

double PID::getP_error() const
{
  return p_error;
}

double PID::getI_error() const
{
  return i_error;
}

double PID::getD_error() const
{
  return d_error;
}
