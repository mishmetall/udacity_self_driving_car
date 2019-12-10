/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

using std::normal_distribution;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;
  particles.reserve(num_particles);
  std::default_random_engine gen;

  // This lines creates a normal (Gaussian) distribution for x, y, theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for ( int i=0; i<num_particles; ++i ) {
    // Sample x, y, theta nearby original x, y, theta
    double sample_x = dist_x(gen);
    double sample_y = dist_y(gen);
    double sample_theta = dist_theta(gen);

    particles.push_back(Particle{i, sample_x, sample_y, sample_theta, 1});
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  // Loop over each particle and apply measurements
  for ( Particle & p : particles ) {

    // Move point using formulas:
    // x = x + v/yaw_rate*(sin(theta + yaw_rate*dt) - sin(theta))
    // y = x + v/yaw_rate*(cos(theta) -cos(theta + yaw_rate*dt))
    // theta = theta + yaw_rate * dt

    if ( yaw_rate ) {
      double yaw_rate_dt = yaw_rate * delta_t; // yaw rate for given delta time
      double velocity_to_rot = velocity / yaw_rate;

      p.x += velocity_to_rot * (sin(p.theta + yaw_rate_dt) - sin(p.theta));
      p.y += velocity_to_rot * (cos(p.theta) - cos(p.theta + yaw_rate_dt));
      p.theta += yaw_rate_dt;
    }
    else {
      double velocity_dt = velocity * delta_t;
      p.x += velocity_dt * cos(p.theta);
      p.y += velocity_dt * sin(p.theta);
    }

    // This lines creates a normal (Gaussian) distribution for x, y, theta
    // given position standard deviation
    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);

    // Sample x, y, theta nearby new x, y, theta
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(const Map & map, Particle & p) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  // Naive nearest neighbour, better to use kd-tree, but let be O(N^2)
  for ( size_t i=0; i<p.associations.size(); ++i ) {
    double nearest_dist = std::numeric_limits<double>::max();
    int nearest_id = -1;
    for ( const auto & ml : map.landmark_list ) {
      double distance = dist(ml.x_f, ml.y_f, p.sense_x[i], p.sense_y[i]);
      if ( distance < nearest_dist ) {
        nearest_dist = distance;
        nearest_id = ml.id_i;
      }
    }

    p.associations[i] = nearest_id;
  }
}

static double gaussian_2d(double mu[], double sigma[], double x, double y) {
    // calculates the probability of x, y for 2-dim Gaussian with mean mu and var. sigmas
    return exp(- (pow(mu[0] - x, 2) / (2 * sigma[0] * sigma[0]) + pow(mu[1] - y, 2) / (2 * sigma[1] * sigma[1])) ) / (2.0 * M_PI * sigma[0] * sigma[1]);
}

double ParticleFilter::measurementProbability(const Particle & p,
                                              const Map &map_landmarks,
                                              double sigma_landmark []) {

    /**
     * calculates how likely a measurement should be
     */
    double prob = 1.0;
    const auto & landmarks = map_landmarks.landmark_list;
    for ( size_t i=0; i<p.associations.size(); ++i ) {
      int li = p.associations[i] - 1;
      double mu[2] = {landmarks[li].x_f, landmarks[li].y_f};
      prob *= gaussian_2d(mu, sigma_landmark, p.sense_x[i], p.sense_y[i]);
    }
    return prob;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // For each particle
  weights.clear();
  weights.reserve(particles.size());
  for ( Particle & p : particles ) {

    // clear associations
    SetAssociations(p, {}, {}, {});

    // 1. Convert to map coordinate system all observations
    p.associations.clear();
    p.associations.resize(observations.size());

    p.sense_x.clear();
    p.sense_y.clear();

    for ( const LandmarkObs & obs : observations ) {
      double x_map = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
      double y_map = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);

      p.sense_x.emplace_back(x_map);
      p.sense_y.emplace_back(y_map);
    }

    // 2. Find nearest neighbours on map
    dataAssociation(map_landmarks, p);

    p.weight = measurementProbability(p, map_landmarks, std_landmark);

    // 3. Find probability for each measurement and use it as a weight for particle
    weights.emplace_back(p.weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<Particle> new_particles(particles.size());
  std::default_random_engine gen;

  discrete_distribution<int> dgen(weights.begin(), weights.end());

  for ( size_t i=0; i<particles.size(); ++i ) {
    int pi = dgen(gen);
    new_particles[i] = particles[pi];
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
