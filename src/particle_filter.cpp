/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // Test code
  /*double sig_x = 0.3;
  double sig_y = 0.3;
  double x_obs = 6;
  double y_obs = 3;
  double mu_x = 5;
  double mu_y = 3;
  double d = multi_variant_gaussian_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);

  std::vector<LandmarkObs> observ(3);
  Particle p;

  observ[0].x=2; observ[0].y=2;
  observ[1].x=3; observ[1].y=-2;
  observ[2].x=0; observ[2].y=-4;

  p.x=4;
  p.y=5;
  p.theta = -1.5708;

  transformCoordinateSystem(p, observ);
*/

	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 500;

  for(int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.weight = 1.0;

    // Setting the position and yaw of the particle with uncertainty of GPU measurement.
    p.x = random_generator.gaussian(x, std[0]);
    p.y = random_generator.gaussian(y, std[1]);
    p.theta = random_generator.gaussian(theta, std[2]);

    // Adding this particle to the vector of internal particles.
    particles.push_back(p);

    // Adding the weight to the vector of weights
    weights.push_back(1.0);
  }

  std::cout << "Initialized particle filter with " << particles.size() << " particles." << std::endl;
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  double noisy_velocity = random_generator.gaussian(velocity, std_pos[0]);
  double noisy_yaw_rate = random_generator.gaussian(yaw_rate, std_pos[1]);

  // PARALLELIZE !
  for(auto & particle: particles) {
    apply_motion_model(particle, delta_t, noisy_velocity, noisy_yaw_rate);
  }
}

void ParticleFilter::apply_motion_model(Particle &particle, const double& delta_t, const double& velocity, const double &yaw_rate) {
  // This is the implementation of the simple bicycle motion model, applied to a particle
  particle.x = particle.x + (velocity / yaw_rate) * (std::sin(particle.theta + (delta_t * yaw_rate)) - std::sin(particle.theta));
  particle.y = particle.y + (velocity / yaw_rate) * (std::cos(particle.theta) - std::cos(particle.theta + (delta_t * yaw_rate)));

  particle.theta = particle.theta + yaw_rate;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}

bool ParticleFilter::dataAssociation(Particle& particle, const std::vector<LandmarkObs> &transformed, const Map &map) {
  // Clear destination
  particle.sense_x.clear();
  particle.sense_y.clear();

  //std::vector<double> dists;

  // For every observed landmark, we look for the nearest map landmark
  double cur_dist, min_dist; // by intention stay uninitialized until first iteration
  double min_dist_x, min_dist_y;

  for(const auto& landmark : transformed) {
    cur_dist = -1.0;
    min_dist = -1.0;
    min_dist_x = -1;
    min_dist_y = -1;

    for(const auto& map_lm : map.landmark_list) {
      // Compute euclidean distance
      cur_dist = dist(landmark.x,landmark.y, map_lm.x_f, map_lm.y_f);

      // Check if this is the nearest landmark
      if(min_dist < 0.0 || cur_dist < min_dist)  {
        min_dist = cur_dist;
        min_dist_x = map_lm.x_f;
        min_dist_y = map_lm.x_f;
      }
    }
    //dists.push_back(cur_dist);

    particle.sense_x.push_back(min_dist_x);
    particle.sense_y.push_back(min_dist_y);
  }
  //std::cout << "particle x: " << particle.x << " y: " << particle.y << std::endl;
  //std::cout << "min_dist : " << *std::min_element(dists.begin(), dists.end()) << " max: " << *std::max_element(dists.begin(), dists.end())  << std::endl;

  return true;
}

void ParticleFilter::setWeight(Particle &particle, const double& std_x, const double& std_y, const std::vector<LandmarkObs> &transformed) {
  std::vector<double> multi_variant_probs;

  particle.weight = 1.0;

  int id=0;
  for(const auto& landmark : transformed) {
    particle.weight *= multi_variant_gaussian_prob(std_x,
                                                   std_y,
                                                   landmark.x,
                                                   landmark.y,
                                                   particle.sense_x[id],
                                                   particle.sense_y[id]);

    id++;
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // For every particle apply the following pipeline:
  for(auto& particle : particles) {
    // copy the original observations from the sensors
    std::vector<LandmarkObs> transformedObservations(observations);

    // Transform to the map coordinate system
    transformCoordinateSystem(particle, transformedObservations);

    if(!dataAssociation(particle, transformedObservations, map_landmarks))
      std::cout << "There was an error computing the associations!" << std::endl;

    setWeight(particle, std_landmark[0], std_landmark[1], transformedObservations);
  }
}

void ParticleFilter::transformCoordinateSystem(const Particle &particle, std::vector<LandmarkObs> &observations) {

  // This will implement the homogenous transformation in 2d
  double new_x, new_y;
  for(auto& obs : observations)
  {
     new_x = particle.x + (std::cos(particle.theta) * obs.x) - (std::sin(particle.theta) * obs.y);
     new_y = particle.y + (std::sin(particle.theta) * obs.x) + (std::cos(particle.theta) * obs.y);

     obs.x = new_x;
     obs.y = new_y;
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<double> weights;
  std::vector<int> resampled_ids;
  std::vector<Particle> new_particles;

  for(const auto& p : particles)  {
    weights.push_back(p.weight);
  }
  std::cout << "min_weight : " << *std::min_element(weights.begin(), weights.end()) << " max: " << *std::max_element(weights.begin(), weights.end())  << std::endl;

  random_generator.resample(weights, particles.size(), resampled_ids);

  for(const auto& id:resampled_ids)
    new_particles.push_back(particles[id]);

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
