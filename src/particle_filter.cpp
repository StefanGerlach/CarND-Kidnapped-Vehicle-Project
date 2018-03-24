/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#define MULTI_THREADING

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <thread>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {

#if 0
  // Test code for verification of implementation.
  // This code will reconstruct the quizzes in Lession 15.

  // Prediction Step
  Particle p_pred;
  p_pred.x = 102.0;
  p_pred.y = 65.0;
  p_pred.theta = 5.0 * M_PI / 8.0;

  double velocity = 110.0;
  double yaw_rate = M_PI / 8.0;
  double delta_t = 0.1; // sec
  apply_motion_model(p_pred, delta_t, velocity, yaw_rate);

  double solution_theta = 51.0*M_PI / 80.0;
  double solution_x = 97.59;
  double solution_y = 75.08;

  // Update step

  double sig_x = 0.3;
  double sig_y = 0.3;
  double x_obs = 6;
  double y_obs = 3;
  double mu_x = 5;
  double mu_y = 3;
  double d = multi_variant_gaussian_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);

  Map map;
  std::vector<LandmarkObs> observ(3);
  Particle p;

  Map::single_landmark_s l1, l2, l3, l4, l5;
  l1.x_f=5.0;  l1.y_f=3.0;
  l2.x_f=2.0;  l2.y_f=1.0;
  l3.x_f=6.0;  l3.y_f=1.0;
  l4.x_f=7.0;  l4.y_f=4.0;
  l5.x_f=4.0;  l5.y_f=7.0;

  map.landmark_list = {l1, l2, l3, l4, l5};

  observ[0].x=2; observ[0].y=2;
  observ[1].x=3; observ[1].y=-2;
  observ[2].x=0; observ[2].y=-4;

  p.x=4;
  p.y=5;
  p.theta = -1.5708;

  transformCoordinateSystem(p, observ);
  dataAssociation(p, observ, map);
  setWeight(p, 0.3, 0.3, observ);
#endif

	// Set the number of particles. Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 500;

  // Set the number of threads
  num_threads = 8;

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

#ifdef MULTI_THREADING
  // This loop is made to be parallelized
  // Create lists of particle references for the n threads
  std::vector<std::vector<Particle*> > particle_vectors(num_threads, std::vector<Particle*>());
  for(int i = 0; i < particles.size(); i++)
    particle_vectors[i % num_threads].push_back(&particles[i]);

  // Create a vector of all threads
  std::vector<std::thread> threads;

  for(auto& particle_vec : particle_vectors)
    // Add a new thread to the list of threads
    threads.push_back(std::thread([&](){
    // This thread will process this lambda function
    // that applies motion model and position noise to the particle
      for(auto& particle : particle_vec) {
        apply_motion_model(*particle, delta_t, velocity, yaw_rate);
        apply_position_noise(*particle, std_pos[0], std_pos[1], std_pos[2]);
      }
    }));

  // Wait for the threads to finish computation
  for(auto& t : threads)
    t.join();

#else

  // Single threading variant
  for(auto & particle: particles) {
      apply_motion_model(particle, delta_t, velocity, yaw_rate);
      apply_position_noise(particle, std_pos[0], std_pos[1], std_pos[2]);
    }

#endif
}

void ParticleFilter::apply_position_noise(Particle &particle, const double& std_x, const double& std_y, const double& std_theta) {

  particle.x = random_generator.gaussian(particle.x, std_x);
  particle.y = random_generator.gaussian(particle.y, std_y);
  particle.theta = random_generator.gaussian(particle.theta, std_theta);
}

void ParticleFilter::apply_motion_model(Particle &particle, const double& delta_t, const double& velocity, const double &yaw_rate) {
  // This is the implementation of the simple bicycle motion model, applied to a particle
  double yaw_rate_dt = yaw_rate * delta_t;
  double theta_yaw_rate_dt = particle.theta + yaw_rate_dt;

  if(abs(yaw_rate) < 1e-8) {
    particle.x += velocity * delta_t * cos(particle.theta); // see Lession 14 - Robot class movement
    particle.y += velocity * delta_t * sin(particle.theta);
  }
  else
  {
    double velo_over_yaw_rate = velocity / yaw_rate;

    particle.x += velo_over_yaw_rate * (sin(theta_yaw_rate_dt) - sin(particle.theta));
    particle.y += velo_over_yaw_rate * (cos(particle.theta) - cos(theta_yaw_rate_dt));
  }

  particle.theta += yaw_rate_dt;
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
  particle.associations.clear();

  //std::vector<double> dists;

  // For every observed landmark, we look for the nearest map landmark
  double cur_dist, min_dist; // by intention stay uninitialized until first iteration
  double min_dist_x, min_dist_y, min_dist_id;

  for(const auto& landmark : transformed) {
    cur_dist = -1.0;
    min_dist = -1.0;
    min_dist_x = -1;
    min_dist_y = -1;
    min_dist_id= -1;
    for(const auto& map_lm : map.landmark_list) {
      // Compute euclidean distance
      cur_dist = dist(landmark.x,landmark.y, map_lm.x_f, map_lm.y_f);

      // Check if this is the nearest landmark
      if(min_dist < 0.0 || cur_dist < min_dist)  {
        min_dist = cur_dist;
        min_dist_x = map_lm.x_f;
        min_dist_y = map_lm.y_f;
        min_dist_id = map_lm.id_i;
      }
    }
    //dists.push_back(cur_dist);

    particle.sense_x.push_back(min_dist_x);
    particle.sense_y.push_back(min_dist_y);
    particle.associations.push_back(min_dist_id);
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
    multi_variant_probs.push_back(multi_variant_gaussian_prob(std_x,
                                                   std_y,
                                                   landmark.x,
                                                   landmark.y,
                                                   particle.sense_x[id],
                                                   particle.sense_y[id]));

    id++;
  }
  for(const auto& cweight:multi_variant_probs) {
    particle.weight *= cweight;
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
  weights.clear();

#ifdef MULTI_THREADING
  // This loop is made to be parallelized
  // Create lists of particle references for the n threads
  std::vector<std::vector<Particle*> > particle_vectors(num_threads, std::vector<Particle*>());
  for(int i = 0; i < particles.size(); i++)
    particle_vectors[i % num_threads].push_back(&particles[i]);

  // Create a vector of all threads
  std::vector<std::thread> threads;

  for(auto& particle_vec : particle_vectors)
    // Add a new thread to the list of threads
    threads.push_back(std::thread([&](){
      // This thread will process this lambda function
      // that applies the processing pipeline to a subset of all particles
      for(auto& particle : particle_vec) {

        // copy the original observations from the sensors
        std::vector<LandmarkObs> transformedObservations(observations);

        // Transform to the map coordinate system
        transformCoordinateSystem(*particle, transformedObservations);

        if(!dataAssociation(*particle, transformedObservations, map_landmarks))
          std::cout << "There was an error computing the associations!" << std::endl;

        setWeight(*particle, std_landmark[0], std_landmark[1], transformedObservations);
      }
    }));

  // Wait for the threads to finish computation
  for(auto& t : threads)
    t.join();

  // Collect weights
  for(const auto& particle: particles)
    weights.push_back(particle.weight);

#else

  // The single threading variant
  for(auto& particle : particles) {
    // copy the original observations from the sensors
    std::vector<LandmarkObs> transformedObservations(observations);

    // Transform to the map coordinate system
    transformCoordinateSystem(particle, transformedObservations);

    if(!dataAssociation(particle, transformedObservations, map_landmarks))
      std::cout << "There was an error computing the associations!" << std::endl;

    setWeight(particle, std_landmark[0], std_landmark[1], transformedObservations);

    weights.push_back(particle.weight);
  }
#endif
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

  std::vector<double> selected_weights;

  for(const auto& id:resampled_ids) {
    selected_weights.push_back(particles[id].weight);
    new_particles.push_back(particles[id]);
  }

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
