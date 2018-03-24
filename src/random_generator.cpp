/*
 * randomgenerator.cpp
 *
 *  Created on: Mar 20, 2018
 *      Author: skutch
 */

#include "random_generator.h"

/*
 * Random generator from normal distribution, given mean and standard deviation.
 * */
double RandomGenerator::gaussian(const double& mean, const double& mu) {
  // Create a normal distribution, aka gaussian
  std::normal_distribution<double> dist(mean, mu);

  // Return a sample from it
  return dist(random_generator);
}

void RandomGenerator::resample(const std::vector<double>& weights, int num_ids, std::vector<int> &ids) {

  ids.clear();
  std::discrete_distribution<int> dist(weights.begin(), weights.end());

  for(int i = 0; i < weights.size(); i++) {
    ids.push_back(static_cast<int>(dist(random_generator)));
  }
}
