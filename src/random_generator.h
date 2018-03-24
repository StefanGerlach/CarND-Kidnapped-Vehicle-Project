/*
 * randomgenerator.h
 *
 *  Created on: Mar 20, 2018
 *      Author: skutch
 */

#ifndef RANDOM_GENERATOR_H_
#define RANDOM_GENERATOR_H_

#include <random>

class RandomGenerator {

 // The default random generator engine
 std::default_random_engine random_generator;

 public:

 /*
  * Random generator from normal distribution, given mean and standard deviation.
  * */
   double gaussian(const double& mean, const double& mu);

   /*
    * discrete distribution for resampling of particles.
    * */
   void resample(const std::vector<double>& weights, int num_ids, std::vector<int> &ids);
};

#endif /* RANDOM_GENERATOR_H_ */
