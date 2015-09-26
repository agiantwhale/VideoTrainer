/*
 * =====================================================================================
 *
 *       Filename:  nnet.hpp
 *
 *    Description:  Neural Network
 *
 *        Version:  1.0
 *        Created:  09/26/2015 18:29:26
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#ifndef NNET_HPP
#define NNET_HPP

#include "train.hpp"

#include <fann.h>

namespace maav {
  class NeuralNet : public LearnInterface {
    public:
      NeuralNet(
          unsigned int num_layers,
          unsigned int num_neurons_hidden,
          unsigned int max_epochs
          );
      virtual ~NeuralNet();
      virtual bool load(const std::string & file_path);
      virtual bool save(const std::string & file_path);
      virtual void train(const std::vector< std::vector<Features> > & positives,
                         const std::vector<Features> & negatives);
      virtual bool test(const Features & features);

    private:
      void train_single_iteration(const Features & features,
                                  const Features & output);

      unsigned int num_layers_;
      unsigned int num_neurons_hidden_;
      unsigned int max_epochs_;
      struct fann * ann_;
  };
}

#endif
