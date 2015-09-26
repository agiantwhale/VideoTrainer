/*
 * =====================================================================================
 *
 *       Filename:  nnet.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  09/26/2015 18:34:15
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#include "nnet.hpp"

maav::NeuralNet::NeuralNet(
  unsigned int num_layers,
  unsigned int num_neurons_hidden
) : num_layers_(num_layers),
    num_neurons_hidden_(num_neurons_hidden),
    ann_(nullptr) {
};

maav::NeuralNet::~NeuralNet() {
  if(ann_) fann_destroy(ann_);
}

bool maav::NeuralNet::load(const std::string & file_path) {
  if(ann_) fann_destroy(ann_);
  ann_=fann_create_from_file(file_path.c_str());
  return true;
}

bool maav::NeuralNet::save(const std::string & file_path) {
  if(ann_) fann_save(ann_, file_path.c_str());
  return true;
}

void maav::NeuralNet::train(const std::vector< std::vector<maav::Features> > & positives,
                            const std::vector< maav::Features > & negatives) {
  const unsigned int num_input=positives.front().size();
  const unsigned int num_output=positives.size()+1;

  if(ann_) fann_destroy(ann_);
  ann_=fann_create_standard(num_layers_, num_input,
                            num_neurons_hidden_, num_output);

  fann_set_activation_function_hidden(ann_, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann_, FANN_SIGMOID_SYMMETRIC);

  // Train the positives first.
  for(unsigned int positives_num=0;
      positives_num<positives.size();
      positives_num++) {
    std::vector<float> output(num_output, 0);
    output[positives_num]=1;

    const std::vector<maav::Features> & positive_features=positives[positives_num];
    for(auto iter=positive_features.begin();
        iter!=positive_features.end();
        iter++) {
      train_single_iteration((*iter), output);
    }
  }

  // Train negatives.
  std::vector<float> output(num_output, 0);
  for(auto iter=negatives.begin();
      iter!=negatives.end();
      iter++) {
    train_single_iteration((*iter), output);
  }
}

bool maav::NeuralNet::test(const maav::Features & features) {
  if(!ann_) return false;
  float * result=fann_run(ann_, (float*)&(features[0]));
  for(unsigned int i=0;i<fann_get_num_output(ann_);i++) {
    if(result[i]>0) return true;
  }
  return false;
}

void maav::NeuralNet::train_single_iteration(const Features & features,
                                             const Features & output) {
  fann_train(ann_, (float*)&(features[0]), (float*)&(output[0]));
}
