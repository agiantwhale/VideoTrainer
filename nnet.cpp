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
  unsigned int num_neurons_hidden,
  unsigned int max_epoch
) : num_layers_(num_layers),
    num_neurons_hidden_(num_neurons_hidden),
    max_epoch_(max_epoch),
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
  unsigned int num_data=negatives.size();
  for(unsigned int i=0;i<positives.size();i++) num_data+=positives[i].size();
  const unsigned int num_input=positives.front().size();
  const unsigned int num_output=positives.size()+1;

  if(ann_) fann_destroy(ann_);
  ann_=fann_create_standard(num_layers_, num_input,
                            num_neurons_hidden_, num_output);

  fann_set_activation_function_hidden(ann_, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann_, FANN_SIGMOID_SYMMETRIC);

  struct fann_train_data * data=
    fann_create_train(num_data, num_input, num_output);

  unsigned int input_count=0;
  for(unsigned int i=0;i<positives.size();i++) {
    for(unsigned int f=0;f<positives[i].size();f++) {
      data->input[input_count]=(float*)positives[i][f].data();
      input_count++;
    }
  }
  for(unsigned int f=0;f<negatives.size();f++) {
    data->input[input_count]=(float*)negatives[f].data();
    input_count++;
  }

  std::vector<maav::Features> positive_outputs(positives.size(),
                                               maav::Features(num_output, 0));
  maav::Features negative_output(num_output, 0);

  unsigned int output_count=0;
  for(unsigned int i=0;i<positives.size();i++) {
    positive_outputs[i][i]=1;
    for(unsigned int f=0;f<positives[i].size();f++) {
      data->output[output_count]=(float*)positives[i].data();
      output_count++;
    }
  }
  for(unsigned int f=0;f<negatives.size();f++) {
    data->output[output_count]=(float*)negative_output.data();
    output_count++;
  }

  fann_train_on_data(ann_, data, max_epoch_, 1000, 0.f);

  for(unsigned int i=0;i<data->num_data;i++) {
    data->input[i]=nullptr;
  }
  for(unsigned int i=0;i<data->num_data;i++) {
    data->output[i]=nullptr;
  }
  fann_destroy_train(data);
}

bool maav::NeuralNet::test(const maav::Features & features) {
  if(!ann_) return false;
  float * result=fann_run(ann_, (float*)&(features[0]));
  for(unsigned int i=0;i<fann_get_num_output(ann_);i++) {
    if(result[i]>0) return true;
  }
  return false;
}
