/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  09/26/2015 18:25:55
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#include "train.hpp"
#include "video.hpp"
#include "nnet.hpp"
#include "hog.hpp"

#include <iostream>

namespace maav {
  class FeatureExtractMethod {
    public:
      FeatureExtractMethod(
          ExtractInterface & extract_interface,
          std::vector<Features> & features_collection
          ) :
        extract_interface_(extract_interface),
        features_collection_(features_collection) {}

      void operator() (const cv::Mat & image) const {
        Features features((features_collection_.size()==0?
              0:features_collection_.front().size()));
        extract_interface_.compute(image, features);
        features_collection_.push_back(features);
      }

    private:
      ExtractInterface & extract_interface_;
      std::vector<Features> & features_collection_;
  };

  class NegativeMiningMethod {
    public:
      NegativeMiningMethod(
          ExtractInterface & extract_interface,
          LearnInterface & learn_interface,
          std::vector<Features> & negative_features_collection
          ) :
        extract_interface_(extract_interface),
        learn_interface_(learn_interface),
        negative_features_collection_(negative_features_collection) {}

      void operator() (const cv::Mat & image) {
        Features features((negative_features_collection_.size()==0?
              0:negative_features_collection_.front().size()));
        extract_interface_.compute(image, features);
        if(learn_interface_.test(features)) negative_features_collection_.push_back(features);
      }

    private:
      ExtractInterface & extract_interface_;
      LearnInterface & learn_interface_;
      std::vector<Features> & negative_features_collection_;
  };
}

int main() {
  const cv::Size window_size=cv::Size(128,72);

  maav::HOGExtractor extractor(window_size);
  maav::NeuralNet learner(3, 3, 5000000);

  std::vector<maav::Features> features_collection;
  std::vector<unsigned int> divider;

  maav::FeatureExtractMethod extract_method(extractor, features_collection);
  boost::function<void (const cv::Mat &)> f=boost::ref(extract_method);

  maav::LoadEachFrameFromFile("pos1.mov", f);
  for(unsigned int i=0;i<(features_collection.size()-divider.size());i++) {
    divider.push_back(0);
  }

  extractor.scale_=false;
  maav::LoadEachFrameFromFile("neg.mov", f);

  learner.train(features_collection, divider);
  learner.save("trained.dat");
}
