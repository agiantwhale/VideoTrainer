/*
 * =====================================================================================
 *
 *       Filename:  train.hpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  09/26/2015 15:27:32
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#ifndef TRAIN_HPP
#define TRAIN_HPP

#include <vector>
#include <string>
#include <boost/function.hpp>
#include <opencv/cv.hpp>

namespace maav {
  typedef std::vector<float> Features;
  typedef std::vector<cv::Mat> Images;

  class ExtactInterface {
    public:
      virtual ~ExtactInterface() {}
      virtual void compute(const cv::Mat & image, Features & features) = 0;
  };

  class LearnInterface {
    public:
      virtual ~LearnInterface() {}
      virtual bool load(const std::string & file_path) = 0;
      virtual bool save(const std::string & file_path) = 0;
      virtual void train(const Features & positive,
                         const Features & negative) = 0;
      virtual bool test(const Features & features) = 0;
  };

  void BuildImagePyramid(const cv::Mat & source,
                         const cv::Size & min_size,
                         float scale,
                         Images & pyramid);
  void ApplySlidingWindow(const cv::Mat & source,
                          const cv::Size & window_size,
                          const unsigned int max_horizontal_steps,
                          const unsigned int max_vertical_steps,
                          boost::function<void (const cv::Mat &)> & func);
  void GetRandomPatchFromImage(const cv::Mat & source,
                               const cv::Size & size,
                               cv::Mat & patch);

  class FeatureExtractFunction {
    public:
      FeatureExtractFunction(
          ExtactInterface & extract_interface,
          std::vector<Features> & features_collection
          ) :
        extract_interface_(extract_interface),
        features_collection_(features_collection) {}

      void operator() (const cv::Mat & image) {
        Features features((features_collection_.size()==0?
              0:features_collection_.front().size()));
        extract_interface_.compute(image, features);
        features_collection_.push_back(features);
      }

    private:
      ExtactInterface & extract_interface_;
      std::vector<Features> & features_collection_;
  };

  class NegativeMiningFunction {
    public:
      NegativeMiningFunction(
          ExtactInterface & extract_interface,
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
      ExtactInterface & extract_interface_;
      LearnInterface & learn_interface_;
      std::vector<Features> & negative_features_collection_;
  };
}

#endif
