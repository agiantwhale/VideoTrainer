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

  class ExtractInterface {
    public:
      virtual ~ExtractInterface() {}
      virtual void compute(const cv::Mat & image, Features & features) = 0;
  };

  class LearnInterface {
    public:
      virtual ~LearnInterface() {}
      virtual bool load(const std::string & file_path) = 0;
      virtual bool save(const std::string & file_path) = 0;
      virtual void train(const std::vector<Features> & features_collection,
                         const std::vector<unsigned int> divider) = 0;
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
}

#endif
