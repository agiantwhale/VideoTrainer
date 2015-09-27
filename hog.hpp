/*
 * =====================================================================================
 *
 *       Filename:  hog.hpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  09/26/2015 19:34:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#ifndef HOG_HPP
#define HOG_HPP

#include "train.hpp"

namespace maav {
  class HOGExtractor : public ExtractInterface {
    public:
      HOGExtractor(const cv::Size & window_size);
      virtual ~HOGExtractor() {}
      virtual void compute(const cv::Mat & image, Features & features) const;

      bool scale_;
    private:
      cv::HOGDescriptor hog_;
  };
}

#endif
