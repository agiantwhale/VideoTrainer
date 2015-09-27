/*
 * =====================================================================================
 *
 *       Filename:  hog.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  09/26/2015 19:41:03
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#include "hog.hpp"

maav::HOGExtractor::HOGExtractor(const cv::Size & window_size) :
hog_(),
scale_(true) {
  hog_.winSize=window_size;
}

void maav::HOGExtractor::compute(const cv::Mat & source,
                                 maav::Features & features) {
  cv::Mat dest;
  if(scale_) cv::resize(source, dest, hog_.winSize);
  else maav::GetRandomPatchFromImage(source, hog_.winSize, dest);
  cv::cvtColor(dest, dest, cv::COLOR_BGR2GRAY);
  hog_.compute(dest, features, cv::Size(8, 8), cv::Size(0, 0));
}
