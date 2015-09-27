/*
 * =====================================================================================
 *
 *       Filename:  train.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  09/26/2015 15:29:56
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#include "train.hpp"

void maav::BuildImagePyramid(const cv::Mat & source,
                             const cv::Size & min_size,
                             float scale,
                             Images & pyramid) {
  for(int i=1;;i++) {
    const cv::Size scaled_size( source.cols/std::pow(scale,i),
                                source.rows/std::pow(scale,i));
    if(scaled_size.width<min_size.width||scaled_size.height<min_size.height) break;

    cv::Mat scaled_source;
    cv::resize(source, scaled_source, scaled_size);

    pyramid.push_back(scaled_source.clone());
  }
}

void maav::ApplySlidingWindow(const cv::Mat & source,
                              const cv::Size & window_size,
                              const unsigned int max_horizontal_steps,
                              const unsigned int max_vertical_steps,
                              boost::function<void (const cv::Mat &)> & func) {
  const unsigned int horizontal_padding=
    max_horizontal_steps!=0?
    std::max(
        source.cols/
        std::max((int)max_horizontal_steps,1)-window_size.width,0):0;
  const unsigned int vertical_padding=
    max_vertical_steps!=0?
    std::max(
        source.rows/
        std::max((int)max_vertical_steps,1)-window_size.height,0):0;

  cv::Rect box;
  box.width=window_size.width;
  box.height=window_size.height;
  cv::Mat sliding_window;
  int x=0,y=0;
  while(y<source.rows) {
    while(x<source.cols) {
      box.x=x;
      box.y=y;
      sliding_window=(source)(box);
      func(sliding_window);
      x+=(horizontal_padding+window_size.width);
    }

    x=0;
    y+=(vertical_padding+window_size.height);
  }
}

void maav::GetRandomPatchFromImage(const cv::Mat & source,
                                   const cv::Size & size,
                                   cv::Mat & patch) {
  srand(time(NULL));

  cv::Rect box;
  box.width = size.width;
  box.height = size.height;
  box.x = rand() % (source.cols-box.width);
  box.y = rand() % (source.rows-box.height);
  patch = (source)(box).clone();
}
