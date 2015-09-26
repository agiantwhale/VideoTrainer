/*
 * =====================================================================================
 *
 *       Filename:  video.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  09/26/2015 16:26:54
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#include "video.hpp"

#include <cstdlib>
#include <time.h>

void maav::LoadEachFrameFromFile(const std::string & video_path,
                             boost::function<void (const cv::Mat &)> & func) {
  cv::VideoCapture video(video_path);
  if(!video.isOpened()) return;
  cv::Mat frame;
  while(video.read(frame)) {
    if(frame.empty()) break;
    func(frame);
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
