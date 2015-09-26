/*
 * =====================================================================================
 *
 *       Filename:  video.hpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  09/26/2015 16:17:23
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#ifndef VIDEO_HPP
#define VIDEO_HPP

#include <vector>
#include <string>
#include <boost/function.hpp>
#include <opencv/cv.hpp>

namespace maav {
  void LoadEachFrameFromFile(const std::string & video_path,
                             boost::function<void (const cv::Mat &)> & func);
}

#endif
