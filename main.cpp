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
#include <fstream>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>

namespace maav {
  class FeatureExtractMethod {
    public:
      FeatureExtractMethod(
          const ExtractInterface & extract_interface,
          std::vector<Features> & features_collection
          ) :
        extract_interface_(extract_interface),
        features_collection_(features_collection) {}

      void operator() (const cv::Mat & image) const {
        std::cout << "Extracted features " << features_collection_.size() << "..." << std::endl;
        Features features((features_collection_.size()==0?
              0:features_collection_.front().size()));
        extract_interface_.compute(image, features);
        features_collection_.push_back(features);
      }

    private:
      const ExtractInterface & extract_interface_;
      std::vector<Features> & features_collection_;
  };

  class NegativeMiningMethod {
    public:
      NegativeMiningMethod(
          const ExtractInterface & extract_interface,
          const LearnInterface & learn_interface,
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
      const ExtractInterface & extract_interface_;
      const LearnInterface & learn_interface_;
      std::vector<Features> & negative_features_collection_;
  };

  class TestWindowMethod {
    public:
      TestWindowMethod (
          const ExtractInterface & extract_interface,
          const LearnInterface & learn_interface,
          std::vector<cv::Rect> & locations
          ) :
        extract_interface_(extract_interface),
        learn_interface_(learn_interface),
        locations_(locations) {}

      void operator() (const cv::Mat & image, const cv::Rect & location) {
        Features features;
        extract_interface_.compute(image, features);
        if(learn_interface_.test(features)) locations_.push_back(location);
      }

    private:
      const ExtractInterface & extract_interface_;
      const LearnInterface & learn_interface_;
      std::vector<cv::Rect> & locations_;
  };
}

void test_it(const maav::ExtractInterface & extract_interface,
             const maav::LearnInterface & learn_interface,
             const cv::Size & window_size) {
  cv::Mat img, draw;
  cv::VideoCapture video;
  std::vector<cv::Rect> locations;
  maav::TestWindowMethod test(extract_interface, learn_interface, locations);

  // Open the camera.
  video.open(1);
  if( !video.isOpened() )
  {
    std::cerr << "Unable to open the device 0" << std::endl;
    exit( -1 );
  }

  char key=0;
  bool end_of_process = false;
  while( !end_of_process )
  {
    video >> img;
    if( img.empty() )
        break;

    draw = img.clone();

    locations.clear();

    boost::function<void (const cv::Mat &, const cv::Rect &)> f=boost::ref(test);
    maav::ApplySlidingWindow(draw, window_size, 0, 0, f);

    if( !locations.empty() )
    {
      std::vector<cv::Rect>::const_iterator loc = locations.begin();
      std::vector<cv::Rect>::const_iterator end = locations.end();
      for( ; loc != end ; ++loc )
      {
        cv::rectangle(draw, *loc, cv::Scalar( 255, 0, 255 ), 2);
      }
    }

    imshow( "Video", draw );
    key = (char)cv::waitKey( 10 );
    if( 27 == key )
        end_of_process = true;
  }
}

int main() {
  const cv::Size window_size=cv::Size(72,128);

  std::vector<maav::Features> features_collection;
  std::vector<unsigned int> divider;

  maav::HOGExtractor extractor(window_size);

  maav::FeatureExtractMethod extract_method(extractor, features_collection);
  boost::function<void (const cv::Mat &)> f=boost::ref(extract_method);

  maav::LoadEachFrameFromFile("/Users/iljae/Development/MHackers/data/positive.MOV", f);
  for(unsigned int i=0;i<(features_collection.size()-divider.size());i++) {
    divider.push_back(0);
  }

  std::cout << "Positive extraction is done!" << std::endl;

  extractor.scale_=false;
  maav::LoadEachFrameFromFile("/Users/iljae/Development/MHackers/data/negative.MOV", f);

  std::cout << "Negative extraction is done!" << std::endl;

  {
    std::ofstream file_dump("/Users/iljae/Development/MHackers/data/features.dump", std::ofstream::binary);
    cereal::BinaryOutputArchive oarchive(file_dump);
    oarchive(features_collection, divider);
  }
  // {
  //   std::ifstream file_dump("/Users/iljae/Development/MHackers/data/features.dump", std::ifstream::binary);
  //   cereal::BinaryInputArchive iarchive(file_dump);
  //   iarchive(features_collection, divider);
  // }

  maav::NeuralNet learner(3, (features_collection.back().size()+divider.back()+2)/2, 50);

  learner.train(features_collection, divider);
  learner.save("/Users/iljae/Development/MHackers/data/trained");
  // learner.load("/Users/iljae/Development/MHackers/data/trained");
  // test_it(extractor, learner, window_size);
}
