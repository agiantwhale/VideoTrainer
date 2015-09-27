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
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
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

void TestResult(const maav::ExtractInterface & extract_interface,
                const maav::LearnInterface & learn_interface,
                const cv::Size & window_size,
                int video_source) {
  cv::Mat img, draw;
  cv::VideoCapture video;
  std::vector<cv::Rect> locations;
  maav::TestWindowMethod test(extract_interface, learn_interface, locations);

  // Open the camera.
  video.open(video_source);
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

int main(int argc, char** argv) {
  bool extract, train;
  int width, height,
      max_horizontal_steps,
      max_vertical_steps,
      video_source;
  std::string output_file;
  std::string model_file;
  std::string positive_source_directory;
  std::string negative_source_directory;

  try {
    namespace po=boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
    ("help,h", "Print help messages")
    ("extract,e", po::value<bool>(&extract)->default_value(true), "Specify whether to extract features")
    ("train,t", po::value<bool>(&train)->default_value(false), "Specify whether to train")
    ("camera,c", po::value<int>(&video_source)->default_value(0), "Specify camera to retrieve test feed")
    ("width,w", po::value<int>(&width)->required(), "Specify train window width")
    ("height,h", po::value<int>(&height)->required(), "Specify train window height")
    ("max_horizontal_steps,mh", po::value<int>(&max_horizontal_steps)->default_value(0), "Specify max horizontal steps the window can take")
    ("max_vertical_steps,mh", po::value<int>(&max_vertical_steps)->default_value(0), "Specify max vertical steps the window can take")
    ("positive,p", po::value<std::string>(&positive_source_directory)->default_value(boost::filesystem::current_path().string<std::string>()+"/positive"), "Specify positive video files directory")
    ("negative,n", po::value<std::string>(&negative_source_directory)->default_value(boost::filesystem::current_path().string<std::string>()+"/negative"), "Specify negative video files direcotry")
    ("output,o", po::value<std::string>(&output_file)->default_value(boost::filesystem::current_path().string<std::string>()+"/features.dump"), "Specify an features file for save/load")
    ("model,m", po::value<std::string>(&model_file)->default_value(boost::filesystem::current_path().string<std::string>()+"/result.model"), "Specify an model file for save/load");

    po::variables_map vm;
    po::store(po::command_line_parser(argc,argv).options(desc).run(), vm);

    if (vm.count("help")) {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << desc;
      return 0;
    }

    po::notify(vm);
  } catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  } catch(...) {
    std::cerr << "Exception of unknown type!" << std::endl;
    return 1;
  }

  const cv::Size window_size=cv::Size(width, height);

  std::vector<maav::Features> features_collection;
  std::vector<unsigned int> divider;

  maav::HOGExtractor extractor(window_size);

  if(extract) {
    maav::FeatureExtractMethod extract_method(extractor, features_collection);
    boost::function<void (const cv::Mat &)> f=boost::ref(extract_method);

    boost::filesystem::path positive_dir(positive_source_directory);
    if(boost::filesystem::is_directory(positive_dir)) {
      boost::filesystem::directory_iterator dir_iter(positive_dir), eod;

      unsigned int counter=0;
      BOOST_FOREACH(boost::filesystem::path const &file_path, std::make_pair(dir_iter, eod)) {
        maav::LoadEachFrameFromFile(file_path.string<std::string>(), f);
        for(unsigned int i=0;i<(features_collection.size()-divider.size());i++) {
          divider.push_back(counter);
        }
        counter++;
      }
    }

    boost::filesystem::path negative_dir(positive_source_directory);
    if(boost::filesystem::is_directory(negative_dir)) {
      boost::filesystem::directory_iterator dir_iter(negative_dir), eod;

      BOOST_FOREACH(boost::filesystem::path const &file_path, std::make_pair(dir_iter, eod)) {
        maav::LoadEachFrameFromFile(file_path.string<std::string>(), f);
      }
    }

    std::ofstream file_dump(output_file, std::ofstream::binary);
    cereal::BinaryOutputArchive oarchive(file_dump);
    oarchive(features_collection, divider);
  } else {
    std::ifstream file_dump(output_file, std::ifstream::binary);
    cereal::BinaryInputArchive iarchive(file_dump);
    iarchive(features_collection, divider);
  }

  maav::NeuralNet learner(3, (features_collection.back().size()+divider.back()+2)/2, 5000);

  if(train) {
    learner.train(features_collection, divider);
    learner.save(model_file);
  } else {
    learner.load(model_file);
  }

  TestResult(extractor, learner, window_size, video_source);
}
