#ifndef __caffe_deep_net__
#define __caffe_deep_net__
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model.h"

using namespace caffe;          // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair < string, float >Prediction;
/*
int l2morm256(float in[256], float out[256]){
  float sum_of_square=0.0;
  float sqrt_sum_of_square=0.0;
  for (int i=0; i<256 ;i++){
    sum_of_square+= in[i] * in[i];
  }
  sqrt_sum_of_square=  sqrt(sum_of_square);
  for (int i=0; i<256 ;i++){
    out[i]=in[i]/sqrt_sum_of_square; 
  }
  return 0;
}
*/
class Classifier {
  public:
    Classifier(const string & model_file,
        const string & trained_file);

    bool getRectsFeature(const cv::Mat & img,
        DETECTIONS & d);
    static Classifier *getInstance();
  private:
    static Classifier *instance;

    std::vector < float >Predict(const cv::Mat & img);

    void WrapInputLayer(std::vector < cv::Mat >
        *input_channels);
    int l2morm256(float in[256], float out[256]);

    void Preprocess(const cv::Mat & img,
        std::vector < cv::Mat > *input_channels);

  private:
    shared_ptr < Net < float >>net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    cv::Mat std_;
    std::vector < string > labels_;
};
#endif
