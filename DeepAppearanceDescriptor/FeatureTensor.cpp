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
#include "classification.h"
Classifier *Classifier::instance = NULL;

Classifier *Classifier::getInstance()
{
    if (instance == NULL) {
        instance =
            new Classifier("2deepNet.prototxt",
            "2deepNet.caffemodel");
    }
    return instance;
}


Classifier::Classifier(const string & model_file,
    const string & trained_file)
{
    Caffe::set_mode(Caffe::CPU);

    /* Load the network. */
    net_.reset(new Net < float >(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(),
        1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(),
        1) << "Network should have exactly one output.";

    Blob < float >*input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ =
        cv::Size(input_layer->width(),
        input_layer->height());

    /* Load labels. */
    //std::ifstream labels(label_file.c_str());
    //CHECK(labels) << "Unable to open labels file " << label_file;
    //string line;
    //while (std::getline(labels, line))
    //  labels_.push_back(string(line));

    //Blob<float>* output_layer = net_->output_blobs()[0];
    //CHECK_EQ(labels_.size(), output_layer->channels())
    //  << "Number of labels is different from the output layer dimension.";
}

int Classifier::l2morm256(float in[256], float out[256])
{
    float sum_of_square = 0.0;
    float sqrt_sum_of_square = 0.0;
    for (int i = 0; i < 256; i++) {
        sum_of_square += in[i] * in[i];
    }
    sqrt_sum_of_square = sqrt(sum_of_square);
    for (int i = 0; i < 256; i++) {
        out[i] = in[i] / sqrt_sum_of_square;
    }
    return 0;
}

/* Return the top N predictions. */
std::vector <
    float >Classifier::Classify(const cv::Mat & img, int N)
{
    std::vector < float >rawFeature = Predict(img);
    //std::vector<float> L2Feature[256];
    std::vector < float >L2Feature;
    float _in[256], _out[256];
    for (int i = 0; i < 256; i++) {
        std::cout << i << " in : " << rawFeature[i] << "\n";
        _in[i] = rawFeature[i];
    }
    l2morm256(_in, _out);
    L2Feature.insert(L2Feature.end(), _out, _out + 256);
    for (int i = 0; i < 256; i++) {
        //L2Feature.insert(L2Feature.end(), _out[i], _out[i]);
        //begin[i]=begin[i]/sqrt_sum_of_square; 
        std::cout << i << " out : " << _out[i] << "\n";
    }

    return L2Feature;
}

bool Classifier::getRectsFeature(const cv::Mat & img,
    DETECTIONS & d)
{
    std::vector < cv::Mat > mats;
  for (DETECTION_ROW & dbox:d) {
        cv::Rect rc =
            cv::Rect(int (dbox.tlwh(0)), int (dbox.tlwh(1)),
            int (dbox.tlwh(2)), int (dbox.tlwh(3)));
        rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
        rc.width = rc.height * 0.5;
        rc.x = (rc.x >= 0 ? rc.x : 0);
        rc.y = (rc.y >= 0 ? rc.y : 0);
        rc.width =
            (rc.x + rc.width <=
            img.cols ? rc.width : (img.cols - rc.x));
        rc.height =
            (rc.y + rc.height <=
            img.rows ? rc.height : (img.rows - rc.y));
        cv::Mat mattmp = img(rc).clone();
        cv::resize(mattmp, mattmp, cv::Size(64, 128));
        std::vector < float >features = Predict(mattmp);
        for (int j = 0; j < 256; j++) {
            dbox.feature[j] = features[j];
        }

        //mats.push_back(mattmp);
    }
    //int count = mats.size();
    return true;
}

std::vector <
    float >Classifier::Predict(const cv::Mat & img)
{
    Blob < float >*input_layer = net_->input_blobs()[0];
    std::cout << "num_channels_," << num_channels_ << "\n";
    std::
        cout << "input_geometry_.height," <<
        input_geometry_.height << "\n";
    std::
        cout << "input_geometry_.width," << input_geometry_.
        width << "\n";
    input_layer->Reshape(1, num_channels_,
        input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector < cv::Mat > input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob < float >*output_layer = net_->output_blobs()[0];
    //for (int i=0; i<256 ;i++){
    //  std::cout << output_layer <<"\n";
    //}
    const float *begin = output_layer->cpu_data();
    std::
        cout << "output_layer->channels() = " <<
        output_layer->channels() << "\n";
    const float *end = begin + output_layer->channels();
    return std::vector < float >(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector < cv::Mat >
    *input_channels)
{
    Blob < float >*input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1,
            input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat & img,
    std::vector < cv::Mat > *input_channels)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    std::cout << "151\n";
    std::cout << "img.channels()= " << img.
        channels() << " \n";
    std::
        cout << "num_channels_= " << num_channels_ << " \n";
    if (img.channels() == 4 && num_channels_ == 3) {
        std::cout << "154\n";
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    } else
        sample = img;

    std::cout << "159\n";
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_) {
        std::cout << "162\n";
        cv::resize(sample, sample_resized, input_geometry_);
    } else
        sample_resized = sample;

    std::cout << "167\n";
    cv::Mat sample_float;
    sample_resized.convertTo(sample_float, CV_32FC3);
    //float* dstData = sample_float.data;
    float *dstData = sample_float.ptr < float >(0);
    std::cout << dstData[0] << "\n";
    std::cout << dstData[1] << "\n";
    std::cout << dstData[2] << "\n";
    //std::cout << dstData[64*128] << "\n";
    //std::cout << dstData[64*128*2] << "\n";
    //std::cout << dstData[64*128*3] << "\n";
    std::cout << "171\n";
    cv::Mat sample_offseted;
    cv::Mat sample_gained;
    /* 
       transform_param {
       mean_value: 0.485
       mean_value: 0.456
       mean_value: 0.406
       std_value: 0.229
       std_value: 0.224
       std_value: 0.225
       }
     */

    float mR = 0.406;
    float mG = 0.456;
    float mB = 0.485;
    float sR = 0.225;
    float sG = 0.224;
    float sB = 0.229;


    mR /= -sR;
    mG /= -sG;
    mB /= -sB;
    sR = 1.0 / sR / 255.0;
    sG = 1.0 / sG / 255.0;
    sB = 1.0 / sB / 255.0;
    std::
        cout << mR << "," << mG << "," << mB << "," << sR <<
        "," << sG << "," << sB << "," << "185\n";
    mean_ =
        (cv::Mat_ < float >(3, 1) << 0.485, 0.456, 0.406);
    std::cout << "187\n";
    std_ =
        (cv::Mat_ < float >(3, 1) << 0.229, 0.224, 0.225);
    std::cout << "189\n";
    //cv::subtract(sample_float, mean_, sample_offseted);
    std::cout << "191\n";
    //cv::divide(sample_offseted, std_, sample_gained);

    cv::transform(sample_float, sample_gained,
        cv::Matx34f(0.0, 0.0, sB, mB,
            0.0, sG, 0.0, mG, sR, 0.0, 0.0, mR));
    float *dstData2 = sample_gained.ptr < float >(0);
    std::cout << dstData2[0] << "\n";
    std::cout << dstData2[1] << "\n";
    std::cout << dstData2[2] << "\n";
    std::cout << "194\n";
    cv::split(sample_gained, *input_channels);

    CHECK(reinterpret_cast <
        float *>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
        <<
        "Input channels are not wrapping the input layer of the network.";
}
