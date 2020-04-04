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
using namespace std;
using namespace cv;

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
        Size(input_layer->width(), input_layer->height());

    /* Load labels. */
    //ifstream labels(label_file.c_str());
    //CHECK(labels) << "Unable to open labels file " << label_file;
    //string line;
    //while (getline(labels, line))
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



bool Classifier::getRectsFeature(const Mat & img,
    DETECTIONS & d)
{
    cout << "getRectsFeature:80 Detections size:" << d.
        size() << endl;
  for (DETECTION_ROW & dbox:d) {
        cout << "getRectsFeature 81 \n";
        Rect rc =
            Rect(int (dbox.tlwh(0)), int (dbox.tlwh(1)),
            int (dbox.tlwh(2)), int (dbox.tlwh(3)));
        rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
        rc.width = rc.height * 0.5;
        rc.x = (rc.x >= 0 ? rc.x : 0);
        rc.y = (rc.y >= 0 ? rc.y : 0);
        cout << "getRectsFeature 89 \n";
        rc.width =
            (rc.x + rc.width <=
            img.cols ? rc.width : (img.cols - rc.x));
        rc.height =
            (rc.y + rc.height <=
            img.rows ? rc.height : (img.rows - rc.y));
        cout << "getRectsFeature 96 rc.width=" << rc.
            width << ", rc.height=" << rc.height << "\n";
        Mat mattmp = img(rc).clone();
        if (mattmp.empty()) {
            cout << "%%%%%%%%%%%%%%%% mattmp.empty \n";
            for (int j = 0; j < 256; j++) {
                //cout <<  j << ", ";
                dbox.feature[j] = 0.0;
            }
        } else {

            cout << "getRectsFeature 102 \n";
            resize(mattmp, mattmp, Size(64, 128));

            vector < float >features = Predict(mattmp);

            cout << "features.size=" << features.size() <<
                "\n";
            float x;

            for (int j = 0; j < 256; j++) {
                //cout <<  j << ", ";
                x = features[j];
                //cout <<  x << ", ";

                dbox.feature[j] = x;
            }
            cout << "\n ";

        }

        //mats.push_back(mattmp);
    }
    //int count = mats.size();
    return true;
}

vector < float >Classifier::Predict(const Mat & img)
{
    Blob < float >*input_layer = net_->input_blobs()[0];
    cout << "num_channels_," << num_channels_ << "\n";

    cout << "input_geometry_.height," <<
        input_geometry_.height << "\n";

    cout << "input_geometry_.width," <<
        input_geometry_.width << "\n";
    input_layer->Reshape(1, num_channels_,
        input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    vector < Mat > input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a vector */
    Blob < float >*output_layer = net_->output_blobs()[0];
    //for (int i=0; i<256 ;i++){
    //  cout << output_layer <<"\n";
    //}
    const float *begin = output_layer->cpu_data();

    cout << "output_layer->channels() = " <<
        output_layer->channels() << "\n";
    const float *end = begin + output_layer->channels();
    return vector < float >(begin, end);
}

/* Wrap the input layer of the network in separate Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(vector < Mat >
    *input_channels)
{
    Blob < float >*input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const Mat & img,
    vector < Mat > *input_channels)
{
    /* Convert the input image to the input image format of the network. */
    Mat sample;
    cout << "151\n";
    cout << "img.channels()= " << img.channels() << " \n";

    cout << "num_channels_= " << num_channels_ << " \n";
    if (img.channels() == 4 && num_channels_ == 3) {
        cout << "154\n";
        cvtColor(img, sample, COLOR_BGRA2BGR);
    } else
        sample = img;

    cout << "159\n";
    Mat sample_resized;
    if (sample.size() != input_geometry_) {
        cout << "162\n";
        resize(sample, sample_resized, input_geometry_);
    } else
        sample_resized = sample;

    cout << "167\n";
    Mat sample_float;
    sample_resized.convertTo(sample_float, CV_32FC3);
    //float* dstData = sample_float.data;
    float *dstData = sample_float.ptr < float >(0);
    cout << dstData[0] << "\n";
    cout << dstData[1] << "\n";
    cout << dstData[2] << "\n";

    cout << "171\n";
    Mat sample_offseted;
    Mat sample_gained;
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

    cout << mR << "," << mG << "," << mB << "," << sR <<
        "," << sG << "," << sB << "," << "185\n";
    mean_ = (Mat_ < float >(3, 1) << 0.485, 0.456, 0.406);
    cout << "187\n";
    std_ = (Mat_ < float >(3, 1) << 0.229, 0.224, 0.225);
    cout << "189\n";
    //subtract(sample_float, mean_, sample_offseted);
    cout << "191\n";
    //divide(sample_offseted, std_, sample_gained);

    transform(sample_float, sample_gained,
        Matx34f(0.0, 0.0, sB, mB,
            0.0, sG, 0.0, mG, sR, 0.0, 0.0, mR));
    float *dstData2 = sample_gained.ptr < float >(0);
    cout << dstData2[0] << "\n";
    cout << dstData2[1] << "\n";
    cout << dstData2[2] << "\n";
    cout << "194\n";
    split(sample_gained, *input_channels);

    CHECK(reinterpret_cast <
        float *>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
        <<
        "Input channels are not wrapping the input layer of the network.";
}
