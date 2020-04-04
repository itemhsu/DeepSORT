// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/types_c.h>

#include <opencv2/bgsegm.hpp>
#include "./DeepAppearanceDescriptor/classification.h"

#include "KalmanFilter/tracker.h"
using namespace std;
using namespace cv;

const char *keys =
    "{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
    "{image i        |<none>| input image   }"
    "{video v       |<none>| input video   }";


// yolo parameter
// Initialize the parameters
const float confThreshold = 0.5;        // Confidence threshold
const float nmsThreshold = 0.4; // Non-maximum suppression threshold
const int inpWidth = 416;       // Width of network's input image
const int inpHeight = 416;      // Height of network's input image
vector < string > classes;

//Deep SORT parameter

const int nn_budget = 100;
const float max_cosine_distance = 0.2;
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat & frame,
    const vector < Mat > &out, DETECTIONS & d);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top,
    int right, int bottom, Mat & frame);

// Get the names of the output layers
vector < String > getOutputsNames(const dnn::Net & net);

void get_detections(DETECTBOX box, float confidence,
    DETECTIONS & d);
int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about
        ("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    //deep SORT
    tracker mytracker(max_cosine_distance, nn_budget);
    //yolo
    // Load names of classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    cout << "### Import coco.names ###" << endl;
    while (getline(ifs, line)) {
        classes.push_back(line);
        cout << line << endl;
    }

    // Give the configuration and weight files for the model
    String modelConfiguration = "yolov3.cfg";
    String modelWeights = "yolov3.weights";

    // Load the network
    dnn::Net net =
        dnn::readNetFromDarknet(modelConfiguration,
        modelWeights);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);


    cout <<
        "### main 89: dnn::readNetFromDarknet ###" << endl;
    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;

    try {

        outputFile = "yolo_out_cpp.avi";
        if (parser.has("image")) {
            // Open the image file
            str = parser.get < String > ("image");
            ifstream ifile(str);
            if (!ifile)
                throw("error");
            cap.open(str);
            str.replace(str.end() - 4, str.end(),
                "_yolo_out_cpp.jpg");
            outputFile = str;
        } else if (parser.has("video")) {
            // Open the video file
            str = parser.get < String > ("video");
            ifstream ifile(str);
            if (!ifile)
                throw("error");
            cap.open(str);
            str.replace(str.end() - 4, str.end(),
                "_yolo_out_cpp.avi");
            outputFile = str;
        } else {

            cout << "### Capture video 0 120###" << endl;
            cap.open(0);

            cout << "### Capture video 0 122###" << endl;
        }
        // Open the webcaom
        // else cap.open(parser.get<int>("device"));

    }
    catch( ...) {
        cout <<
            "Could not open the input image/video stream" <<
            endl;
        return 0;
    }

    cout << "### main 135: Capture video ###" << endl;
    // Get the video writer initialized to save the output video
    if (!parser.has("image")) {
        video.open(outputFile, VideoWriter::fourcc('M',
                'J', 'P', 'G'), 28.0,
            Size(static_cast <
                int >(cap.get(CAP_PROP_FRAME_WIDTH)),
                static_cast <
                int >(cap.get(CAP_PROP_FRAME_HEIGHT))));
    }
    // Create a window
    static const string kWinName =
        "Multiple Object Tracking";
    namedWindow(kWinName, WINDOW_NORMAL);


    // Process frames.
    while (waitKey(1) < 0) {
        // get frame from the video
        cout << "### main 154: while loop ###" << endl;
        cap >> frame;

        cout << "### main 156: while loop ###" << endl;
        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " <<
                outputFile << endl;
            waitKey(3000);
            break;
        }
        // Create a 4D blob from a frame.

        cout << "### main 167: while loop 4D blob ###"
            << endl;
        dnn::blobFromImage(frame, blob, 1 / 255.0,
            cvSize(inpWidth, inpHeight), Scalar(0, 0,
                0), true, false);

        //Sets the input to the network
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector < Mat > outs;
        net.forward(outs, getOutputsNames(net));

        // Remove the bounding boxes with low confidence
        DETECTIONS detections;
        postprocess(frame, outs, detections);


        cout << "Detections size:" << detections.size()
            << endl;

        cout << "### main 185: before FeatureTensor ###"
            << endl;
        bool tfIns =
            Classifier::
            getInstance()->getRectsFeature(frame,
            detections);


        cout << "### main 187: before FeatureTensor ###"
            << endl;
        if (tfIns) {
            cout << "### main 188 ###" << endl;
            cout << "Tensorflow get feature succeed!"
                << endl;
            mytracker.predict();
            cout << "### main 189 ###" << endl;
            mytracker.update(detections);
            cout << "### main 190 ###" << endl;
            vector < RESULT_DATA > result;
          for (Track & track:mytracker.tracks) {
                cout << "### main 196 ###" << endl;
                if (!track.is_confirmed()
                    || track.time_since_update > 1)
                    continue;
                result.push_back(make_pair(track.track_id,
                        track.to_tlwh()));
            }
            for (unsigned int k = 0; k < detections.size();
                k++) {
                cout << "### main 205 ###" << endl;
                DETECTBOX tmpbox = detections[k].tlwh;
                Rect rect(tmpbox(0), tmpbox(1),
                    tmpbox(2), tmpbox(3));
                rectangle(frame, rect, Scalar(0, 0,
                        255), 4);
                // cvScalar的储存顺序是B-G-R，CV_RGB的储存顺序是R-G-B

                for (unsigned int k = 0; k < result.size();
                    k++) {
                    cout << "### main 215 ###" << endl;
                    DETECTBOX tmp = result[k].second;
                    Rect rect = Rect(tmp(0), tmp(1), tmp(2),
                        tmp(3));
                    rectangle(frame, rect, Scalar(255,
                            255, 0), 2);

                    string label =
                        format("%d", result[k].first);
                    putText(frame, label,
                        Point(rect.x, rect.y),
                        FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(255, 255, 0), 2);
                }
            }
        } else {
            cout << "### main 232 ###" << endl;
            cout << "Tensorflow get feature failed!" <<
                endl;;
        }
        cout << "### main 237 ###" << endl;
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector < double >layersTimes;
        double freq = getTickFrequency() / 1000;
        cout << "### main 238 ###" << endl;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label =
            format
            ("Inference time for a frame : %.2f ms", t);
        cout << "### main 239 ###" << endl;
        putText(frame, label, Point(0, 15),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

        cout << "### main 250 ###" << endl;
        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        cout << "### main 260 ###" << endl;
        if (parser.has("image")){
            cout << "### main 261 ###" << endl;
            imwrite(outputFile, detectedFrame);
        }
        else{
            cout << "### main 261 ###" << endl;
            //video.write(detectedFrame);//matthew debug
        }
        cout << "### main 268 ###" << endl;

        imshow(kWinName, frame);
        cout << "### main 271 ###" << endl;

    }

    cout << "### main 283 ###" << endl;
    cap.release();
    if (!parser.has("image"))
        video.release();

    cout << "### main 288 ###" << endl;
    return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat & frame,
    const vector < Mat > &outs, DETECTIONS & d)
{
    vector < int >classIds;
    vector < float >confidences;
    vector < Rect > boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float *data = (float *) outs[i].data;
        for (int j = 0; j < outs[i].rows;
            ++j, data += outs[i].cols) {
            Mat scores =
                outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0,
                &classIdPoint);
            if (static_cast < float >(confidence) >
                (confThreshold)) {
                int centerX = (int) (data[0] * frame.cols);
                int centerY = (int) (data[1] * frame.rows);
                int width = (int) (data[2] * frame.cols);
                int height = (int) (data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float) confidence);
                boxes.push_back(Rect(left, top, width,
                        height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector < int >indices;
    dnn::NMSBoxes(boxes, confidences, confThreshold,
        nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = static_cast < size_t > (indices[i]);
        Rect box = boxes[idx];
        //目标检测 代码的可视化
        //drawPred(classIds[idx], confidences[idx], box.x, box.y,box.x + box.width, box.y + box.height, frame);

        get_detections(DETECTBOX(box.x, box.y, box.width,
                box.height), confidences[idx], d);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top,
    int right, int bottom, Mat & frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top),
        Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
        label = classes[classId] + ":" + label;
    }
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize =
        getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1,
        &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left,
            top - round(1.5 * labelSize.height)),
        Point(left + round(1.5 * labelSize.width),
            top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top),
        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector < String > getOutputsNames(const dnn::Net & net)
{
    static vector < String > names;
    if (names.empty()) {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector < int >outLayers =
            net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector < String > layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void get_detections(DETECTBOX box, float confidence,
    DETECTIONS & d)
{
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;          //DETECTBOX(x, y, w, h);

    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}
