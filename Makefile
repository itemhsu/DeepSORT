TEMPLATE = app
CONFIG =  --std=c++14  -O3   -fPIC 
INCLUDE = -I/home/itemhsu/src/c/tensorflow/ -I/home/itemhsu/.cache/bazel/_bazel_itemhsu/eeac80d65a0203b7e393eff1c69755ff/execroot/org_tensorflow/bazel-out/k8-py2-opt/bin/ -I/home/itemhsu/.cache/bazel/_bazel_itemhsu/eeac80d65a0203b7e393eff1c69755ff/external/protobuf_archive/src/ -I/usr/local/include/eigen3/ -I/home/itemhsu/.cache/bazel/_bazel_itemhsu/eeac80d65a0203b7e393eff1c69755ff/external/com_google_absl/



OBJS := \
    KalmanFilter/kalmanfilter.o \
    KalmanFilter/linear_assignment.o \
    KalmanFilter/nn_matching.o \
    KalmanFilter/track.o \
    KalmanFilter/tracker.o \
    MunkresAssignment/munkres/munkres.o \
    MunkresAssignment/hungarianoper.o \
    main.o

LIBS :=   -L/home/itemhsu/src/c/tensorflow/tensorflow/contrib/makefile/gen/protobuf-host/lib -lDeepAppearanceDescriptor  -ltensorflow_framework -ltensorflow  -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_video -lopencv_dnn   -lglog -lstdc++ -lprotobuf -lz -lm -ldl -lpthread 

#LIBS := -Wl,--whole-archive -ltensorflow_framework -ltensorflow_cc  -Wl,--no-whole-archive -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_video -lopencv_dnn   -lglog

LLIBPATH := -L/usr/local/lib -L./DeepAppearanceDescriptor


CC:=g++
exe:=main

all:$(OBJS)
	$(CC) $(CONFIG) -o $(exe) $(OBJS) $(LIBS) $(LLIBPATH)
%.o:%.cpp
	$(CC) $(CONFIG) $(INCLUDE) -c $^ -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) 
	rm -f $(exe) 

