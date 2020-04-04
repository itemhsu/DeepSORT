TEMPLATE = app
CONFIG =-g -DCPU_ONLY -DNDEBUG -O2 -DUSE_OPENCV -DUSE_LEVELDB -DUSE_LMDB --std=c++14  -O3   -fPIC 
INCLUDE = -I/usr/local/include/eigen3/ -I/usr/local/include -I/home/itemhsu/src/c/ssd/.build_release/src -I/home/itemhsu/src/c/ssd/src -I/home/itemhsu/src/c/ssd/include -isystem -I/home/itemhsu/src/c/tensorflow/ -I/usr/local/include/eigen3/ 




OBJS := \
    KalmanFilter/kalmanfilter.o \
    KalmanFilter/linear_assignment.o \
    KalmanFilter/nn_matching.o \
    KalmanFilter/track.o \
    KalmanFilter/tracker.o \
    MunkresAssignment/munkres/munkres.o \
    MunkresAssignment/hungarianoper.o \
    DeepAppearanceDescriptor/FeatureTensor.o \
    DeepAppearanceDescriptor/model.o \
    main.o

LIBS :=    -L/home/itemhsu/src/c/ssd/.build_release/lib  -lcaffe -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_video -lopencv_dnn -lboost_system -lboost_filesystem -lboost_regex -lm -lboost_regex  -lglog -lstdc++ -lprotobuf -lz -lm -ldl -lpthread 

#LIBS := -Wl,--whole-archive -ltensorflow_framework -ltensorflow_cc  -Wl,--no-whole-archive -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_video -lopencv_dnn   -lglog

LLIBPATH := -L/usr/local/lib 


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

