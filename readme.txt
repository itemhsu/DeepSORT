############### Caffe support ########
1. make sure caffee compiled 
2. modify the caffee path in Makefile
3. make 
4. install UVC webcam
5. ./main


 



################ Old #############
1. bazel build //tensorflow:libtensorflow_cc.so
2. make -C DeepAppearanceDescriptor
3. make
4. cp DeepAppearanceDescriptor/libDeepAppearanceDescriptor.so /usr/local/lib
5. sudo ldconfig 

################# tf cc lib build ######################
./configure
bazel build --config=opt //tensorflow:libtensorflow_cc.so

###### below is static link solution. not recommand ######

./tensorflow/contrib/makefile/build_all_linux.sh
 - tenorflow/contrib/makefile/gen/lib/libtensorflow-core.a - tensorflow/contrib/makefile/gen/protobuf/lib/libprotobuf.a -tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11/libnsync.a

./tensorflow/contrib/makefile/build_all_linux.sh

sudo cp /home/itemhsu/src/c/tensorflow/tensorflow/contrib/makefile/gen/protobuf-host/lib/* /usr/local/lib -R

sudo ldconfig 


vi /home/itemhsu/src/c/tensorflow/tensorflow/contrib/makefile/tf_op_files.txt

add these line at end of the file
tensorflow/core/ops/bitwise_ops.cc
tensorflow/core/ops/lookup_ops.cc
tensorflow/core/ops/dataset_ops.cc
tensorflow/core/ops/stateless_random_ops.cc
tensorflow/core/ops/decode_proto_ops.cc
tensorflow/core/ops/encode_proto_ops.cc
tensorflow/core/ops/spectral_ops.cc
tensorflow/core/ops/stateful_random_ops.cc

