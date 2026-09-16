0. Setup
## GPU Server
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./nelssa_comm.proto

/usr/bin/c++ -O3 -Wall -shared -std=c++17 -fPIC -I/usr/include -I/usr/include/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu $(python3 -m pybind11 --includes) nelssa_wrapper.cpp -o nelssa_wrapper$(python3-config --extension-suffix) -libverbs

## PNM Server (CPU PoC)
protoc -I. --grpc_out=. --cpp_out=. ./nelssa_comm.proto --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`
./compile.sh
./build/nelssa_host


1. NelssaClient 
from nelssa_comm.NelssaClient import NelssaClient

