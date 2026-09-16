python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./nelssa_comm.proto

#/usr/bin/c++ -O3 -Wall -shared -std=c++17 -fPIC -I/usr/include -I/usr/include/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu $(python3 -m pybind11 --includes) nelssa_wrapper.cpp -o nelssa_wrapper$(python3-config --extension-suffix) -libverbs

/usr/bin/c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  $(python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))") \
  nelssa_wrapper.cpp \
  -o nelssa_wrapper$(python3-config --extension-suffix) \
  $(python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join('-L'+p for p in library_paths()))") \
  -ltorch -ltorch_cpu -lc10 -ltorch_python \
  -libverbs
