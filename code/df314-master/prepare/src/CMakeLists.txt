CUDA_ADD_LIBRARY(proc SHARED proc.cu)
ADD_LIBRARY(utils SHARED cnpy.cpp utils.cpp)

ADD_EXECUTABLE(generate_result main.cpp)
TARGET_LINK_LIBRARIES(generate_result utils proc -lz -lpthread -lstdc++)
