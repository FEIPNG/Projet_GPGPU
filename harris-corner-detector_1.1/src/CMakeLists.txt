ADD_LIBRARY(hcd STATIC gaussian.cpp gradient.cpp harris.cpp interpolation.cpp zoom.cpp iio.c)
ADD_LIBRARY(hcd_cuda STATIC gradient_gpu.cu)
