FROM python:3.7
MAINTAINER Eko Rudiawan <eko.rudiawan@gmail.com>

RUN apt-get update 
# For compile
RUN apt-get install -y build-essential 
RUN apt-get install -y cmake 
RUN apt-get install -y git 
RUN apt-get install -y pkg-config 
RUN apt-get install -y wget 
RUN apt-get install -y unzip
# OpenCV Require
RUN apt-get install -y libjpeg-dev 
RUN apt-get install -y libtiff-dev 
RUN apt-get install -y libpng-dev 
# RUN apt-get install -y libjasper-dev
# Video
RUN apt-get install -y libavcodec-dev 
RUN apt-get install -y libavformat-dev 
RUN apt-get install -y libswscale-dev 
RUN apt-get install -y libv4l-dev
# GUI
RUN apt-get install -y libgtk2.0-dev
# Optimization
RUN apt-get install -y libatlas-base-dev gfortran
# Python Libraries
RUN pip install numpy
RUN pip install matplotlib
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install scikit-image
# Lokasi install di home
WORKDIR /home/
# Versi OpenCV
ENV OPENCV_VERSION="3.4.6"
# RUN cd $WORKDIR$
RUN git clone -b 3.4 https://github.com/ekorudiawan/opencv.git
# OpenCV Contrib
RUN wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip ${OPENCV_VERSION}.zip 
RUN rm -rf ${OPENCV_VERSION}.zip
RUN mv opencv opencv-${OPENCV_VERSION}
RUN mkdir opencv-${OPENCV_VERSION}/cmake_binary
RUN cd /home/opencv-${OPENCV_VERSION}/cmake_binary \ 
# Build OpenCV
    && cmake -DBUILD_TIFF=ON \
    -DBUILD_opencv_java=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=/home/opencv_contrib-${OPENCV_VERSION}/modules \
    -DWITH_CUDA=OFF \
    -DWITH_OPENGL=ON \
    -DWITH_OPENCL=ON \
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DWITH_V4L=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$(python3.7 -c "import sys; print(sys.prefix)") \
    -DPYTHON_EXECUTABLE=$(which python3.7) \
    -DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_PACKAGES_PATH=$(python3.7 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    .. \
    && make -j8 install 
RUN rm -rf /home/opencv-${OPENCV_VERSION} \
    rm -rf /home/opencv_contrib-${OPENCV_VERSION}
# COPY . .
CMD ["bash"]
