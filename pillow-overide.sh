wget -q https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.5.tar.gz -O libjpeg-turbo.tar.gz && \
    echo "b3090cd37b5a8b3e4dbd30a1311b3989a894e5d3c668f14cbc6739d77c9402b7 libjpeg-turbo.tar.gz" | sha256sum -c && \
    tar xf libjpeg-turbo.tar.gz && \
    rm libjpeg-turbo.tar.gz && \
    cd libjpeg-turbo* && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DREQUIRE_SIMD=On -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cd ../../ && \
    rm -rf libjpeg-turbo*

