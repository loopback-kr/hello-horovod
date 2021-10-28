FROM horovod/horovod:latest


### Set environment variables
# Change default Shell to bash
SHELL ["/bin/bash", "-c"]
# Set Timezone
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
# Set python encoding type
ENV PYTHONIOENCODING=UTF-8


### Change settings
# Change bash shell prompt color of root account
RUN sed -i 's/    xterm-color) color_prompt=yes;;/    #xterm-color) color_prompt=yes;;\n    xterm-color|*-256color) color_prompt=yes;;/' /root/.bashrc
# Change original green color of bash shell prompt to red color
RUN sed -i 's/    PS1=\x27${debian_chroot:+(\$debian_chroot)}\\\[\\033\[01;32m\\\]\\u@\\h\\\[\\033\[00m\\\]:\\\[\\033\[01;34m\\\]\\w\\\[\\033\[00m\\\]\\\$ \x27/    PS1=\x27\${debian_chroot:+(\$debian_chroot)}\\\[\\033\[01;31m\\\]\\u@\\h\\\[\\033\[00m\\\]:\\\[\\033\[01;34m\\\]\\w\\\[\\033\[00m\\\]\\\$ \x27/' /root/.bashrc
# Install essential packages
RUN apt update \
    && apt install -y \
        tzdata \
        git \
        ca-certificates
# Change Ubuntu repository address to Kakao server in Republic of Korea
RUN sed -i 's/^deb http:\/\/archive.ubuntu.com/deb http:\/\/ftp.kaist.ac.kr/g' /etc/apt/sources.list
RUN sed -i 's/^deb http:\/\/security.ubuntu.com/deb http:\/\/ftp.kaist.ac.kr/g' /etc/apt/sources.list


### Setup requirements
# Install OpenCV dependencies
RUN apt update \
    && apt install -y \
        libglu1-mesa-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev

# Configure default Pypi repository address and default progress bar style
RUN mkdir -p ~/.config/pip \
    && echo -e \
"[global]\n"\
"index-url=http://ftp.kaist.ac.kr/pypi/simple/\n"\
"trusted-host=ftp.kaist.ac.kr\n"\
"progress-bar=emoji"\
        > ~/.config/pip/pip.conf \
    && pip install -U pip

### Postprocessing & Cleaning
# Clean the cache
RUN apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 cache purge
# Set workspace
WORKDIR /root/workspace
