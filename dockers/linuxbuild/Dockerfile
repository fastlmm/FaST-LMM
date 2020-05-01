FROM quay.io/pypa/manylinux2010_x86_64
ENV PATH="${PATH}:/opt/python/cp37-cp37m/bin:/opt/rh/devtoolset-8/root/usr/bin"
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | bash
RUN yum install git-lfs -y
RUN git clone --branch openBLAS https://github.com/fastlmm/FaST-LMM.git fastlmm
WORKDIR /fastlmm
RUN git lfs install
RUN git lfs fetch
RUN git lfs checkout
RUN pip install cython
RUN pip install virtualenv
ENV PYTHONPATH="/fastlmm"
RUN pip install -r requirements.txt
RUN virtualenv fresh