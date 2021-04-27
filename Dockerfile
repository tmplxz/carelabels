FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV OMP_NUM_THREADS=1

RUN apt-get update &&   \
    apt-get install -y  \
        git             \
        vim             \
        python3         \
        python3-pip     \
        wget            \
        sed             \
        unzip

RUN /usr/bin/pip3 install --upgrade pip                                 && \
    cd /tmp                                                             && \
    git clone https://github.com/Breakend/experiment-impact-tracker.git && \
    cd experiment-impact-tracker                                        && \
    /usr/bin/pip3 install .                                             && \
    cd /tmp && git clone https://github.com/wkatsak/py-rapl             && \
    cd py-rapl && /usr/bin/pip3 install .                               && \
    cd /tmp && git clone https://github.com/cleverhans-lab/cleverhans   && \
    cd cleverhans && /usr/bin/pip3 install .

RUN cd /tmp && git clone https://github.com/tmplxz/care-label-certification-suite.git   && \
    cd care-label-certification-suite                                                   && \
    /usr/bin/pip3 install -r requirements.txt                                           && \
    /usr/bin/pip3 install .                                                             && \
    cp lib/* /usr/local/lib/

RUN mkdir -p /usr/local/data/       && \
    cd /tmp                         && \
    wget https://www.dropbox.com/sh/bcawvws67uy0v9s/AADF2TP6SVDEVeUahSUidvwVa?dl=0 && \
    mv "AADF2TP6SVDEVeUahSUidvwVa?dl=0" /tmp/datasets.zip   && \
    unzip  /tmp/datasets.zip -x / -d /usr/local/

# Fix pxpy gpu
RUN sed -i '797s/EXTINF = ext_lib.external(itype,vtype)/EXTINF = ext_lib.external(self.itype,self.vtype)/' /usr/local/lib/python3.8/dist-packages/pxpy/__init__.py && \
    sed -i '873s/EXTINF = ext_lib.external(itype,vtype)/EXTINF = ext_lib.external(self.itype,self.vtype)/' /usr/local/lib/python3.8/dist-packages/pxpy/__init__.py && \
    sed -i '2251s/EXTINF = ext_lib.external(itype,vtype)/EXTINF = ext_lib.external(switch_type(itype), switch_type(vtype))/' /usr/local/lib/python3.8/dist-packages/pxpy/__init__.py
