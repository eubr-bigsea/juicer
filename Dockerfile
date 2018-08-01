FROM python:2.7-alpine as base

FROM alpine:3.7 as mesos_build

ARG MESOS_VERSION=1.6.1
ARG MESOS_PKG_URL=http://www.apache.org/dist/mesos/${MESOS_VERSION}/mesos-${MESOS_VERSION}.tar.gz

RUN apk add --no-cache \
      apr-dev \
      curl-dev \
      cyrus-sasl-crammd5 \
      cyrus-sasl-dev \
      fts-dev \
      g++ \
      linux-headers \
      make \
      patch \
      subversion-dev \
      zlib-dev

RUN wget $MESOS_PKG_URL -O- | tar xz \
    && cd mesos-$MESOS_VERSION \
    && ./configure --disable-java --disable-python --enable-silent-rules \
    && make -j $(nproc) \
    && make install

FROM base as pip_build

ARG APK_EDGE_COMMUNITY_REPO=http://dl-cdn.alpinelinux.org/alpine/edge/community
ARG APK_EDGE_TESTING_REPO=http://dl-cdn.alpinelinux.org/alpine/edge/testing
ARG APK_REPOS="${APK_EDGE_TESTING_REPO}\n${APK_EDGE_COMMUNITY_REPO}"

RUN echo -e $APK_REPOS >> /etc/apk/repositories \
    && apk add --no-cache --virtual .build-deps \
      cython-dev \
      freetype-dev \
      g++ \
      hdf5-dev \
      libpng-dev \
      linux-headers \
      musl-dev \
      openblas-dev \
      py-numpy-dev \
    && apk add --no-cache \
      cython \
      gfortran \
      py-matplotlib \
      py2-cycler \
      py2-numpy \
      py2-scipy \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h

COPY requirements.txt /
#numpy scipy pandas matplotlib
ENV PYTHONPATH=/usr/lib/python2.7/site-packages
RUN pip install --no-build-isolation $(grep pandas requirements.txt) \
    && pip install -r /requirements.txt \
    && apk del .build-deps

#RUN pip install keras # tensorflow

FROM base
LABEL maintainer="Vinicius Dias <viniciusvdias@dcc.ufmg.br>, Guilherme Maluf <guimaluf@dcc.ufmg.br>, Walter Santos <walter@dcc.ufmg.br>"

ENV PYTHONPATH=${PYTHONPATH}:${JUICER_HOME}:${SPARK_HOME}/python \
    LC_ALL=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    JUICER_HOME=/usr/local/juicer

COPY --from=pip_build /etc/apk/repositories /etc/apk/repositories
COPY --from=pip_build /usr/lib /usr/lib
RUN apk --no-cache add \
      apr \
      curl \
      cyrus-sasl \
      cython \
      freetype \
      hdf5 \
      musl \
      openblas \
      openjdk8-jre \
      subversion \
      fts \
      tcl \
      tk \
      zlib

COPY --from=mesos_build /usr/local/* /usr/local/
COPY --from=pip_build /usr/local/* /usr/local/

ARG HADOOP_VERSION=2.7
ARG SPARK_VERSION=2.3.1
ARG SPARK_HADOOP_PKG=spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}
ARG SPARK_HADOOP_URL=https://www.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_HADOOP_PKG}.tgz
ARG SPARK_HOME=/usr/local/spark

RUN wget ${SPARK_HADOOP_URL} -O- | tar xz -C /usr/local/  \
  && ln -s /usr/local/$SPARK_HADOOP_PKG $SPARK_HOME

WORKDIR $JUICER_HOME
COPY . $JUICER_HOME

CMD ["/usr/local/juicer/sbin/juicer-daemon.sh", "docker"]
