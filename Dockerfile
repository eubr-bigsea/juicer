FROM python:2.7-alpine as base

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
      openssl-dev \
      libffi-dev  \
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

ENV LC_ALL=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    JUICER_HOME=/usr/local/juicer \
    SPARK_HOME=/usr/local/spark
ENV PYTHONPATH=${PYTHONPATH}:${JUICER_HOME}:${SPARK_HOME}/python

COPY --from=pip_build /etc/apk/repositories /etc/apk/repositories
COPY --from=pip_build /usr/lib /usr/lib
RUN apk --no-cache add \
      apr \
      bash \
      curl \
      cyrus-sasl \
      cython \
      freetype \
      fts \
      hdf5 \
      musl \
      openblas \
      openjdk8-jre \
      subversion \
      tcl \
      tk \
      zlib

COPY --from=eubrabigsea/mesos:alpine /usr/local/* /usr/local/
COPY --from=pip_build /usr/local/* /usr/local/

ARG HADOOP_VERSION=2.7
ARG SPARK_VERSION=2.3.1
ARG SPARK_HADOOP_PKG=spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}
ARG SPARK_HADOOP_URL=https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_HADOOP_PKG}.tgz

RUN wget ${SPARK_HADOOP_URL} -O- | tar xz -C /usr/local/  \
  && ln -s /usr/local/$SPARK_HADOOP_PKG $SPARK_HOME

WORKDIR $JUICER_HOME
COPY . $JUICER_HOME

CMD ["/usr/local/juicer/sbin/juicer-daemon.sh", "docker"]
