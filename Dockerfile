FROM python:2.7.15-alpine3.7 as libmesos-build
label maintainer="Vinicius Dias <viniciusvdias@dcc.ufmg.br>, Guilherme Maluf <guimaluf@dcc.ufmg.br>"

ARG MESOS_VERSION=1.6.1
ENV SPARK_HOME /usr/local/spark
ENV JUICER_HOME /usr/local/juicer
ENV PYTHONPATH $PYTHONPATH:$JUICER_HOME:$SPARK_HOME/python

RUN echo 'http://dl-cdn.alpinelinux.org/alpine/v3.7/main' >> /etc/apk/repositories && \
    apk --no-cache --update-cache --virtual=lemonade-deps add \
        python \
        py-pip \
        openjdk8-jre && \
    apk --no-cache --update-cache --virtual=python-deps add \
        gcc \
        gfortran \
        python-dev \
        build-base \
        curl \
        freetype-dev \
        libpng-dev \
        openblas-dev && \
    apk add --no-cache --virtual=mesos-build-deps \
        zlib-dev \
        zlib \
        curl-dev \
        apr \
        apr-dev \
        subversion-libs \
        subversion-dev \
        libsasl \
        cyrus-sasl-dev \
        cyrus-sasl-crammd5 \
        fts-dev \
        patch \
        linux-headers 

RUN curl -s http://www.apache.org/dist/mesos/${MESOS_VERSION}/mesos-${MESOS_VERSION}.tar.gz | tar -xz -C /tmp/ 
RUN mkdir -p /tmp/mesos-${MESOS_VERSION}/build   \
     && cd /tmp/mesos-${MESOS_VERSION}/build  \
     && PYTHON_VERSION="2.7" ../configure --enable-optimized --disable-java --enable-silent-rules -disable-python \
     && make -j7 
RUN  cd /tmp/mesos-${MESOS_VERSION}/build &&  make install  

# Scipy, Numpy and Scikit
COPY requirements.txt /tmp
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h && \
    pip install -r /tmp/requirements.txt #numpy scipy pandas matplotlib

RUN apk add --no-cache --allow-untrusted \
    --repository http://dl-3.alpinelinux.org/alpine/edge/testing \
    hdf5 hdf5-dev

RUN pip install keras # tensorflow
# Cleanup
RUN rm -fr /var/cache/apk/* \
      && rm -fr /tmp/* \
      && apk del mesos-build-deps \
      && apk del python-deps


FROM python:2.7.15-alpine3.7

ARG MESOS_VERSION=1.6.1
ENV SPARK_HOME /usr/local/spark
ENV JUICER_HOME /usr/local/juicer
ENV PYTHONPATH $PYTHONPATH:$JUICER_HOME:$SPARK_HOME/python

RUN echo 'http://dl-cdn.alpinelinux.org/alpine/v3.7/main' >> /etc/apk/repositories && \
    apk --no-cache --update-cache --virtual=lemonade-deps add \
        python \
        py-pip \
        openjdk8-jre
COPY --from=libmesos-build /usr/local/lib/libmesos-${MESOS_VERSION}.so /usr/local/lib/ 
RUN ln -s /usr/local/lib/libmesos-${MESOS_VERSION}.so /usr/lib/libmesos.so
COPY --from=libmesos-build /usr/local/lib/python2.7/ /usr/local/lib/python2.7/
RUN apk --no-cache --update-cache add openblas curl

# RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
#   && locale-gen \
#   && update-locale LANG=en_US.UTF-8 \
#   && echo "LANG=en_US.UTF-8" >> /etc/default/locale \
#   && echo "LANGUAGE=en_US.UTF-8" >> /etc/default/locale \
#   && echo "LC_ALL=en_US.UTF-8" >> /etc/default/locale

ENV SPARK_VERSION=2.3.1
ENV HADOOP_VERSION=2.7
ENV SPARK_HADOOP_PKG spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}
ENV SPARK_HADOOP_URL http://www-eu.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_HADOOP_PKG}.tgz
RUN curl -s ${SPARK_HADOOP_URL} | tar -xz -C /usr/local/  \
  && mv /usr/local/$SPARK_HADOOP_PKG $SPARK_HOME

WORKDIR $JUICER_HOME
COPY requirements.txt $JUICER_HOME
COPY --from=libmesos-build /usr/local/bin/pybabel /usr/local/bin/

RUN pip install -r $JUICER_HOME/requirements.txt

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

COPY . $JUICER_HOME
RUN pybabel compile -d $JUICER_HOME/juicer/i18n/locales

CMD ["/usr/local/juicer/sbin/juicer-daemon.sh", "docker"]
