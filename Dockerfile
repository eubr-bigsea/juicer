FROM ubuntu:18.04 as base

LABEL maintainer="Vinicius Dias <viniciusvdias@dcc.ufmg.br>, Guilherme Maluf <guimaluf@dcc.ufmg.br>, Walter Santos <walter@dcc.ufmg.br>"

ENV SPARK_HOME /usr/local/spark
ENV JUICER_HOME /usr/local/juicer
ENV PYTHONPATH $PYTHONPATH:$JUICER_HOME:$SPARK_HOME/python:${SPARK_HOME}/python/lib/pyspark.zip:${SPARK_HOME}/python/lib/py4j-*.zip
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV PATH="${PATH}:${JAVA_HOME}"
ENV TERM=xterm\
    TZ=America/Sao_Paulo\
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \ 
      python3.7-dev \
      python3-pip \
      python3-dev \
      python3-tk \
      python3-setuptools \
      openjdk-8-jdk \
      curl \
      graphviz \
      locales \
      libffi-dev \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
  && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1\
  && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
  && locale-gen \
  && update-locale LANG=en_US.UTF-8 \
  && echo "LANG=en_US.UTF-8" >> /etc/default/locale \
  && echo "LANGUAGE=en_US.UTF-8" >> /etc/default/locale \
  && echo "LC_ALL=en_US.UTF-8" >> /etc/default/locale \
  && rm -rf /var/lib/apt/lists/*

ENV SPARK_VERSION=2.4
ENV HADOOP_VERSION=2.7
ENV SPARK_BASE_URL=https://archive.apache.org/dist/spark

# Get latest spark based on major.minor version
RUN SPARK_LATEST_VERSION=$(\
      curl -sL ${SPARK_BASE_URL} | \
      grep -Eo "spark-${SPARK_VERSION}\.[0-9]{1}" | \
      tail -1 \
    ) \
  && SPARK_HADOOP_PKG=${SPARK_LATEST_VERSION}-bin-hadoop${HADOOP_VERSION} \
  && SPARK_HADOOP_URL=${SPARK_BASE_URL}/${SPARK_LATEST_VERSION}/${SPARK_HADOOP_PKG}.tgz \
  && curl -sL ${SPARK_HADOOP_URL} | tar -xz -C /usr/local/  &&\
    mv /usr/local/$SPARK_HADOOP_PKG $SPARK_HOME &&\
    ln -s /usr/local/spark /opt/spark

WORKDIR $JUICER_HOME

ENV HADOOP_VERSION_BASE=2.7.7
ENV HADOOP_HOME /opt/hadoop-${HADOOP_VERSION_BASE}
ENV ARROW_LIBHDFS_DIR $HADOOP_HOME/native
ENV LD_LIBRARY_PATH=$HADOOP_HOME/native:$LD_LIBRARY_PATH

RUN curl -sL https://archive.apache.org/dist/hadoop/core/hadoop-${HADOOP_VERSION_BASE}/hadoop-${HADOOP_VERSION_BASE}.tar.gz | tar -xz -C /opt/

COPY requirements.txt $JUICER_HOME

RUN pip3 install -U pip wheel
RUN python -m pip install -r $JUICER_HOME/requirements.txt --no-cache-dir

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

COPY . $JUICER_HOME
RUN pybabel compile -d $JUICER_HOME/juicer/i18n/locales

COPY ./entrypoint.sh /opt/
RUN mkdir -p /usr/local/juicer/jars/ && ln -s $HADOOP_HOME /opt/hadoop 
RUN curl -Lo $JUICER_HOME/jars/spark-lof_2.11-1.0.jar https://github.com/dccspeed/spark-lof/raw/master/dist/spark-lof_2.11-1.0.jar
RUN curl -Lo $JUICER_HOME/jars/lemonade-spark-ext-1.0.jar https://github.com/eubr-bigsea/lemonade-spark-ext/raw/master/dist/lemonade-spark-ext-1.0.jar
ENV PATH=$PATH:$HADOOP_HOME/bin
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
RUN echo "export CLASSPATH=$(hadoop classpath --glob):/usr/local/juicer/jars/spark-lof_2.11-1.0.jar:/usr/local/juicer/jars/lemonade-spark-ext-1.0.jar" >> /etc/profile.d/juicer.sh && \
    echo "export SPARK_EXTRA_CLASSPATH=$CLASSPATH" >> /etc/profile.d/juicer.sh && \
    echo "export HADOOP_HOME=$HADOOP_HOME" >> /etc/profile.d/juicer.sh && \
    echo "export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop" >> /etc/profile.d/juicer.sh && \
    chmod a+x /etc/profile.d/juicer.sh
    
ENTRYPOINT ["/opt/entrypoint.sh"]
