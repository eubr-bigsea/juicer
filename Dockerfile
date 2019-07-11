FROM ubuntu:16.04 as base

LABEL maintainer="Vinicius Dias <viniciusvdias@dcc.ufmg.br>, Guilherme Maluf <guimaluf@dcc.ufmg.br>, Walter Santos <walter@dcc.ufmg.br>"

ENV SPARK_HOME /usr/local/spark
ENV JUICER_HOME /usr/local/juicer
ENV PYTHONPATH $PYTHONPATH:$JUICER_HOME:$SPARK_HOME/python:${SPARK_HOME}/python/lib/pyspark.zip:${SPARK_HOME}/python/lib/py4j-*.zip
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV PATH="${PATH}:${JAVA_HOME}"

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv E56151BF \
  && echo deb http://repos.mesosphere.io/ubuntu trusty main > /etc/apt/sources.list.d/mesosphere.list \
  && apt-get update && apt-get install -y  \
      python-pip \
      python3-pip \
      python-tk \
      openjdk-8-jdk \
      curl \
      locales \
      mesos \
  && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
  && locale-gen \
  && update-locale LANG=en_US.UTF-8 \
  && echo "LANG=en_US.UTF-8" >> /etc/default/locale \
  && echo "LANGUAGE=en_US.UTF-8" >> /etc/default/locale \
  && echo "LC_ALL=en_US.UTF-8" >> /etc/default/locale \
  && rm -rf /var/lib/apt/lists/*

ENV SPARK_VERSION=2.4
ENV HADOOP_VERSION=2.7
ENV SPARK_BASE_URL=http://www.apache.org/dist/spark

# Get latest spark based on major.minor version
RUN SPARK_LATEST_VERSION=$(\
      curl -sL ${SPARK_BASE_URL} | \
      grep -Eo "spark-${SPARK_VERSION}\.[0-9]{1}" | \
      head -1 \
    ) \
  && SPARK_HADOOP_PKG=${SPARK_LATEST_VERSION}-bin-hadoop${HADOOP_VERSION} \
  && SPARK_HADOOP_URL=${SPARK_BASE_URL}/${SPARK_LATEST_VERSION}/${SPARK_HADOOP_PKG}.tgz \
  && curl -s ${SPARK_HADOOP_URL} | tar -xz -C /usr/local/  &&\
    mv /usr/local/$SPARK_HADOOP_PKG $SPARK_HOME &&\
    ln -s /usr/local/spark /opt/spark

WORKDIR $JUICER_HOME
COPY requirements.txt $JUICER_HOME

RUN pip install -r $JUICER_HOME/requirements.txt

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

COPY . $JUICER_HOME
RUN pybabel compile -d $JUICER_HOME/juicer/i18n/locales

COPY ./entrypoint.sh /opt/
CMD [ "/opt/entrypoint.sh" ]
