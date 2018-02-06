FROM ubuntu:16.04
MAINTAINER Vinicius Dias <viniciusvdias@dcc.ufmg.br>

ARG SPARK_VERSION=2.2.1 HADOOP_VERSION=2.6
ENV SPARK_HOME=/usr/local/spark JUICER_HOME=/usr/local/juicer
ENV PYTHONPATH=$PYTHONPATH:$JUICER_HOME:$SPARK_HOME/python
ENV SPARK_HADOOP_URL http://ftp.unicamp.br/pub/apache/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

WORKDIR $JUICER_HOME
COPY requirements.txt $JUICER_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
      python-pip \
      python-setuptools \
      python-tk \
      openjdk-8-jre \
      build-essential \
      python-dev \
      curl \
  && curl -s ${SPARK_HADOOP_URL} | tar -xz -C /usr/local/ \
  && mv /usr/local/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION $SPARK_HOME \
  && pip install --no-cache-dir -r $JUICER_HOME/requirements.txt \
  && apt-get remove -y curl python-pip python-setuptools build-essential python-dev \
  && apt-get autoremove -y  \
  && rm -rf /var/lib/apt/lists/*

COPY . $JUICER_HOME
RUN pybabel compile -d $JUICER_HOME/juicer/i18n/locales

CMD ["/usr/local/juicer/sbin/juicer-daemon.sh", "startf"]
