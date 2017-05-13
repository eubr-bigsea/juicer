FROM ubuntu:16.04
MAINTAINER Vinicius Dias <viniciusvdias@dcc.ufmg.br>

ENV SPARK_HADOOP_PKG spark-2.1.0-bin-hadoop2.6
ENV SPARK_HADOOP_URL http://www-eu.apache.org/dist/spark/spark-2.1.0/${SPARK_HADOOP_PKG}.tgz
ENV SPARK_HOME /usr/local/spark
ENV JUICER_HOME /usr/local/juicer
ENV PYTHONPATH $PYTHONPATH:$JUICER_HOME:$SPARK_HOME/python

WORKDIR $JUICER_HOME
COPY requirements.txt $JUICER_HOME
RUN pip install -r $JUICER_HOME/requirements.txt

RUN apt-get update && apt-get install -y \
      python-pip \
      python-tk \
      openjdk-8-jdk \
      curl \
  && rm -rf /var/lib/apt/lists/* \
  && curl -s ${SPARK_HADOOP_URL} | tar -xz -C /usr/local/  \
  && mv /usr/local/$SPARK_HADOOP_PKG $SPARK_HOME

COPY . $JUICER_HOME

CMD ["/usr/local/juicer/sbin/juicer-daemon.sh", "startf"]
