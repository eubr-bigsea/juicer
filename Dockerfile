FROM ubuntu:16.04
MAINTAINER Vinicius Dias <viniciusvdias@dcc.ufmg.br>

# Install python and jdk
RUN apt-get update \
   && apt-get install -qy python-pip \
   && apt-get install -qy openjdk-8-jdk

# Install spark
RUN apt-get install -qy curl
RUN curl -s http://www-eu.apache.org/dist/spark/spark-2.1.0/spark-2.1.0-bin-hadoop2.6.tgz \
   | tar -xz -C /usr/local/
RUN cd /usr/local && ln -s spark-2.1.0-bin-hadoop2.6 spark
ENV SPARK_HOME /usr/local/spark
ENV PYTHONPATH $PYTHONPATH:$SPARK_HOME/python

# Install juicer
ENV JUICER_HOME /usr/local/juicer
RUN mkdir -p $JUICER_HOME/conf
RUN mkdir -p $JUICER_HOME/sbin
RUN mkdir -p $JUICER_HOME/juicer
ADD sbin $JUICER_HOME/sbin
ADD juicer $JUICER_HOME/juicer

# Install juicer requirements and entrypoint
ADD requirements.txt $JUICER_HOME
RUN pip install -r $JUICER_HOME/requirements.txt
CMD ["/usr/local/juicer/sbin/juicer-daemon.sh", "startf"]
