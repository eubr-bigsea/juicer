FROM openjdk:8-slim AS common
ENV HADOOP_VERSION_BASE=2.7.7
ENV HADOOP_HOME=/opt/hadoop-${HADOOP_VERSION_BASE}
ENV LD_LIBRARY_PATH=$HADOOP_HOME/native
ENV JAVA_HOME=/usr/local/openjdk-8
ENV JUICER_HOME=/usr/local/juicer
ENV PATH="${JUICER_HOME}/.venv/bin/:${PATH}:${JAVA_HOME}:${HADOOP_HOME}/bin"
ENV \
    ARROW_LIBHDFS_DIR=$HADOOP_HOME/native \
    DEBIAN_FRONTEND=noninteractive \
    HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop \
    HADOOP_VERSION=2.7 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHON_VERSION=3.7.9 \
    SPARK_BASE_URL=https://archive.apache.org/dist/spark \
    SPARK_HOME=/usr/local/spark \
    SPARK_HOME=/usr/local/spark \
    SPARK_VERSION=2.4 \
    TERM=xterm \
    TZ=America/Sao_Paulo
WORKDIR $JUICER_HOME

RUN \
    apt-get update && apt-get install -y \
      libpython3-dev \
      python3-pip \
      python3-dev \
      python3-tk \
      curl \
      graphviz \
      locales \
      procps \
      libffi-dev
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
  && locale-gen \
  && update-locale LANG=en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*


# ============== Builder
FROM common AS builder
COPY pyproject.toml $JUICER_HOME
RUN pip install -U pip wheel uv && \
    uv python install $PYTHON_VERSION && \
    uv python pin $PYTHON_VERSION && \
    uv sync

# =========== Downloader
FROM common AS downloader
# Download SPARK
RUN SPARK_LATEST_VERSION=$(curl -sL ${SPARK_BASE_URL} | \
    grep -Eo "spark-${SPARK_VERSION}.[0-9]{1}" | tail -1) \
    && SPARK_HADOOP_PKG=${SPARK_LATEST_VERSION}-bin-hadoop${HADOOP_VERSION} \
    && SPARK_HADOOP_URL=${SPARK_BASE_URL}/${SPARK_LATEST_VERSION}/${SPARK_HADOOP_PKG}.tgz \
    && curl -sL ${SPARK_HADOOP_URL} | tar -xz -C /usr/local/ \
    && mv /usr/local/$SPARK_HADOOP_PKG $SPARK_HOME

# Download Hadoop
RUN \
    curl -sL https://archive.apache.org/dist/hadoop/core/hadoop-${HADOOP_VERSION_BASE}/hadoop-${HADOOP_VERSION_BASE}.tar.gz | tar -xz -C /opt/ && \
    mkdir -p /usr/local/juicer/jars/ && ln -s $HADOOP_HOME /opt/hadoop
# Download libs
RUN \
    curl -Lo $JUICER_HOME/jars/spark-lof_2.11-1.0.jar https://github.com/dccspeed/spark-lof/raw/master/dist/spark-lof_2.11-1.0.jar && \
    curl -Lo $JUICER_HOME/jars/lemonade-spark-ext-1.0.jar https://github.com/eubr-bigsea/lemonade-spark-ext/raw/master/dist/lemonade-spark-ext-1.0.jar

# ================ Runtime stage
FROM common AS runtime
COPY --from=downloader /opt /opt
COPY --chmod=755 ./entrypoint.sh /opt/
COPY --from=builder /usr/local/juicer /usr/local/juicer
COPY --from=builder /root/.local /root/.local
COPY . $JUICER_HOME
ENV PYTHONPATH=$JUICER_HOME:.
RUN \
    chmod +x /opt/entrypoint.sh \
    && echo "export SPARK_DIST_CLASSPATH=$(hadoop classpath --glob):/usr/local/juicer/jars/spark-lof_2.11-1.0.jar:/usr/local/juicer/jars/lemonade-spark-ext-1.0.jar" >> /etc/profile.d/juicer.sh \
    && chmod a+x /etc/profile.d/juicer.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
