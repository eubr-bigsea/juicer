# Juicer
[logo]: docs/img/juicer.png "Lemonade Juicer"

[![Build Status](https://travis-ci.org/eubr-bigsea/juicer.svg?branch=master)](https://travis-ci.org/eubr-bigsea/juicer)

![alt text][logo]

Juicer is the workflow processor manager for the Lemonade. Its responsibilities include:

1. Receive a workflow specification in JSON format from Citron and convert it into executable code. Current version *transpile* code only to Python language (interpreted),
runnable in Apache Spark.
2. Execute the generated code, controlling the execution flow.
3. Report execution status to the user interface (Citron).
4. Interact with Limonero API in order to create new intermediate data sets.
Such data sets can not be used as input to other workflows, except if explicitly specified.
They are used to enable Citron to show intermediate processed data to the user.

Under the hood, Lemonade generates code targeting a distributed processing platform,
such as COMPSs or Spark. Current version supports only Spark and it is executed in batch mode.
Future versions may implement support to interactive execution.
This kind of execution has advantages because keeping Spark context loaded cuts
out any overhead when starting the processing environment and data loading.
This approach (keeping the context) is used in many implementations of data
analytics notebooks, such as Jupyter, Cloudera Hue and Databricks notebook.

## Other optional dependencies

MLlib algorithms can use OpenBlas/Atlas optimizations. So, you need to install the packages
 libgfortran3, libatlas3-base and libopenblas-base. If you are using Debian/Ubuntu, you can
 install these libraries running the following command:

 ```
    sudo apt-get install libatlas3-base libopenblas-base libqhull-dev libpng-dev \ 
           libfreetype6-dev libgfortran10-dev libatlas-base-dev libffi-dev
```

If the application is reporting this message: `WARN util.NativeCodeLoader: Unable to load
native-hadoop library for your platform... using builtin-java classes where applicable`, you
need to add `$HADOOP_HOME/lib/native` to `LD_LIBRARY_PATH`:

```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native```
## Configuration
All configuration is defined in a Yaml file located in `conf/juicer-config.yaml`,
with the following structure:

```
juicer:
    debug: true
    servers:
        database_url: mysql+pymysql://user:password@server:port/database
        redis_url: redis://redis_server:6379
    services:
        tahiti:
            url: http://server/tahiti
            auth_token: "authorization_token"
    config:
        tmp_dir: /tmp
```

You will find the template above in `conf/juicer-config.yaml.template`.

## Running

```
cd <download_dir>
./sbin/juicer-daemon.sh start
```

You can check the stand daemon status with:
```
./sbin/juicer-daemon.sh status
```

You can stop the stand daemon with:
```
./sbin/juicer-daemon.sh stop
```

## Internationalization
Juicer uses GNU gettext format for internationalization (i18n). In order to
generate translation files (`*.mo` files), you must run the following commands
in the project's base dir:

```
pybabel extract -F babel.cfg -o juicer/i18n/juicer.pot .
pybabel compile -d juicer/i18n/locales
```
If you need to translate Juicer to another language, consider adding the new
message file using the following command (`es_ES` is the locale code):

```
pybabel init -i juicer/i18n/juicer.pot -d juicer/i18n/locales -l es
```

New messages can be extracted and written to the template file using this command:
```
pybabel extract -F babel.cfg -o juicer/i18n/juicer.pot .
```
