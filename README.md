# Juicer
[logo]: docs/img/juicer.png "Lemonade Juicer"

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