#!/usr/bin/env bash

# This script controls the juicer server daemon initialization, status reporting
# and termination
# TODO: rotate logs

usage="Usage: juicer-daemon.sh (start|startf|stop|status)"

# this sript requires the command parameter
if [ $# -le 0 ]; then
  echo $usage
  exit 1
fi

# parameter option
cmd_option=$1

# set juicer_home if unset
if [ -z "${JUICER_HOME}" ]; then
  export JUICER_HOME="$(cd "`dirname "$0"`"/..; pwd)"
fi
echo "JUICER_HOME=${JUICER_HOME}"

# get log directory
if [ "$JUICER_LOG_DIR" = "" ]; then
  export JUICER_LOG_DIR="${JUICER_HOME}/logs"
fi
mkdir -p "$JUICER_LOG_DIR"

# get pid directory
if [ "$JUICER_PID_DIR" = "" ]; then
  export JUICER_PID_DIR=/tmp
fi
mkdir -p "$JUICER_PID_DIR"

# log and pid files
log="$JUICER_LOG_DIR/juicer-server-$USER-$HOSTNAME.out"
pid="$JUICER_PID_DIR/juicer-server-$USER.pid"

case $cmd_option in

   (start)
      # set python path
      PYTHONPATH=$JUICER_HOME:$PYTHONPATH nohup -- python $JUICER_HOME/juicer/runner/juicer_server.py \
         -c $JUICER_HOME/conf/juicer-config.yaml >> $log 2>&1 < /dev/null &
      juicer_server_pid=$!

      # persist the pid
      echo $juicer_server_pid > $pid

      echo "Juicer server started, logging to $log (pid=$juicer_server_pid)"
      ;;

   (startf)
      trap "$0 stop" SIGINT SIGTERM
      # set python path
      PYTHONPATH=$JUICER_HOME:$PYTHONPATH python $JUICER_HOME/juicer/runner/juicer_server.py \
         -c $JUICER_HOME/conf/juicer-config.yaml &
      juicer_server_pid=$!

      # persist the pid
      echo $juicer_server_pid > $pid

      echo "Juicer server started, logging to $log (pid=$juicer_server_pid)"
      wait
      ;;

   (stop)

      if [ -f $pid ]; then
         TARGET_ID="$(cat "$pid")"
         if [[ $(ps -p "$TARGET_ID" -o comm=) =~ "python" ]]; then
            echo "stopping juicer server, user=$USER, hostname=$HOSTNAME"
            kill -SIGTERM "$TARGET_ID" && rm -f "$pid"
         else
            echo "no juicer server to stop"
         fi
      else
         echo "no juicer server to stop"
      fi
      ;;

   (status)

      if [ -f $pid ]; then
         TARGET_ID="$(cat "$pid")"
         if [[ $(ps -p "$TARGET_ID" -o comm=) =~ "python" ]]; then
            echo "juicer server is running (pid=$TARGET_ID)"
            exit 0
         else
            echo "$pid file is present (pid=$TARGET_ID) but juicer server not running"
            exit 1
         fi
      else
         echo juicer server not running.
         exit 2
      fi
      ;;

   (*)
      echo $usage
      exit 1
      ;;
esac
