# coding=utf-8
# noinspection PyPep8Naming,PyMethodMayBeStatic



# noinspection PyPep8Naming
class SparkListener(object):
    def __init__(self, emit):
        self.emit = emit
        self.stages = {}

    def onApplicationEnd(self, applicationEnd):
        pass

    def onApplicationStart(self, applicationStart):
        pass

    def onBlockManagerRemoved(self, blockManagerRemoved):
        pass

    def onBlockUpdated(self, blockUpdated):
        pass

    def onEnvironmentUpdate(self, environmentUpdate):
        pass

    def onExecutorAdded(self, executorAdded):
        pass

    def onExecutorMetricsUpdate(self, executorMetricsUpdate):
        pass

    def onExecutorRemoved(self, executorRemoved):
        pass

    def onJobEnd(self, jobEnd):
        pass

    # noinspection PyMethodMayBeStatic
    def onJobStart(self, jobStart):
        try:
            l = jobStart.stageIds()
            l2 = []
            for i in range(l.size()):
                l2.append(l.remove(0))
            print('#' * 20, l2)
        except Exception as e:
            print('@' * 20, str(e), dir(jobStart))

    def onOtherEvent(self, event):
        pass

    def onStageCompleted(self, stageCompleted):
        self.stages[stageCompleted.stageInfo().stageId()]['finished'] = True

    def onStageSubmitted(self, stageSubmitted):
        self.stages[stageSubmitted.stageInfo().stageId()] = {
            'numTasks': stageSubmitted.stageInfo().numTasks(),
            'completedTask': 0,
            'startedTasks': 0,
            'finished': False
        }

    #
    def onTaskEnd(self, taskEnd):
        stage_info = self.stages[taskEnd.stageId()]
        stage_info['completedTask'] += 1
        self.emit('Completed task {}/{}.'.format(stage_info['completedTask'],
                                                 stage_info['numTasks']))

    def onTaskStart(self, taskStart):
        stage_info = self.stages[taskStart.stageId()]
        stage_info['startedTasks'] += 1
        self.emit('Starting task {}/{}.'.format(stage_info['startedTasks'],
                                                stage_info['numTasks']))

    def onTaskGettingResult(self, taskGettingResult):
        pass

    def onUnpersistRDD(self, unpersistRDD):
        pass

    # noinspection PyClassHasNoInit
    class Java:
        implements = ["org.apache.spark.scheduler.SparkListenerInterface"]
