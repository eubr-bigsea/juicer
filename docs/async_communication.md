# Async communication

Communication between some Lemonade components is made through message passing,
specially the integration between Stand and Juicer.

Message passing is implemented using structures (abstractions) stored in a Redis cluster.
Different structures are used: lists, queues, single key, sets and hashes.

Redis does not impose a format when storing messages. In the Lemonade platform,
all messages are JSON encoded. To hide the details of implementation, a Python
module called `juicer.runner.control.StateControlRedis` is used.
Such messages are described in table bellow:


| Id | Description| Producer | Consumer | Abstraction | Fields |
|:---:|:---:|:---:|:---:|:---:|:---:|
| queue_start | Starts a job for workflow | Stand | Juicer (server) | Queue | workflow\_id, job\_id, workflow |
| queue\_job\{job_id} | Runs the job using a minion | Juicer (server) | Juicer (minion) | Queue | workflow\_id, job\_id, workflow |
| record_workflow_\{workflow_id} | Controls workflow's execution status | Juicer (master) | Juicer (master) | Hash | workflow\_id(key), status (column) |
| key\_minion\_workflow_{workflow_id} | Controls if minion is alive by using a key with TTL set (30s) | Juicer (minion) | Juicer (server) | Key with TTL | workflow\_id, status, pid (process id)|
| queue\_output\_job\_{job_id} | Queue used to output job execution information to clients | Juicer (minion) | Stand | Queue | job\_id, workflow\_id, code, message|
| queue\_master | Queue used by the minion to communicate with master in case of severe failure | Juicer (minion) | Juicer (master) | Queue | job\_id, workflow\_id, reason|
| queue\_delivery\_job\_{job_id} | Output queue for requests for data | Juicer (minion) | Stand | Queue | Data from requested data source in CSV format |

