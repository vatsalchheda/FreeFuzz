from paddle.distributed.fleet.utils import HDFSClient
hadoop_home = "/home/client/hadoop-client/hadoop/"

configs = {
    "fs.default.name": "hdfs://xxx.hadoop.com:54310",
    "hadoop.job.ugi": "hello,hello123"
}

client = HDFSClient(hadoop_home, configs)
client.ls_dir("hdfs:/test_hdfs_client")