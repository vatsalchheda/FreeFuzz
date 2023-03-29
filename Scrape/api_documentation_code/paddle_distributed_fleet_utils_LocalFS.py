from paddle.distributed.fleet.utils import LocalFS

client = LocalFS()
subdirs, files = client.ls_dir("./")