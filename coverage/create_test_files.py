import os
import shutil

Original_dir = "E:/UIUC/Spring 2023/CS 527/FreeFuzz/"
types = ['crash', 'cuda', 'precision']
Original_dir += "src/paddle-output/type-oracle/success/"
test_dir = "./test/"

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

for t in types:
    FreeFuzz_dir = Original_dir.replace('type', t)
    folders = os.listdir(FreeFuzz_dir)

    for folder in folders:
        folder_dir = FreeFuzz_dir + folder + "/"
        for file in os.listdir(folder_dir):
            if file.endswith('.py'):
                file_dir = folder_dir + file
                contents = open(file_dir, "r+")
                lines = contents.readlines()
                # lines.insert(1, "paddle.utils.run_check()\n") if lines[0].__contains__('import paddle') else lines.insert(2, "paddle.utils.run_check()\n")
                code = '\t'.join(lines)
                new_file_name = "test_" + t + "_" + folder.replace(".","_") + "_" + file
                test_line = f"def {new_file_name[:-3]}():\n"
                test_file = open(test_dir+new_file_name, "w")
                test_file.write(test_line + '\t' + code + "\tassert True")
                test_file.close()
                contents.close()