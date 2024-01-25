# 整合lmdb文件
import lmdb
import os
from tqdm import tqdm

## gtdb
# 输入的 LMDB 文件列表
lmdb_dir = "/data4/yaoshuai/gtdb_reps_r214/protein_faa_reps/lmdb/"
input_lmdb_files = list(map(lambda x: lmdb_dir+x, os.listdir(lmdb_dir)))
# 输出的总 LMDB 文件路径
output_lmdb_file = '/data5/zhanghaohong/projects/BGC/data/gtdb_all_lmdb/'

# 打开输出的总 LMDB 文件
env_out = lmdb.open(output_lmdb_file, subdir=True, map_size=1099511627776*2, readonly=False, meminit=False, map_async=True)  # 设置适当的 map_size 2 TB

# 开始合并
with env_out.begin(write=True) as txn_out:
    for lmdb_file in tqdm(input_lmdb_files):
        # 打开输入的 LMDB 文件
        env_in = lmdb.open(lmdb_file, readonly=True)
        with env_in.begin() as txn_in:
            # 迭代输入的 LMDB 文件中的键值对，并将其复制到输出的总 LMDB 文件中
            cursor = txn_in.cursor()
            for key, value in cursor:
                txn_out.put(key, value)
        env_in.close()

env_out.close()

# ## metagenome
# # 输入的 LMDB 文件列表
# lmdb_dir = "/data4/yaoshuai/metadata_set/lmdb/"
# input_lmdb_files = list(map(lambda x: lmdb_dir+x, os.listdir(lmdb_dir)))
# # 输出的总 LMDB 文件路径
# output_lmdb_file = '/data4/yaoshuai/metadata_set/lmdb_all/'

# # 打开输出的总 LMDB 文件
# env_out = lmdb.open(output_lmdb_file, subdir=True, map_size=1099511627776, readonly=False, meminit=False, map_async=True)  # 设置适当的 map_size 2 TB

# # 开始合并
# with env_out.begin(write=True) as txn_out:
#     for lmdb_file in input_lmdb_files:
#         # 打开输入的 LMDB 文件
#         env_in = lmdb.open(lmdb_file, readonly=True)
#         with env_in.begin() as txn_in:
#             # 迭代输入的 LMDB 文件中的键值对，并将其复制到输出的总 LMDB 文件中
#             cursor = txn_in.cursor()
#             for key, value in cursor:
#                 txn_out.put(key, value)
#         env_in.close()

# env_out.close()