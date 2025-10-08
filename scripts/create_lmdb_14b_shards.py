"""
python create_lmdb_14b_shards.py \
--data_path /mnt/localssd/wanx_14b_data \
--lmdb_path /mnt/localssd/wanx_14B_shift-3.0_cfg-5.0_lmdb
"""
from tqdm import tqdm
import numpy as np
import argparse
import torch
import lmdb
import glob
import os

from utils.lmdb import store_arrays_to_lmdb, process_data_dict


def main():
    """
    Aggregate all ode pairs inside a folder into a lmdb dataset.
    Each pt file should contain a (key, value) pair representing a
    video's ODE trajectories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        required=True, help="path to ode pairs")
    parser.add_argument("--lmdb_path", type=str,
                        required=True, help="path to lmdb")
    parser.add_argument("--num_shards", type=int,
                        default=16, help="num_shards")

    args = parser.parse_args()

    all_dirs = sorted(os.listdir(args.data_path))

    # figure out the maximum map size needed
    map_size = int(1e12)  # adapt to your need, set to 1TB by default
    os.makedirs(args.lmdb_path, exist_ok=True)
    # 1) Open one LMDB env per shard
    envs = []
    num_shards = args.num_shards
    for shard_id in range(num_shards):
        print("shard_id ", shard_id)
        path = os.path.join(args.lmdb_path, f"shard_{shard_id}")
        env = lmdb.open(path,
                        map_size=map_size,
                        subdir=True,       # set to True if you want a directory per env
                        readonly=False,
                        metasync=True,
                        sync=True,
                        lock=True,
                        readahead=False,
                        meminit=False)
        envs.append(env)

    counters = [0] * num_shards
    seen_prompts = set()  # for deduplication
    total_samples = 0
    all_files = []

    for part_dir in all_dirs:
        all_files += sorted(glob.glob(os.path.join(args.data_path, part_dir, "*.pt")))

    # 2) Prepare a write transaction for each shard
    for idx, file in tqdm(enumerate(all_files)):
        try:
            data_dict = torch.load(file)
            data_dict = process_data_dict(data_dict, seen_prompts)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

        if data_dict["latents"].shape != (1, 21, 16, 60, 104):
            continue

        shard_id = idx % num_shards
        # write to lmdb file
        store_arrays_to_lmdb(envs[shard_id], data_dict, start_index=counters[shard_id])
        counters[shard_id] += len(data_dict['prompts'])
        data_shape = data_dict["latents"].shape

    total_samples += len(all_files)

    print(len(seen_prompts))

    # save each entry's shape to lmdb
    for shard_id, env in enumerate(envs):
        with env.begin(write=True) as txn:
            for key, val in (data_dict.items()):
                assert len(data_shape) == 5
                array_shape = np.array(data_shape)  # val.shape)
                array_shape[0] = counters[shard_id]
                shape_key = f"{key}_shape".encode()
                print(shape_key, array_shape)
                shape_str = " ".join(map(str, array_shape))
                txn.put(shape_key, shape_str.encode())

    print(f"Finished writing {total_samples} examples into {num_shards} shards under {args.lmdb_path}")


if __name__ == "__main__":
    main()
