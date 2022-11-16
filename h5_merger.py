import h5py
import numpy as np
import os
import shutil
from PIL import Image
import argparse
from tqdm import tqdm
import yaml

folder_config_path = './folder_config.yml'
datasets_folder = './datasets/'

def merge_h5_file(args, name, split, indexes):
    # Check input
    if not name in ['database', 'queries']:
        raise NotImplementedError('Name must be database or queries')
    if not split in ['train', 'val', 'test']:
        raise NotImplementedError('Split must be train or val or test')

    # Check h5 names
    if name == 'database':
        read_path = []
        indexes_list = [*indexes]
        for i in indexes_list:
            if args.database_name == 'satellite':
                read_path.append(os.path.join(
                    datasets_folder, f'{args.database_name}_{i}_{args.queries_name}_{indexes[0]}/{split}_database.h5'))
            elif args.database_name == 'thermalmapping':
                read_path.append(os.path.join(
                    datasets_folder, f'{args.database_name}_{i}_{args.queries_name}_{i}/{split}_database.h5'))
            else:
                raise NotImplementedError()
        save_path = os.path.join(
            datasets_folder, f'{args.database_name}_{indexes}_{split}_database.h5')
    else:
        read_path = []
        indexes_list = [*indexes]
        for i in indexes_list:
            if args.database_name == 'satellite':
                read_path.append(os.path.join(
                    datasets_folder, f'{args.database_name}_{indexes[0]}_{args.queries_name}_{i}/{split}_queries.h5'))
            elif args.database_name == 'thermalmapping':
                read_path.append(os.path.join(
                    datasets_folder, f'{args.database_name}_{i}_{args.queries_name}_{i}/{split}_queries.h5'))
            else:
                raise NotImplementedError()
        save_path = os.path.join(
            datasets_folder, f'{args.queries_name}_{indexes}_{split}_queries.h5')

    if os.path.isfile(save_path):
        os.remove(save_path)

    # Write h5
    with h5py.File(save_path, "a") as hf:
        start = False
        for read_path_single in tqdm(read_path):
            with h5py.File(read_path_single, "r") as hf_single:
                t = h5py.string_dtype(encoding="utf-8")
                if not start:
                    if args.compress:
                        hf.create_dataset(
                            "image_data",
                            data=hf_single["image_data"],
                            chunks=True,
                            maxshape=(None, None, None, 3,),
                            compression="lzf",
                        )
                        hf.create_dataset(
                            "image_size",
                            data=hf_single["image_size"],
                            chunks=True,
                            maxshape=(None, 2,),
                            compression="lzf",
                        )
                        hf.create_dataset(
                            "image_name",
                            data=hf_single["image_name"],
                            chunks=True,
                            maxshape=(None,),
                            compression="lzf",
                            dtype=t
                        )
                    else:
                        hf.create_dataset(
                            "image_data",
                            data=hf_single["image_data"],
                            chunks=True,
                            maxshape=(None, None, None, 3,),
                        )
                        hf.create_dataset(
                            "image_size", 
                            data=hf_single["image_name"], 
                            chunks=True, 
                            maxshape=(None, 2)
                        )
                        hf.create_dataset(
                            "image_name",
                            data=hf_single["image_size"],
                            chunks=True,
                            maxshape=(None,),
                            dtype=t
                        )
                    start = True
                else:
                    hf["image_data"].resize(
                        hf["image_data"].shape[0] + hf_single["image_data"].shape[0], axis=0
                    )
                    hf["image_data"][-hf_single["image_data"].shape[0]:] = hf_single["image_data"]
                    hf["image_size"].resize(
                        hf["image_size"].shape[0] + hf_single["image_size"].shape[0], axis=0
                    )
                    hf["image_size"][-hf_single["image_size"].shape[0]:] = hf_single["image_size"]
                    hf["image_name"].resize(
                        hf["image_name"].shape[0] + hf_single["image_name"].shape[0], axis=0
                    )
                    hf["image_name"][-hf_single["image_name"].shape[0]:] = hf_single["image_name"]
                # for i in tqdm(range(len(hf_single["image_data"]))):
                #     if not start:
                #         if args.compress:
                #             hf.create_dataset(
                #                 "image_data",
                #                 data=np.expand_dims(hf_single["image_data"][i], axis=0),
                #                 chunks=True,
                #                 maxshape=(None, None, None, 3,),
                #                 compression="lzf",
                #             )
                #             hf.create_dataset(
                #                 "image_size",
                #                 data=np.expand_dims(hf_single["image_size"][i], axis=0),
                #                 chunks=True,
                #                 maxshape=(None, 2,),
                #                 compression="lzf",
                #             )
                #             hf.create_dataset(
                #                 "image_name",
                #                 data=np.expand_dims(hf_single["image_name"][i], axis=0),
                #                 chunks=True,
                #                 maxshape=(None,),
                #                 compression="lzf",
                #                 dtype=t
                #             )
                #         else:
                #             hf.create_dataset(
                #                 "image_data",
                #                 data=np.expand_dims(hf_single["image_data"][i], axis=0),
                #                 chunks=True,
                #                 maxshape=(None, None, None, 3,),
                #             )
                #             hf.create_dataset(
                #                 "image_size", 
                #                 data=np.expand_dims(hf_single["image_name"][i], axis=0), 
                #                 chunks=True, 
                #                 maxshape=(None, 2)
                #             )
                #             hf.create_dataset(
                #                 "image_name",
                #                 data=np.expand_dims(hf_single["image_size"][i], axis=0),
                #                 chunks=True,
                #                 maxshape=(None,),
                #                 dtype=t
                #             )
                #         start = True
                #     else:
                #         hf["image_data"].resize(
                #             hf["image_data"].shape[0] + np.expand_dims(hf_single["image_data"][i], axis=0).shape[0], axis=0
                #         )
                #         hf["image_data"][-hf_single["image_data"].shape[0]:] = hf_single["image_data"][i]
                #         hf["image_size"].resize(
                #             hf["image_size"].shape[0] + np.expand_dims(hf_single["image_size"][i], axis=0).shape[0], axis=0
                #         )
                #         hf["image_size"][-hf_single["image_size"].shape[0]:] = hf_single["image_size"][i]
                #         hf["image_name"].resize(
                #             hf["image_name"].shape[0] + np.expand_dims(hf_single["image_name"][i], axis=0).shape[0], axis=0
                #         )
                #         hf["image_name"][-hf_single["image_name"].shape[0]:] = hf_single["image_name"][i]

        print("hdf5 file size: %d bytes" % os.path.getsize(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database_name",
        type=str,
        choices=['satellite', 'sirmionemapping',
                 'thermalmapping', 'foxtechmapping'],
        help="The name of database map you want to use"
    )
    parser.add_argument(
        "--database_indexes",
        type=str,
        default="0",
        help="The index of database flight you want to use. For satellite map, 0 is bing."
    )
    parser.add_argument(
        "--queries_name",
        type=str,
        choices=['satellite', 'sirmionemapping',
                 'thermalmapping', 'foxtechmapping'],
        help="The name of queries map you want to use"
    )
    parser.add_argument(
        "--queries_indexes",
        type=str,
        default="012345",
        help="The index of queries flight you want to use. For satellite map, it is forced to be 0"
    )
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--region_num", type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()

    if os.path.isdir(os.path.join(datasets_folder, args.database_name + '_' + args.database_indexes + '_' + args.queries_name + '_' + args.queries_indexes)):
        shutil.rmtree(os.path.join(datasets_folder, args.database_name + '_' + args.database_indexes + '_' + args.queries_name + '_' + args.queries_indexes))
    os.mkdir(os.path.join(datasets_folder, args.database_name + '_' + args.database_indexes + '_' + args.queries_name + '_' + args.queries_indexes))

    if args.region_num >= 1:
        merge_h5_file(args, name='database', split='train', indexes=args.database_indexes)
        merge_h5_file(args, name='queries', split='train', indexes=args.queries_indexes)
        shutil.move(os.path.join(datasets_folder, f'{args.database_name}_{args.database_indexes}_train_database.h5'),
                    os.path.join(datasets_folder, args.database_name + '_' + str(args.database_indexes) + '_' + args.queries_name + '_' + str(args.queries_indexes), 'train_database.h5'))
        shutil.move(os.path.join(datasets_folder, f'{args.queries_name}_{args.queries_indexes}_train_queries.h5'),
                    os.path.join(datasets_folder, args.database_name + '_' + str(args.database_indexes) + '_' + args.queries_name + '_' + str(args.queries_indexes), 'train_queries.h5'))

    if args.region_num >= 2:
        merge_h5_file(args, name='database', split='val', indexes=args.database_indexes)
        merge_h5_file(args, name='queries', split='val', indexes=args.queries_indexes)
        shutil.move(os.path.join(datasets_folder, f'{args.database_name}_{args.database_indexes}_val_database.h5'),
                    os.path.join(datasets_folder, args.database_name + '_' + str(args.database_indexes) + '_' + args.queries_name + '_' + str(args.queries_indexes), 'val_database.h5'))
        shutil.move(os.path.join(datasets_folder, f'{args.queries_name}_{args.queries_indexes}_val_queries.h5'),
                    os.path.join(datasets_folder, args.database_name + '_' + str(args.database_indexes) + '_' + args.queries_name + '_' + str(args.queries_indexes), 'val_queries.h5'))

    if args.region_num <= 2:
        # Not enough test data. Use val as test
        os.symlink(os.path.abspath(os.path.join(datasets_folder, args.database_name + '_' + str(args.database_indexes) + '_' + args.queries_name + '_' + str(args.queries_indexes), 'val_database.h5')),
                os.path.join(datasets_folder, args.database_name + '_' + str(args.database_indexes) + '_' + args.queries_name + '_' + str(args.queries_indexes), 'test_database.h5'))
        os.symlink(os.path.abspath(os.path.join(datasets_folder, args.database_name + '_' + str(args.database_indexes) + '_' + args.queries_name + '_' + str(args.queries_indexes), 'val_queries.h5')),
                os.path.join(datasets_folder, args.database_name + '_' + str(args.database_indexes) + '_' + args.queries_name + '_' + str(args.queries_indexes), 'test_queries.h5'))
    else:
        merge_h5_file(args, name='database', split='test', indexes=args.database_indexes)
        merge_h5_file(args, name='queries', split='test', indexes=args.queries_indexes)
        shutil.move(os.path.join(datasets_folder, f'{args.database_name}_{args.database_indexes}_test_database.h5'),
                    os.path.join(datasets_folder, args.database_name + '_' + str(args.database_indexes) + '_' + args.queries_name + '_' + str(args.queries_indexes), 'test_database.h5'))
        shutil.move(os.path.join(datasets_folder, f'{args.queries_name}_{args.queries_indexes}_test_queries.h5'),
                    os.path.join(datasets_folder, args.database_name + '_' + str(args.database_indexes) + '_' + args.queries_name + '_' + str(args.queries_indexes), 'test_queries.h5'))