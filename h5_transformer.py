import h5py
import numpy as np
import os
import shutil
from PIL import Image
import argparse
from tqdm import tqdm
import yaml
from glob import glob

folder_config_path = './folder_config.yml'
datasets_folder = './datasets/'
Image.MAX_IMAGE_PIXELS = None

def calc_overlap(database_region, query_region):
    valid_region = []
    valid_region.append(max(database_region[0], query_region[0])) # top
    valid_region.append(max(database_region[1], query_region[1])) # left
    valid_region.append(min(database_region[2], query_region[2])) # bottom
    valid_region.append(min(database_region[3], query_region[3])) # right
    
    # Check if the region is valid
    if valid_region[2]<=valid_region[0] or valid_region[3]<=valid_region[1]:
        raise ValueError('The area of valid region is less or equal to zero.')
        
    print("Get valid region: " + str(valid_region))
    return valid_region

def create_h5_file(args, name, split, sample_num):
    # Check input
    if not name in ['database', 'queries']:
        raise NotImplementedError('Name must be database or queries')
    if not split in ['train', 'val', 'test']:
        raise NotImplementedError('Split must be train or val or test')

    # Load yaml
    with open(folder_config_path, 'r') as f:
        folder_config = yaml.safe_load(f)

    # Check name
    if name == 'database':
        image = np.array(Image.open(os.path.join(
            datasets_folder, folder_config[args.database_name]['name'], folder_config[args.database_name]['maps'][args.database_index])).convert('RGB'))
        save_path = os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), f'{split}_database.h5')
    else:
        image = np.array(Image.open(os.path.join(
            datasets_folder, folder_config[args.queries_name]['name'], folder_config[args.queries_name]['maps'][args.queries_index])).convert('RGB'))
        save_path = os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), f'{split}_queries.h5') 
    if os.path.isfile(save_path):
        os.remove(save_path)

    # Check valid region
    database_region = folder_config[args.database_name]['valid_regions'][args.database_index]
    queries_region = folder_config[args.queries_name]['valid_regions'][args.queries_index]
    valid_region = calc_overlap(database_region, queries_region)

    # database region must be overlap with queries region
    if args.region_num == 2:
        # train at left half and val at right half
        if split == 'train':
            database_queries_region = [valid_region[0] + args.crop_width//2,
                                    valid_region[1] + args.crop_width//2,
                                    valid_region[2] - args.crop_width//2,
                                    (valid_region[1] + valid_region[3])//2 - args.crop_width//2]  # top, left, bottom, right
            print(f'Train region: {database_queries_region}')
        elif split == 'val':
            database_queries_region = [valid_region[0] + args.crop_width//2,
                                    (valid_region[1] + valid_region[3])//2 + args.crop_width//2,
                                    valid_region[2] - args.crop_width//2,
                                    valid_region[3] - args.crop_width//2]  # top, left, bottom, right
            print(f'Val region: {database_queries_region}')
        else:
            raise ValueError('Generate test option is false. Please add --generate_test to generate test set.')
    elif args.region_num == 3:
        if split == 'train': # bottom left
            database_queries_region = [(valid_region[0] + valid_region[2])//2 + args.crop_width//2,
                                        valid_region[1] + args.crop_width//2,
                                        valid_region[2] - args.crop_width//2,
                                        (valid_region[1] + valid_region[3])//2 - args.crop_width//2]  # top, left, bottom, right
            print(f'Train region: {database_queries_region}')
        elif split == 'val': # bottom right
            database_queries_region = [(valid_region[0] + valid_region[2])//2 + args.crop_width//2,
                                       (valid_region[1] + valid_region[3])//2 + args.crop_width//2,
                                        valid_region[2] - args.crop_width//2,
                                        valid_region[3] - args.crop_width//2]  # top, left, bottom, right
            print(f'Val region: {database_queries_region}')
        else: # top
            database_queries_region = [valid_region[0] + args.crop_width//2,
                                       valid_region[1] + args.crop_width//2,
                                      (valid_region[0] + valid_region[2])//2 - args.crop_width//2,
                                       valid_region[3] - args.crop_width//2]  # top, left, bottom, right
            print(f'Test region: {database_queries_region}')
    else:
        # train, val and test at the entire region
        database_queries_region = [valid_region[0] + args.crop_width//2,
                        valid_region[1] + args.crop_width//2,
                        valid_region[2] - args.crop_width//2,
                        valid_region[3] - args.crop_width//2]  # top, left, bottom, right
        if split == 'train':
            print(f'Train region: {database_queries_region}')
        elif split == 'val':
            print(f'Val region: {database_queries_region}')
        else:
            print(f'Test region: {database_queries_region}')
    # Write h5
    with h5py.File(save_path, "a") as hf:
        start = False
        img_names = []

        if args.sample_method == 'random':
            cood_y = np.random.randint(
                database_queries_region[0], database_queries_region[2], size=sample_num)
            cood_x = np.random.randint(
                database_queries_region[1], database_queries_region[3], size=sample_num)
        elif args.sample_method == 'grid':
            cood_y_only = np.linspace(
                database_queries_region[0], database_queries_region[2], size=round(np.sqrt(sample_num)))
            cood_x_only = np.linspace(
                database_queries_region[1], database_queries_region[3], size=round(np.sqrt(sample_num)))
            cood_x, cood_y = np.meshgrid(cood_x_only, cood_y_only)
            cood_y = cood_y.flatten()
            cood_x = cood_x.flatten()
        elif args.sample_method == 'stride':
            print("Warning: Stride sampling overrides sample num. You may get less or more samples.")
            cood_y_only = np.arange(
                database_queries_region[0], database_queries_region[2], step=args.stride)
            cood_x_only = np.arange(
                database_queries_region[1], database_queries_region[3], step=args.stride)
            cood_x, cood_y = np.meshgrid(cood_x_only, cood_y_only)
            cood_y = cood_y.flatten()
            cood_x = cood_x.flatten()
        else:
            raise NotImplementedError()

        for i in tqdm(range(len(cood_y))):
            name = f'@{cood_y[i]}@{cood_x[i]}'
            img_names.append(name)
            img_np = image[cood_y[i]-args.crop_width//2: cood_y[i]+args.crop_width //
                           2, cood_x[i]-args.crop_width//2: cood_x[i]+args.crop_width//2, :]
            img_np = np.expand_dims(img_np, axis=0)
            size_np = np.expand_dims(
                np.array([img_np.shape[1], img_np.shape[2]]), axis=0)
            if not start:
                if args.compress:
                    hf.create_dataset(
                        "image_data",
                        data=img_np,
                        chunks=(1, 512, 512, 3),
                        maxshape=(None, 512, 512, 3),
                        compression="lzf",
                    )  # write the data to hdf5 file
                    hf.create_dataset(
                        "image_size",
                        data=size_np,
                        chunks=True,
                        maxshape=(None, 2),
                        compression="lzf",
                    )
                else:
                    hf.create_dataset(
                        "image_data",
                        data=img_np,
                        chunks=(1, 512, 512, 3),
                        maxshape=(None, 512, 512, 3),
                    )  # write the data to hdf5 file
                    hf.create_dataset(
                        "image_size", data=size_np, chunks=True, maxshape=(None, 2)
                    )
                start = True
            else:
                hf["image_data"].resize(
                    hf["image_data"].shape[0] + img_np.shape[0], axis=0
                )
                hf["image_data"][-img_np.shape[0]:] = img_np
                hf["image_size"].resize(
                    hf["image_size"].shape[0] + size_np.shape[0], axis=0
                )
                hf["image_size"][-size_np.shape[0]:] = size_np
        t = h5py.string_dtype(encoding="utf-8")
        if args.compress:
            hf.create_dataset("image_name", data=img_names,
                              dtype=t, compression="lzf")
        else:
            hf.create_dataset("image_name", data=img_names, dtype=t)
        print("hdf5 file size: %d bytes" % os.path.getsize(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database_name",
        type=str,
        choices=['satellite', 'sirmionemapping',
                 'thermalmapping', 'foxtechmapping', 'ADASI', 'ADASI_thermal', 'thermalmappingDJI'],
        help="The name of database map you want to use"
    )
    parser.add_argument(
        "--database_index",
        type=int,
        help="The index of database flight you want to use. For satellite map, 0 is bing."
    )
    parser.add_argument(
        "--queries_name",
        type=str,
        choices=['satellite', 'sirmionemapping',
                 'thermalmapping', 'foxtechmapping', 'ADASI', 'ADASI_thermal', 'thermalmappingDJI'],
        help="The name of queries map you want to use"
    )
    parser.add_argument(
        "--queries_index",
        type=int,
        help="The index of queries flight you want to use. For satellite map, it is forced to be 0"
    )
    parser.add_argument("--crop_width", type=int, default=512)
    parser.add_argument("--train_sample_num", type=int, default=10000)
    parser.add_argument("--val_sample_num", type=int, default=10000)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--region_num", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--sample_method", type=str, default="random", choices=["random", "grid", "stride"])
    parser.add_argument("--stride", type=int, default=35)
    args = parser.parse_args()

    if os.path.isdir(os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index))):
        rmpaths = glob(os.path.join(datasets_folder, args.database_name + '_' + str(
            args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), '*'))
        for rmpath in rmpaths:
            os.remove(rmpath)
    else:
        os.mkdir(os.path.join(datasets_folder, args.database_name + '_' +
                str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index)))

    np.random.seed(0)
    if args.region_num >= 1:
        create_h5_file(args, name='database', split='train', sample_num=args.train_sample_num)
        create_h5_file(args, name='queries', split='train', sample_num=args.train_sample_num)

    if args.region_num >= 2:
        create_h5_file(args, name='database', split='val', sample_num=args.val_sample_num)
        create_h5_file(args, name='queries', split='val', sample_num=args.val_sample_num)

    if args.region_num == 2:
        # Not enough test data. Use val as test
        os.symlink(os.path.abspath(os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'val_database.h5')),
                os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'test_database.h5'))
        os.symlink(os.path.abspath(os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'val_queries.h5')),
                os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'test_queries.h5'))
    elif args.region_num == 3:
        create_h5_file(args, name='database', split='test', sample_num=args.val_sample_num)
        create_h5_file(args, name='queries', split='test', sample_num=args.val_sample_num)