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
Image.MAX_IMAGE_PIXELS = None
valid_region_uav = [750, 550, 4600, 6600]  # top left bottom right foxtech
valid_region_satellite = [0, 0, 9088, 23744]  # top left bottom right satellite
queries_database_sampling = 'random'


def create_h5_file(args, name=None, split=None):
    # Check input
    if not name in ['database', 'queries']:
        raise NotImplementedError('Name must be database or queries')
    if not split in ['train', 'val']:
        raise NotImplementedError('Split must be train or val')

    # Load yaml
    with open(folder_config_path, 'r') as f:
        folder_config = yaml.safe_load(f)

    # Check name
    if name == 'database':
        image = np.array(Image.open(os.path.join(
            datasets_folder, folder_config[args.database_name]['name'], folder_config[args.database_name]['maps'][args.database_index])).convert('RGB'))
        save_path = os.path.join(
            datasets_folder, f'{args.database_name}_{args.database_index}_{split}_database.h5')
    else:
        image = np.array(Image.open(os.path.join(
            datasets_folder, folder_config[args.queries_name]['name'], folder_config[args.queries_name]['maps'][args.queries_index])).convert('RGB'))
        save_path = os.path.join(
            datasets_folder, f'{args.queries_name}_{args.queries_index}_{split}_queries.h5')
    if os.path.isfile(save_path):
        os.remove(save_path)

    # Check valid region
    if 'foxtech' in args.database_name or 'foxtech' in args.queries_name:
        valid_region = valid_region_uav
    else:
        valid_region = valid_region_satellite
    # database region must be overlap with queries region
    # train at left half and val at right half
    if split == 'train':
        database_queries_region = [valid_region[0] + args.crop_width,
                                   valid_region[1] + args.crop_width,
                                   (valid_region[0] + valid_region[2]
                                    )//2 - args.crop_width,
                                   (valid_region[1] + valid_region[3])//2 - args.crop_width]  # top, left, bottom, right
    else:
        database_queries_region = [(valid_region[0] + valid_region[2])//2 + args.crop_width,
                                   (valid_region[1] + valid_region[3]
                                    )//2 + args.crop_width,
                                   valid_region[2] - args.crop_width,
                                   valid_region[3] - args.crop_width]  # top, left, bottom, right
    # Write h5
    with h5py.File(save_path, "a") as hf:
        start = False
        img_names = []
        if queries_database_sampling == 'random':
            cood_y = np.random.randint(
                database_queries_region[0], database_queries_region[2], size=args.sample_num)
            cood_x = np.random.randint(
                database_queries_region[1], database_queries_region[3], size=args.sample_num)
        for i in tqdm(range(len(cood_y))):
            name = f'@{cood_y[i]}@{cood_x[i]}'
            img_names.append(name)

            img_np = image[cood_y[i]-args.crop_width//2: cood_y[i]+args.crop_width //
                           2, cood_x[i]-args.crop_width//2: cood_x[i]+args.crop_width//2, :]
            img_np = np.expand_dims(img_np, axis=0)
            size_np = np.expand_dims(
                np.array([img_np.shape[0], img_np.shape[1]]), axis=0)
            if not start:
                if args.compress:
                    hf.create_dataset(
                        "image_data",
                        data=img_np,
                        chunks=True,
                        maxshape=(None, None, None, 3),
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
                        chunks=True,
                        maxshape=(None, None, None, 3),
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
                 'thermalmapping', 'foxtechmapping'],
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
                 'thermalmapping', 'foxtechmapping'],
        help="The name of queries map you want to use"
    )
    parser.add_argument(
        "--queries_index",
        type=int,
        help="The index of queries flight you want to use. For satellite map, it is forced to be 0"
    )
    parser.add_argument("--crop_width", type=int, default=512)
    parser.add_argument("--sample_num", type=int)
    parser.add_argument("--compress", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index))):
        shutil.rmtree(os.path.join(datasets_folder, args.database_name + '_' + str(
            args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index)))
    os.mkdir(os.path.join(datasets_folder, args.database_name + '_' +
             str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index)))

    np.random.seed(0)
    create_h5_file(args, name='database', split='train')
    create_h5_file(args, name='queries', split='train')
    shutil.move(os.path.join(datasets_folder, f'{args.database_name}_{args.database_index}_train_database.h5'),
                os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'train_database.h5'))
    shutil.move(os.path.join(datasets_folder, f'{args.queries_name}_{args.queries_index}_train_queries.h5'),
                os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'train_queries.h5'))

    create_h5_file(args, name='database', split='val')
    create_h5_file(args, name='queries', split='val')
    shutil.move(os.path.join(datasets_folder, f'{args.database_name}_{args.database_index}_val_database.h5'),
                os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'val_database.h5'))
    shutil.move(os.path.join(datasets_folder, f'{args.queries_name}_{args.queries_index}_val_queries.h5'),
                os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'val_queries.h5'))

    # Not enough test data. Use val as test
    os.symlink(os.path.abspath(os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'val_database.h5')),
               os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'test_database.h5'))
    os.symlink(os.path.abspath(os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'val_queries.h5')),
               os.path.join(datasets_folder, args.database_name + '_' + str(args.database_index) + '_' + args.queries_name + '_' + str(args.queries_index), 'test_queries.h5'))
