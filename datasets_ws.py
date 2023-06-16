import os
import torch
import faiss
import faiss.contrib.torch_utils
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
import h5py
import time

base_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)

# Translation has different normalization for tanh activation
base_translation_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images = torch.cat([e[0] for e in batch])
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(
        zip(triplets_local_indexes, triplets_global_indexes)
    ):
        local_indexes += (
            len(global_indexes) * i
        )  # Increment local indexes by offset (len(global_indexes) is 12)
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes

class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(
        self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train", loading_queries=True
    ):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.split = split
        # self.dataset_folder = join(datasets_folder, dataset_name, "images", split)
        # if not os.path.exists(self.dataset_folder): raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        self.resize = args.resize
        self.test_method = args.test_method

        # Redirect datafolder path to h5
        self.database_folder_h5_path = join(
            datasets_folder, dataset_name, split + "_database.h5"
        )
        if loading_queries:
            self.queries_folder_h5_path = join(
                datasets_folder, dataset_name, split + "_queries.h5"
            )
        else:
            # Do not load queries when generating thermal with pix2pix
            self.queries_folder_h5_path = join(
                datasets_folder, dataset_name, split + "_database.h5"
            ) 
        database_folder_h5_df = h5py.File(self.database_folder_h5_path, "r")
        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r")

        # Map name to index
        self.database_name_dict = {}
        self.queries_name_dict = {}

        # Duplicated elements are removed below
        for index, database_image_name in enumerate(database_folder_h5_df["image_name"]):
            self.database_name_dict[database_image_name.decode(
                "UTF-8")] = index
        for index, queries_image_name in enumerate(queries_folder_h5_df["image_name"]):
            self.queries_name_dict[queries_image_name.decode("UTF-8")] = index

        # Read paths and UTM coordinates for all images.
        # database_folder = join(self.dataset_folder, "database")
        # queries_folder  = join(self.dataset_folder, "queries")
        # if not os.path.exists(database_folder): raise FileNotFoundError(f"Folder {database_folder} does not exist")
        # if not os.path.exists(queries_folder) : raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        self.database_paths = sorted(self.database_name_dict)
        self.queries_paths = sorted(self.queries_name_dict)
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.database_paths]
        ).astype(np.float)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.queries_paths]
        ).astype(np.float)

        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(
            self.queries_utms,
            radius=args.val_positive_dist_threshold,
            return_distance=False,
        )

        # Find hard_negatives_per_query. Hard negative is out of prior position threshold and we don't care
        if args.prior_location_threshold != -1:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.hard_negatives_per_query = knn.radius_neighbors(
                self.queries_utms,
                radius=args.prior_location_threshold,
                return_distance=False,
            )

        # Add database, queries prefix
        for i in range(len(self.database_paths)):
            self.database_paths[i] = "database_" + self.database_paths[i]
        for i in range(len(self.queries_paths)):
            self.queries_paths[i] = "queries_" + self.queries_paths[i]

        self.images_paths = list(self.database_paths) + \
            list(self.queries_paths)

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

        # Close h5 and initialize for h5 reading in __getitem__
        self.database_folder_h5_df = None
        self.queries_folder_h5_df = None
        database_folder_h5_df.close()
        queries_folder_h5_df.close()

        identity_transform = transforms.Lambda(lambda x: x)
        self.query_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3)
                if self.args.G_gray
                else identity_transform,
                base_transform
            ]
        )

    def __getitem__(self, index):
        # Init
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r")
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r")
        if self.is_index_in_queries(index):
            if self.args.G_contrast:
                img = self.query_transform(
                    transforms.functional.adjust_contrast(self._find_img_in_h5(index), contrast_factor=3))
            else:
                img = self.query_transform(
                    self._find_img_in_h5(index))
        else:
            img = self._find_img_in_h5(index)
            img = base_transform(img)
        # With database images self.test_method should always be "hard_resize"
        if self.test_method == "hard_resize":
            # self.test_method=="hard_resize" is the default, resizes all images to the same size.
            img = transforms.functional.resize(img, self.resize)
        else:
            img = self._test_query_transform(img)
        return img, index

    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            # self.test_method=="single_query" is used when queries have varying sizes, and can't be stacked in a batch.
            processed_img = transforms.functional.resize(img, min(self.resize))
        elif self.test_method == "central_crop":
            # Take the biggest central crop of size self.resize. Preserves ratio.
            scale = max(self.resize[0] / H, self.resize[1] / W)
            processed_img = torch.nn.functional.interpolate(
                img.unsqueeze(0), scale_factor=scale
            ).squeeze(0)
            processed_img = transforms.functional.center_crop(
                processed_img, self.resize
            )
            assert processed_img.shape[1:] == torch.Size(
                self.resize
            ), f"{processed_img.shape[1:]} {self.resize}"
        elif (
            self.test_method == "five_crops"
            or self.test_method == "nearest_crop"
            or self.test_method == "maj_voting"
        ):
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.resize)
            processed_img = transforms.functional.resize(img, shorter_side)
            processed_img = torch.stack(
                transforms.functional.five_crop(processed_img, shorter_side)
            )
            assert processed_img.shape == torch.Size(
                [5, 3, shorter_side, shorter_side]
            ), f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img

    def _find_img_in_h5(self, index, database_queries_split=None):
        # Find inside index for h5
        if database_queries_split is None:
            image_name = "_".join(self.images_paths[index].split("_")[1:])
            database_queries_split = self.images_paths[index].split("_")[0]
        else:
            if database_queries_split == "database":
                image_name = "_".join(
                    self.database_paths[index].split("_")[1:])
            elif database_queries_split == "queries":
                image_name = "_".join(self.queries_paths[index].split("_")[1:])
            else:
                raise KeyError("Dont find correct database_queries_split!")

        if database_queries_split == "database":
            img = Image.fromarray(
                self.database_folder_h5_df["image_data"][
                    self.database_name_dict[image_name]
                ]
            )
        elif database_queries_split == "queries":
            img = Image.fromarray(
                self.queries_folder_h5_df["image_data"][
                    self.queries_name_dict[image_name]
                ]
            )
        else:
            raise KeyError("Dont find correct database_queries_split!")

        return img

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"

    def get_positives(self):
        return self.soft_positives_per_query

    def get_hard_negatives(self):
        return self.hard_negatives_per_query

    def __del__(self):
        if (
            hasattr(self, "database_folder_h5_df")
            and self.database_folder_h5_df is not None
        ):
            self.database_folder_h5_df.close()
            self.queries_folder_h5_df.close()
        
    def find_black_region(self):
        # Only for thermal
        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r")
        queries_with_black_region = []
        for index, path in enumerate(self.queries_paths):
            real_path = path[len("queries_"):]
            image_index = self.queries_name_dict[real_path]
            image = queries_folder_h5_df["image_data"][image_index]
            if np.count_nonzero(image==0):
                queries_with_black_region.append(index)
        queries_folder_h5_df.close()
        return queries_with_black_region

    def is_index_in_queries(self, index):
        if index >= self.database_num:
            return True
        else:
            return False

class PCADataset(BaseDataset):
    def __init__(
        self, args, datasets_folder="datasets", dataset_name="pitts30k"
        ):
        # Always use train for PCA fit
        super().__init__(args, datasets_folder, dataset_name, split="train")

    def __getitem__(self, index):
        # return base_transform(path_to_pil_img(self.images_paths[index]))
        # Init
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r")
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r")
        img = self._find_img_in_h5(index)
        img = base_transform(img)
        # PCA resize or not?
        if self.test_method == "hard_resize":
            # self.test_method=="hard_resize" is the default, resizes all images to the same size.
            img = transforms.functional.resize(img, self.resize)
        else:
            img = self._test_query_transform(img)
        return img

class TripletsDataset(BaseDataset):
    """Dataset used for training, it is used to compute the triplets
    with TripletsDataset.compute_triplets() with various mining methods.
    If is_inference == True, uses methods of the parent class BaseDataset,
    this is used for example when computing the cache, because we compute features
    of each image, not triplets.
    """

    def __init__(
        self,
        args,
        datasets_folder="datasets",
        dataset_name="pitts30k",
        split="train",
        negs_num_per_query=10,
    ):
        super().__init__(args, datasets_folder, dataset_name, split)
        self.mining = args.mining
        self.neg_samples_num = (
            args.neg_samples_num
        )  # Number of negatives to randomly sample
        self.negs_num_per_query = (
            negs_num_per_query  # Number of negatives per query in each batch
        )
        if (
            self.mining == "full"
        ):  # "Full database mining" keeps a cache with last used negatives
            self.neg_cache = [
                np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)
            ]
        self.is_inference = False

        identity_transform = transforms.Lambda(lambda x: x)
        self.resized_transform = transforms.Compose(
            [
                transforms.Resize(self.resize)
                if self.resize is not None
                else identity_transform,
                base_transform,
            ]
        )

        self.query_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3)
                if self.args.G_gray
                else identity_transform,
                transforms.ColorJitter(brightness=args.brightness)
                if args.brightness != None
                else identity_transform,
                transforms.ColorJitter(contrast=args.contrast)
                if args.contrast != None
                else identity_transform,
                transforms.ColorJitter(saturation=args.saturation)
                if args.saturation != None
                else identity_transform,
                transforms.ColorJitter(hue=args.hue)
                if args.hue != None
                else identity_transform,
                transforms.RandomPerspective(args.rand_perspective)
                if args.rand_perspective != None
                else identity_transform,
                transforms.RandomResizedCrop(
                    size=self.resize, scale=(1 - args.random_resized_crop, 1)
                )
                if args.random_resized_crop != None
                else identity_transform,
                transforms.RandomRotation(degrees=args.random_rotation)
                if args.random_rotation != None
                else identity_transform,
                self.resized_transform,
            ]
        )

        # Find hard_positives_per_query, which are within train_positives_dist_threshold (10 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(
            knn.radius_neighbors(
                self.queries_utms,
                radius=args.train_positives_dist_threshold,  # 10 meters
                return_distance=False,
            )
        )

        # Some queries might have no positive, we should remove those queries.
        queries_without_any_hard_positive = np.where(
            np.array([len(p)
                     for p in self.hard_positives_per_query], dtype=object) == 0
        )[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(
                f"There are {len(queries_without_any_hard_positive)} queries without any positives "
                + "within the training set. They won't be considered as they're useless for training."
            )
        # Remove queries without positives
        self.hard_positives_per_query = np.delete(
            self.hard_positives_per_query, queries_without_any_hard_positive
        )
        self.queries_paths = np.delete(
            self.queries_paths, queries_without_any_hard_positive
        )

        # Recompute images_paths and queries_num because some queries might have been removed
        self.images_paths = list(self.database_paths) + \
            list(self.queries_paths)
        self.queries_num = len(self.queries_paths)

        # msls_weighted refers to the mining presented in MSLS paper's supplementary.
        # Basically, images from uncommon domains are sampled more often. Works only with MSLS dataset.
        if self.mining == "msls_weighted":
            notes = [p.split("@")[-2] for p in self.queries_paths]
            try:
                night_indexes = np.where(
                    np.array([n.split("_")[0] == "night" for n in notes])
                )[0]
                sideways_indexes = np.where(
                    np.array([n.split("_")[1] == "sideways" for n in notes])
                )[0]
            except IndexError:
                raise RuntimeError(
                    "You're using msls_weighted mining but this dataset "
                    + "does not have night/sideways information. Are you using Mapillary SLS?"
                )
            self.weights = np.ones(self.queries_num)
            assert (
                len(night_indexes) != 0 and len(sideways_indexes) != 0
            ), "There should be night and sideways images for msls_weighted mining, but there are none. Are you using Mapillary SLS?"
            self.weights[night_indexes] += self.queries_num / \
                len(night_indexes)
            self.weights[sideways_indexes] += self.queries_num / \
                len(sideways_indexes)
            self.weights /= self.weights.sum()
            logging.info(
                f"#sideways_indexes [{len(sideways_indexes)}/{self.queries_num}]; "
                + "#night_indexes; [{len(night_indexes)}/{self.queries_num}]"
            )

    def __getitem__(self, index):
        if self.is_inference:
            # At inference time return the single image. This is used for caching or computing NetVLAD's clusters
            return super().__getitem__(index)

        # Init
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r")
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r")

        query_index, best_positive_index, neg_indexes = torch.split(
            self.triplets_global_indexes[index], (1,
                                                  1, self.negs_num_per_query)
        )

        if self.args.G_contrast and self.split!="extended": # Avoid double CE for extended dataset (TGM has already generated enhanced results)
            query = self.query_transform(
                transforms.functional.adjust_contrast(self._find_img_in_h5(query_index, "queries"), contrast_factor=3))
        else:
            query = self.query_transform(
                self._find_img_in_h5(query_index, "queries"))
                
        positive = self.resized_transform(
            self._find_img_in_h5(best_positive_index, "database")
        )
        negatives = [
            self.resized_transform(self._find_img_in_h5(i, "database"))
            for i in neg_indexes
        ]
        images = torch.stack((query, positive, *negatives), 0)
        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat(
                (
                    triplets_local_indexes,
                    torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3),
                )
            )
        return images, triplets_local_indexes, self.triplets_global_indexes[index]

    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            return super().__len__()
        else:
            return len(self.triplets_global_indexes)

    def compute_triplets(self, args, model, model_db=None):
        self.is_inference = True
        if self.mining == "full":
            self.compute_triplets_full(args, model, model_db)
        elif self.mining == "partial" or self.mining == "msls_weighted":
            self.compute_triplets_partial(args, model, model_db)
        elif self.mining == "random":
            self.compute_triplets_random(args, model, model_db)

    @staticmethod
    def compute_cache(args, model, model_db, subset_ds, cache_shape):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""

        # RAMEfficient2DMatrix can be replaced by np.zeros, but using
        # RAMEfficient2DMatrix is RAM efficient for full database mining.
        if args.use_faiss_gpu:
            cache = RAMEfficient2DMatrixGPU(cache_shape, dtype=torch.float32, device=args.device)
        else:
            cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)

        if model_db is not None:
            subset_db_dl = DataLoader(
                dataset=subset_ds[0],
                num_workers=args.num_workers,
                batch_size=args.infer_batch_size,
                shuffle=False,
                pin_memory=(args.device == "cuda"),
            )
            subset_qr_dl = DataLoader(
                dataset=subset_ds[1],
                num_workers=args.num_workers,
                batch_size=args.infer_batch_size,
                shuffle=False,
                pin_memory=(args.device == "cuda"),
            )
            model = model.eval()
            model_db = model_db.eval()
            
            with torch.no_grad():
                for images, indexes in tqdm(subset_db_dl, ncols=100):
                    images = images.to(args.device)
                    features = model_db(images)
                    if args.use_faiss_gpu:
                        cache[indexes] = features
                    else:
                        cache[indexes.numpy()] = features.cpu().numpy()
                        
                for images, indexes in tqdm(subset_qr_dl, ncols=100):
                    images = images.to(args.device)
                    features = model(images)
                    if args.use_faiss_gpu:
                        cache[indexes] = features
                    else:
                        cache[indexes.numpy()] = features.cpu().numpy()
        else:
            subset_dl = DataLoader(
                dataset=subset_ds,
                num_workers=args.num_workers,
                batch_size=args.infer_batch_size,
                shuffle=False,
                pin_memory=(args.device == "cuda"),
            )
            model = model.eval()
            
            with torch.no_grad():
                for images, indexes in tqdm(subset_dl, ncols=100):
                    images = images.to(args.device)
                    features = model(images)
                    if args.use_faiss_gpu:
                        cache[indexes] = features
                    else:
                        cache[indexes.numpy()] = features.cpu().numpy()
        return cache

    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(
                f"For query {self.queries_paths[query_index]} "
                + f"with index {query_index} features have not been computed!\n"
                + "There might be some bug with caching"
            )
        return query_features

    def get_best_positive_index(self, args, query_index, cache, query_features):
        positives_features = cache[self.hard_positives_per_query[query_index]]
        if args.use_faiss_gpu:
            faiss_index = faiss.GpuIndexFlatL2(
                self.gpu_resources[0], args.features_dim)
        else:
            faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(
            query_features.reshape(1, -1), 1)
        best_positive_index = self.hard_positives_per_query[query_index][best_positive_num[0]].item(
        )
        return best_positive_index

    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_features = cache[neg_samples]
        if args.use_faiss_gpu:
            faiss_index = faiss.GpuIndexFlatL2(self.gpu_resources[1], args.features_dim)
        else:
            faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)
        # Search the 10 nearest negatives (further than 25 meters and nearest in features space)
        _, neg_nums = faiss_index.search(
            query_features.reshape(1, -1), self.negs_num_per_query
        )
        if args.use_faiss_gpu:
            neg_nums = neg_nums.reshape(-1).cpu()
        else:
            neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        if not hasattr(neg_indexes, "__len__"):
            neg_indexes = np.expand_dims(neg_indexes, 0)
        return neg_indexes

    def compute_triplets_random(self, args, model, model_db):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(
            self.queries_num, args.cache_refresh_rate, replace=False
        )
        # Take all the positives
        positives_indexes = [
            self.hard_positives_per_query[i] for i in sampled_queries_indexes
        ]
        positives_indexes = [
            p for pos in positives_indexes for p in pos
        ]  # Flatten list of lists to a list
        positives_indexes = list(np.unique(positives_indexes))

        # Compute the cache only for queries and their positives, in order to find the best positive
        if model_db is not None:
            subset_db_ds = Subset(
                self, positives_indexes
            )
            subset_qr_ds = Subset(
                self, list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, [subset_db_ds, subset_qr_ds], (len(self), args.features_dim)
            )
        else:
            subset_ds = Subset(
                self, positives_indexes +
                list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, subset_ds, (len(self), args.features_dim)
            )
        
        if args.use_faiss_gpu:
            # Tmp memory for faiss
            self.gpu_resources = []
            for i in range(2):
                # 2 gpu resource for positive
                res = faiss.StandardGpuResources()
                res.setTempMemory(200 * 1024 * 1024)  # 200 MB
                self.gpu_resources.append(res)

        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(
                args, query_index, cache, query_features
            )

            # Choose some random database images, from those remove the soft_positives, and then take the first 10 images as neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.random.choice(
                self.database_num,
                size=self.negs_num_per_query + len(soft_positives),
                replace=False,
            )

            # Remove hard_negatives
            if args.prior_location_threshold == -1:
                neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)[
                    : self.negs_num_per_query
                ]
            else:
                neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)
                hard_negatives = self.hard_negatives_per_query[query_index]
                neg_indexes = np.intersect1d(neg_indexes, hard_negatives, assume_unique=True)[
                    : self.negs_num_per_query
                ]
            
            self.triplets_global_indexes.append(
                (query_index, best_positive_index, *neg_indexes)
            )

        del cache
        # Remove Tmp memory for faiss
        if args.use_faiss_gpu:
            del self.gpu_resources

        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(
            self.triplets_global_indexes)

    def compute_triplets_full(self, args, model, model_db):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(
            self.queries_num, args.cache_refresh_rate, replace=False
        )
        # Take all database indexes
        database_indexes = list(range(self.database_num))
        # Compute features for all images and store them in cache
        if model_db is not None:
            subset_db_ds = Subset(
                self, database_indexes
            )
            subset_qr_ds = Subset(
                self, list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, [subset_db_ds, subset_qr_ds], (len(self), args.features_dim)
            )
        else:
            subset_ds = Subset(
                self, database_indexes +
                list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, subset_ds, (len(self), args.features_dim)
            )

        if args.use_faiss_gpu:
            # Tmp memory for faiss
            self.gpu_resources = []
            for i in range(2):
                # 2 gpu resource for positive and negative
                res = faiss.StandardGpuResources()
                res.setTempMemory(200 * 1024 * 1024)  # 200 MB
                self.gpu_resources.append(res)

        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(
                args, query_index, cache, query_features
            )
            # Choose 1000 random database images (neg_indexes)
            neg_indexes = np.random.choice(
                self.database_num, self.neg_samples_num, replace=False
            )
            # Remove the eventual soft_positives from neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(
                neg_indexes, soft_positives, assume_unique=True)

            # Remove the eventual hard_negatives from neg_indexes
            if args.prior_location_threshold != -1:
                hard_negatives = self.hard_negatives_per_query[query_index]
                neg_indexes = np.intersect1d(neg_indexes, hard_negatives, assume_unique=True)

            # Concatenate neg_indexes with the previous top 10 negatives (neg_cache)
            neg_indexes = np.unique(
                np.concatenate([self.neg_cache[query_index], neg_indexes])
            )
            # Search the hardest negatives
            neg_indexes = self.get_hardest_negatives_indexes(
                args, cache, query_features, neg_indexes
            )
            # Update nearest negatives in neg_cache
            self.neg_cache[query_index] = neg_indexes
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))

        # Remove Tmp memory for faiss
        del cache
        if args.use_faiss_gpu:
            del self.gpu_resources

        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(
            self.triplets_global_indexes)

    def compute_triplets_partial(self, args, model, model_db):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        if self.mining == "partial":
            sampled_queries_indexes = np.random.choice(
                self.queries_num, args.cache_refresh_rate, replace=False
            )
        elif (
            self.mining == "msls_weighted"
        ):  # Pick night and sideways queries with higher probability
            sampled_queries_indexes = np.random.choice(
                self.queries_num, args.cache_refresh_rate, replace=False, p=self.weights
            )

        # Sample 1000 random database images for the negatives
        sampled_database_indexes = np.random.choice(
            self.database_num, self.neg_samples_num, replace=False
        )
        # Take all the positives
        positives_indexes = [
            self.hard_positives_per_query[i] for i in sampled_queries_indexes
        ]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        # Merge them into database_indexes and remove duplicates
        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))

        if model_db is not None:
            subset_db_ds = Subset(
                self, database_indexes
            )
            subset_qr_ds = Subset(
                self, list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, [subset_db_ds, subset_qr_ds], (len(self), args.features_dim)
            )
        else:
            subset_ds = Subset(
                self, database_indexes +
                list(sampled_queries_indexes + self.database_num)
            )
            cache = self.compute_cache(
                args, model, model_db, subset_ds, (len(self), args.features_dim)
            )

        if args.use_faiss_gpu:
            # Tmp memory for faiss
            self.gpu_resources = []
            for i in range(2):
                # 2 gpu resource for positive and negative
                res = faiss.StandardGpuResources()
                res.setTempMemory(200 * 1024 * 1024)  # 200 MB
                self.gpu_resources.append(res)

        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(
                args, query_index, cache, query_features
            )

            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(
                sampled_database_indexes, soft_positives, assume_unique=True
            )

            # Remove the eventual hard_negatives from neg_indexes
            if args.prior_location_threshold != -1:
                hard_negatives = self.hard_negatives_per_query[query_index]
                neg_indexes = np.intersect1d(neg_indexes, hard_negatives, assume_unique=True)

            # Take all database images that are negatives and are within the sampled database images (aka database_indexes)
            neg_indexes = self.get_hardest_negatives_indexes(
                args, cache, query_features, neg_indexes
            )
            self.triplets_global_indexes.append(
                (query_index, best_positive_index, *neg_indexes)
            )

        # Remove Tmp memory for faiss
        del cache
        if args.use_faiss_gpu:
            del self.gpu_resources

        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(
            self.triplets_global_indexes)


class RAMEfficient2DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""

    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]

    def __len__(self):
        return len(self.matrix)

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]

class RAMEfficient2DMatrixGPU:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""

    def __init__(self, shape, dtype=torch.float32, device=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.matrix = [None] * shape[0]

    def __len__(self):
        return len(self.matrix)

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.type(self.dtype).to(self.device)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return torch.stack([self.matrix[i] for i in index])
        else:
            return self.matrix[index]

class TranslationDataset(BaseDataset):
    """Dataset used for training, it is used to compute the pairs
    for image-to-image translation training.
    If is_inference == True, uses methods of the parent class BaseDataset.
    """

    def __init__(
        self,
        args,
        datasets_folder="datasets",
        dataset_name="pitts30k",
        split="train",
        clean_black_region=False,
        loading_queries=True
    ):
        super().__init__(args, datasets_folder, dataset_name, split, loading_queries)
        self.is_inference = False
        self.loading_queries = loading_queries

        identity_transform = transforms.Lambda(lambda x: x)
        self.resize = args.GAN_resize
        self.resized_transform = transforms.Compose(
            [
                transforms.Resize(self.resize)
                if self.resize is not None
                else identity_transform,
                base_translation_transform,
            ]
        )

        self.query_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1)
                if self.args.G_gray
                else identity_transform,
                self.resized_transform,
            ]
        )

        # Find hard_positives_per_query, which are within train_positives_dist_threshold (0.1 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(
            knn.radius_neighbors(
                self.queries_utms,
                radius=0.1,  # 0.1 meters
                return_distance=False,
            )
        )

        # Some queries might have no positive, we should remove those queries.
        queries_without_any_hard_positive = np.where(
            np.array([len(p)
                     for p in self.hard_positives_per_query], dtype=object) == 0
        )[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(
                f"There are {len(queries_without_any_hard_positive)} queries without any positives "
                + "within the training set. They won't be considered as they're useless for training."
            )
        # Remove queries without positives
        self.hard_positives_per_query = np.delete(
            self.hard_positives_per_query, queries_without_any_hard_positive
        )
        self.queries_paths = np.delete(
            self.queries_paths, queries_without_any_hard_positive
        )

        if clean_black_region:
            # Remove queries with black region
            queries_with_black_region = self.find_black_region()
            self.hard_positives_per_query = np.delete(
                self.hard_positives_per_query, queries_with_black_region
            )
            self.queries_paths = np.delete(
                self.queries_paths, queries_with_black_region
            )
            if len(queries_with_black_region) != 0:
                logging.info(
                    f"There are {len(queries_with_black_region)} queries with black regions "
                    + "within the training set. They won't be considered."
                )

        # Recompute images_paths and queries_num because some queries might have been removed
        self.images_paths = list(self.database_paths) + \
            list(self.queries_paths)
        self.queries_num = len(self.queries_paths)

    def __getitem__(self, index):
        # Init
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r")
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r")

        query_index, best_positive_index = torch.split(
            self.pairs_global_indexes[index], (1, 1)
        )

        if self.args.G_contrast:
            query = self.query_transform(
                transforms.functional.adjust_contrast(self._find_img_in_h5(query_index, "queries"), contrast_factor=3))
        else:
            query = self.query_transform(
                self._find_img_in_h5(query_index, "queries"))
            
        positive = self.resized_transform(
            self._find_img_in_h5(best_positive_index, "database")
        )
        return query, positive, self.queries_paths[query_index], self.database_paths[best_positive_index]

    def __len__(self):
        return len(self.pairs_global_indexes)
    
    def compute_pairs(self, args):
        self.is_inference = True
        self.compute_pairs_random(args)

    def get_best_positive_index(self, query_index):
        best_positive_index = self.hard_positives_per_query[query_index].item()
        return best_positive_index

    def compute_pairs_random(self, args):
        self.pairs_global_indexes = []
        # Take 1000 random queries
        if self.loading_queries:
            sampled_queries_indexes = np.random.choice(
                self.queries_num, args.cache_refresh_rate, replace=False
            )
        else:
            sampled_queries_indexes = np.arange(self.queries_num)

        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in sampled_queries_indexes:
            best_positive_index = self.get_best_positive_index(query_index)
            self.pairs_global_indexes.append(
                (query_index, best_positive_index)
            )

        # self.pairs_global_indexes is a tensor of shape [1000, 2]
        self.pairs_global_indexes = torch.tensor(self.pairs_global_indexes)