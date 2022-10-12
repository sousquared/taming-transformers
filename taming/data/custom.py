import albumentations
import numpy as np
import os
import webdataset as wds
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset, DataLoader

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

from taming.data.utils import custom_collate


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomDataModuleForWebDataset:
    def __init__(self, train_urls, val_urls, size, batch_size, num_workers, random_crop=False):
        self.train_urls = train_urls
        self.val_urls = val_urls
        self.size = size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_crop = random_crop

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def prepare_train_dataset(self, urls):
        return wds.DataPipeline(
            wds.SimpleShardList(urls),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            wds.shuffle(10000),
            wds.decode("pilrgb"),
            wds.to_tuple("jpg", "json"),
            wds.map(self.preprocess_image),
            # partial=FalseはDataloaderのdrop_last=Trueと同じ効果
            wds.batched(self.batch_size, partial=False),
        )

    def prepare_val_dataset(self, urls):
        return wds.DataPipeline(
            wds.SimpleShardList(urls),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            wds.decode("pilrgb"),
            wds.to_tuple("jpg", "json"),
            wds.map(self.preprocess_image),
            # partial=FalseはDataloaderのdrop_last=Trueと同じ効果
            wds.batched(self.batch_size, partial=False),
        )

    def preprocess_image(self, sample):
        image, json = sample
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def prepare_data_loaders(self):
        train_dataset = self.prepare_train_dataset(urls=self.train_urls)
        val_dataset = self.prepare_val_dataset(urls=self.val_urls)

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=None,  # wds.Pipelineでbatch済み
            num_workers=self.num_workers,
            shuffle=False,  # wds.Pipelineでshuffle済み
            collate_fn=custom_collate,
        )

        val_data_loader = DataLoader(
            val_dataset,
            batch_size=None,  # wds.Pipelineでbatch済み
            num_workers=self.num_workers,
            shuffle=False,  # wds.Pipelineでshuffle済み
            collate_fn=custom_collate,
        )

        return train_data_loader, val_data_loader
