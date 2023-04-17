import os

import cv2
import torch.utils.data as data

# DUTSDataset数据集读取
class DUTSDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        # 判断数据集根目录路径是否存在
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            # 得到训练的数据集图片和标签的路径
            self.image_root = os.path.join(root, "DUTS-TR", "DUTS-TR-Image")
            self.mask_root = os.path.join(root, "DUTS-TR", "DUTS-TR-Mask")
        else:
            # 得到验证的数据集图片和标签的路径
            self.image_root = os.path.join(root, "DUTS-TE", "DUTS-TE-Image")
            self.mask_root = os.path.join(root, "DUTS-TE", "DUTS-TE-Mask")
        # 断言判断路径是否存在
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        # 分别获取self.image_root和self.mask_root文件夹下图片和标签文件的名称，分别存储在image_names和mask_names中
        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check images and mask
        # 检查每一张图片是否有对应的mask
        re_mask_names = []
        for p in image_names:
            mask_name = p.replace(".jpg", ".png")
            assert mask_name in mask_names, f"{p} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        # 获取每一张图片和标签文件的路径
        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        # 根据传入索引idx可以得到对应图片和标签文件的路径
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        # 读取图片
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        assert image is not None, f"failed to read image: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = image.shape

        # 标签文件以灰度图的形式读取
        # 背景像素值设为0，前景像素值设为255
        target = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert target is not None, f"failed to read mask: {mask_path}"

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    train_dataset = DUTSDataset("./", train=True)
    print(len(train_dataset))

    val_dataset = DUTSDataset("./", train=False)
    print(len(val_dataset))

    i, t = train_dataset[0]
