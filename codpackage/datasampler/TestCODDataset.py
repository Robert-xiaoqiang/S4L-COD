from .CODDataset import CODDataset

class TestCODDataset(CODDataset):
    def __init__(self, dataset_root, train_size):
        super().__init__(dataset_root, train_size)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.image_transform = transforms.Compose([
                transforms.Resize((train_size, train_size))
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

    def __getitem__(self, index):
        image_path, mask_path = self.data_list[index]

        image = Image.open(image_path).convert('RGB')

        image = self.image_transform(image)
        image_main_name = os.path.splitext(os.path.basename(image_path))[0]

        return image, mask_path, image_main_name        