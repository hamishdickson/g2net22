import torch
import numpy as np
import cv2


class G2Net2Dataset(torch.utils.data.Dataset):
    def __init__(self, df, config, transforms):
        self.df = df

        self.event_ids = df.id.values
        self.targets = df.target.values
        self.sim_ids = df.sim_id.values
        self.image_locs = df.image_loc.values
        self.config = config
        self.transforms = transforms


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        target = np.float32(self.targets[idx])
        image_loc = self.image_locs[idx]
        sim_id = self.sim_ids[idx]
        

        img = np.ones((3, self.config.TARGET_HEIGHT, self.config.TARGET_WIDTH_COMPRESSED), dtype=np.float32)

        if image_loc == "train":
            # then it's from the supplied training data
            L1_filename = f"{self.config.save_root}/preprocessed_train/{event_id}_L1.png"
            H1_filename = f"{self.config.save_root}/preprocessed_train/{event_id}_H1.png"
            
            img[0] = cv2.imread(H1_filename, -1)
            img[1] = cv2.imread(L1_filename, -1)
        elif (sim_id < 0) and (image_loc == "fake_noise"):
            # then it's fake noise only
            L1_filename = f"{self.config.save_root}/fake_noise/{event_id}_L1.png"
            H1_filename = f"{self.config.save_root}/fake_noise/{event_id}_H1.png"

            img[0] = cv2.imread(H1_filename, -1)
            img[1] = cv2.imread(L1_filename, -1)
        elif sim_id < 0:
            # then it's realistic noise only
            L1_filename = f"{self.config.save_root}/realistic_noise/images/{event_id}/L1.png"
            H1_filename = f"{self.config.save_root}/realistic_noise/images/{event_id}/H1.png"

            img[0] = cv2.imread(H1_filename, -1)
            img[1] = cv2.imread(L1_filename, -1)
        elif image_loc == "fake_noise":
            # then it's fake noise + some wave
            L1_filename = f"{self.config.save_root}/fake_noise/{event_id}_L1.png"
            H1_filename = f"{self.config.save_root}/fake_noise/{event_id}_H1.png"

            img[0] = cv2.imread(H1_filename, -1)
            img[1] = cv2.imread(L1_filename, -1)

            L1_sig = f"{self.config.save_root}/pure_signal/{sim_id}_L1.png"
            H1_sig = f"{self.config.save_root}/pure_signal/{sim_id}_H1.png"

            img_sig = np.ones((3, self.config.TARGET_HEIGHT, self.config.TARGET_WIDTH_COMPRESSED), dtype=np.float32)

            img_sig[0] = cv2.imread(L1_sig, -1)
            img_sig[1] = cv2.imread(H1_sig, -1)

            SIGNAL_LOW = 0.05 #0.02
            SIGNAL_HIGH = 0.25 # 0.10

            signal_strength = np.random.uniform(SIGNAL_LOW, SIGNAL_HIGH)

            img[0] = img[0] + signal_strength * img_sig[0]
            img[1] = img[1] + signal_strength * img_sig[1]

        else:
            # then it's realistic noise + some wave
            
            L1_filename = f"{self.config.save_root}/realistic_noise/images/{event_id}/L1.png"
            H1_filename = f"{self.config.save_root}/realistic_noise/images/{event_id}/H1.png"

            img[0] = cv2.imread(H1_filename, -1)
            img[1] = cv2.imread(L1_filename, -1)

            L1_sig = f"{self.config.save_root}/pure_signal/{sim_id}_L1.png"
            H1_sig = f"{self.config.save_root}/pure_signal/{sim_id}_H1.png"

            img_sig = np.ones((3, self.config.TARGET_HEIGHT, self.config.TARGET_WIDTH_COMPRESSED), dtype=np.float32)

            img_sig[0] = cv2.imread(L1_sig, -1)
            img_sig[1] = cv2.imread(H1_sig, -1)

            SIGNAL_LOW = 0.05
            SIGNAL_HIGH = 0.25

            signal_strength = np.random.uniform(SIGNAL_LOW, SIGNAL_HIGH)

            img[0] = img[0] + signal_strength * img_sig[0]
            img[1] = img[1] + signal_strength * img_sig[1]


        gaussian_noise = np.random.randn(*img.shape)
        img += self.config.gaussian * gaussian_noise
        
        img = self.transforms(image=img.T)["image"]
        # # raise Exception(img)
        img = img[:,:,:2].T

        # raise Exception(img.shape)

        return img, target





class G2Net2Dataset_Valid(torch.utils.data.Dataset):
    def __init__(self, df, config, transforms):
        self.df = df

        self.event_ids = df.id.values
        self.targets = df.target.values
        self.config = config
        self.transforms = transforms


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        target = np.float32(self.targets[idx])
        
        img = np.ones((3, self.config.TARGET_HEIGHT, self.config.TARGET_WIDTH_COMPRESSED), dtype=np.float32)

        L1_filename = f"{self.config.save_root}/preprocessed_train/{event_id}_L1.png"
        H1_filename = f"{self.config.save_root}/preprocessed_train/{event_id}_H1.png"

        img[0] = cv2.imread(H1_filename, -1)
        img[1] = cv2.imread(L1_filename, -1)

        img = self.transforms(image=img.T)["image"]
        # # raise Exception(img)
        img = img[:,:,:2].T

        return img, target



class G2Net2Dataset_Test(torch.utils.data.Dataset):
    def __init__(self, df, config, transforms):
        self.df = df

        self.event_ids = df.id.values
        self.config = config
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        
        img = np.ones((3, self.config.TARGET_HEIGHT, self.config.TARGET_WIDTH_COMPRESSED), dtype=np.float32)

        L1_filename = f"{self.config.save_root}/preprocessed_test/{event_id}_L1.png"
        H1_filename = f"{self.config.save_root}/preprocessed_test/{event_id}_H1.png"

        img[0] = cv2.imread(H1_filename, -1)
        img[1] = cv2.imread(L1_filename, -1)

        img = self.transforms(image=img.T)["image"]
        # # raise Exception(img)
        img = img[:,:,:2].T

        return img



###############################################

class DiscriminatorDataset(torch.utils.data.Dataset):
    def __init__(self, df, config):
        self.df = df

        self.event_ids = df.id.values
        self.targets = df.dtarget.values
        self.config = config


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        target = np.float32(self.targets[idx])

        img = np.zeros((2, self.config.TARGET_HEIGHT, self.config.TARGET_WIDTH_COMPRESSED), dtype=np.float32)

        if target == 1:
            L1_filename = f"{self.config.save_root}/preprocessed_test/{event_id}_L1.png"
            H1_filename = f"{self.config.save_root}/preprocessed_test/{event_id}_H1.png"
            # L1_filename = f"{self.config.save_root}/preprocessed_train/{event_id}_L1.png"
            # H1_filename = f"{self.config.save_root}/preprocessed_train/{event_id}_H1.png"
        else:
            L1_filename = f"{self.config.save_root}/{self.config.generated_data_loc}/{event_id}_L1.png"
            H1_filename = f"{self.config.save_root}/{self.config.generated_data_loc}/{event_id}_H1.png"           

        img[0] = cv2.imread(H1_filename, -1)
        img[1] = cv2.imread(L1_filename, -1)


        return img, target