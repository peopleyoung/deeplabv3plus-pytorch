import random
import cv2
import numpy as np
import os

class RandomFlip:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)

        return  img, mask

class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()

class Rotate:
    def __init__(self, limit=90, prob=1.0):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask

class Shift:
    def __init__(self, limit=50, prob=1.0):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            limit = self.limit
            dx = round(random.uniform(-limit, limit))
            dy = round(random.uniform(-limit, limit))

            height, width, channel = img.shape
            y1 = limit + 1 + dy
            y2 = y1 + height
            x1 = limit + 1 + dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, limit+1, limit + 1, limit + 1, limit +1,
                                      borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2, :]
            if mask is not None:
                mask1 = cv2.copyMakeBorder(mask, limit+1, limit + 1, limit + 1, limit +1,
                                      borderType=cv2.BORDER_REFLECT_101)
                mask = mask1[y1:y2, x1:x2, :]

        return img, mask

class Cutout:
    def __init__(self, num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, prob=1.):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            h = img.shape[0]
            w = img.shape[1]
            # c = img.shape[2]
            # img2 = np.ones([h, w], np.float32)
            for _ in range(self.num_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(max(0, y - self.max_h_size // 2), 0, h)
                y2 = np.clip(max(0, y + self.max_h_size // 2), 0, h)
                x1 = np.clip(max(0, x - self.max_w_size // 2), 0, w)
                x2 = np.clip(max(0, x + self.max_w_size // 2), 0, w)
                img[y1: y2, x1: x2, :] = self.fill_value
                if mask is not None:
                    mask[y1: y2, x1: x2, :] = self.fill_value
        return img, mask

class Rescale(object):
    def __init__(self, output_size, prob=0.75):
        self.prob = prob
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size

    def __call__(self, image, label):
        if random.random() < self.prob:
            raw_h, raw_w = image.shape[:2]

            img = cv2.resize(image, (self.output_size, self.output_size))
            lbl = cv2.resize(label, (self.output_size, self.output_size))

            h, w = img.shape[:2]

            if h > raw_w:
                i = random.randint(0, h - raw_h)
                j = random.randint(0, w - raw_h)
                img = img[i:i + raw_h, j:j + raw_h]
                lbl = lbl[i:i + raw_h, j:j + raw_h]
            else:
                res_h = raw_w - h
                img = cv2.copyMakeBorder(img, res_h, 0, res_h, 0, borderType=cv2.BORDER_REFLECT)
                lbl = cv2.copyMakeBorder(lbl, res_h, 0, res_h, 0, borderType=cv2.BORDER_REFLECT)
            return img, lbl
        else:
            return image, label

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None, img_name=None, save_path_img=None, save_path_mask=None):
        for i in range(len(self.transforms)):
            x, mask = self.transforms[i](x, mask)
            cv2.imwrite(os.path.join(save_path_img, img_name + '_new_' + str(i) + '.jpg'), x)
            cv2.imwrite(os.path.join(save_path_mask, img_name + '_new_' + str(i) + '.png'), mask)


class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x, mask = t(x, mask)
        return x, mask


class OneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            x, mask = self.first(x, mask)
        else:
            x, mask = self.second(x, mask)
        return x, mask

transform = DualCompose([
                RandomFlip(),
                RandomRotate90(),
                Rotate(),
                Shift(),
                Cutout(),
                Rescale(512),
            ])
if __name__ == '__main__':

    img_path = r'./DRIVE/training/new_images'
    mask_path = r'./DRIVE/training/new_1st_manual'
    img_save_path = r'./DRIVE/training/new_images'
    mask_save_path = r'./DRIVE/training/new_1st_manual'
    transform = DualCompose([
                RandomFlip(),
                RandomRotate90(),
                Rotate(),
                Shift(),
                Cutout(),
                Rescale(512),
            ])
    for file in os.listdir(img_path):
        name = file.split('.')[0]
        img = cv2.imread(os.path.join(img_path, file))
        mask = cv2.imread(os.path.join(mask_path, name+'.png'))
    
        transform(img, mask, name, img_save_path, mask_save_path)
    
