from numpy.core.numeric import identity
from torch.utils.data import  Dataset
import os
from PIL import Image
import numpy as np
from torchvision import  transforms
import torch
from glob import glob
import scipy.io as scio

class VGG2MixDataset(Dataset):
    def __init__(self, hp, transform=None):
        super(VGG2MixDataset, self).__init__()
        self.data = []
        self.hp = hp
        self.transform = transform
        self.ddfa_dataset = DataSet_300WLP(hp.data.train_300WLP_dir,transform)
        self.vgg_dataset = VGG2Dataset(hp.data.dataset_list,transform)
        self.len_vgg = len(self.vgg_dataset)
        self.len_ddfa = len(self.ddfa_dataset)
    
    def __getitem__(self, index):
        label = {}
        if index < self.len_vgg:
            label =  self.ddfa_dataset.__getitem__(index % self.len_ddfa)
        else:
            label =  self.vgg_dataset.__getitem__(index - self.len_vgg)
        return label
    def __len__(self):
        return self.len_vgg*2


class VGG2Dataset(Dataset):
    def __init__(self, indexfile, transform=None):
        super(VGG2Dataset, self).__init__()
        self.indexfile = indexfile
        self.data = []
        self.transform = transform
        self.transform_raw = transforms.Compose([
            transforms.CenterCrop((224,224)),    
            transforms.ToTensor()
        ])
        if not os.path.exists(self.indexfile):
            raise ValueError("Cannot find the index file!")
        with open(indexfile,'r') as f:
            lines = f.readlines()
        root_dir = lines[0].split('/n')[0]
        classes, class_to_idx = self._find_classes(root_dir)
        # print(len(classes))
        # exit()
        self.class_to_idx = class_to_idx
        #print(class_to_idx)
        for idx,line in enumerate(lines): 
            landmark_path=line.replace('\n', '')
            image_path = landmark_path.replace('landmark','train_align3').replace('npy','jpg')
            class_c =  landmark_path.split('/')[-2]
            self.data.append((image_path,landmark_path,class_to_idx[class_c]))
           
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        label = {}
        imgname, ldname, id = self.data[index]
        image=Image.open(imgname)
        #print(ldname)
        landmark=np.load(ldname,allow_pickle=True).astype('float32').T
        #print(landmark)
        raw_img = self.transform_raw(image)
        img = self.transform(image)
        landmark = torch.from_numpy(landmark) - 112
        landmark[1] = -landmark[1]
        label['image'] = img
        label['raw_img'] = raw_img
        label['landmark'] = landmark[:2]
        label['id'] = id
        return label

    def __len__(self):
        return len(self.data)

class DataSet_300WLP(Dataset):
    def __init__(self, root, transform):
        super(DataSet_300WLP, self).__init__()
        self.root = root
        self.data = []
        self.transform = transform
        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")
        types = ('*.jpg', '*.png')
        image_path_list= []
        for files in types:
            image_path_list.extend(glob(os.path.join(root, files)))
        #file_list_path = root  + '../file_path_list_AFLW2000_align.txt'
        #print(len(image_path_list))
        #count=0
        
        for ori_img_path in image_path_list:
            # if not '_f_' in ori_img_path:
            ori_mat_path = ori_img_path.replace('image_align','landmark_align')[:-3] + 'npy'
            self.data.append([ori_img_path, ori_mat_path])
      

    def __getitem__(self, index):
        #index = 0
        imgname, mat_path = self.data[index]
        img = Image.open(imgname)
        raw_image = transforms.ToTensor()(img)
        lms = np.load(mat_path) - 112
        lms = torch.from_numpy(lms)
        lms[:,1] = -lms[:,1]
        #print(lms)
        img = self.transform(img)
        label = {}
        label['image'] = img
        label['raw_img'] = raw_image
        label['landmark'] = (lms.T)[:2]
        label['id'] = -1
        return label

    def __len__(self):
        return len(self.data)

class AFLW2000DataSet(Dataset):
    def __init__(self, root,transform):
        self.root = root
        self.data = []
        self.transform = transform
        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")
        types = ('*.jpg', '*.png')
        image_path_list= []
        for files in types:
            image_path_list.extend(glob(os.path.join(root, files)))
        if not os.path.exists('data/alfw.txt'):

            with open('data/alfw.txt','w') as f:
                for ori_img_path in image_path_list:
                    f.write(ori_img_path)
                    f.write('\n')
        #file_list_path = root  + '../file_path_list_AFLW2000_align.txt'
        #print(len(image_path_list))
        #count=0
        
        for ori_img_path in image_path_list:
            ori_mat_path = ori_img_path.replace('image_align','landmark_align')[:-3] + 'npy'
            self.data.append([ori_img_path, ori_mat_path])
      

    def __getitem__(self, index):
        #index = 0
        imgname, mat_path = self.data[index]
        img = Image.open(imgname)
        raw_image = transforms.ToTensor()(img)
        lms = np.load(mat_path) - 112
        lms[:,1] = -lms[:,1]
        img = self.transform(img)
        return img, lms, raw_image, imgname

    def __len__(self):
        return len(self.data)

class LFWDataSet(Dataset):
    def __init__(self, root,transform):
        self.root = root
        self.data = []
        self.transform = transform
        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")
        self.labels = np.load('./propressing/lfw_list.npy')
        for i in range(12000):
            image_path = os.path.join(root,f'{i+1}.jpg')
            self.data.append(image_path)
      

    def __getitem__(self, index):
        imgname1 = self.data[index*2]
        imgname2 = self.data[index*2+1]
        img1 = Image.open(imgname1)
        img1 = self.transform(img1)
        img2 = Image.open(imgname2)
        img2=  self.transform(img2)
        label = self.labels[index]
        return img1, img2, label

    def __len__(self):
        return len(self.data) // 2

class CFPDataSet(Dataset):
    def __init__(self, root,transform):
        self.root = root
        self.data = []
        self.transform = transform
        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")
        self.labels = np.load('./propressing/cfp_fp_list.npy')
        for i in range(14000):
            image_path = os.path.join(root,f'{i+1}.jpg')
            self.data.append(image_path)
      

    def __getitem__(self, index):
        imgname1 = self.data[index*2]
        imgname2 = self.data[index*2+1]
        img1 = Image.open(imgname1)
        img1 = self.transform(img1)
        img2 = Image.open(imgname2)
        img2=  self.transform(img2)
        label = self.labels[index]
        return img1, img2, label

    def __len__(self):
        return len(self.data) // 2
        
class AgeDBDataSet(Dataset):
    def __init__(self, root,transform):
        self.root = root
        self.data = []
        self.transform = transform
        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")
        self.labels = np.load('./propressing/agedb_30_list.npy')
        for i in range(12000):
            image_path = os.path.join(root,f'{i+1}.jpg')
            self.data.append(image_path)
      

    def __getitem__(self, index):
        imgname1 = self.data[index*2]
        imgname2 = self.data[index*2+1]
        img1 = Image.open(imgname1)
        img1 = self.transform(img1)
        img2 = Image.open(imgname2)
        img2=  self.transform(img2)
        label = self.labels[index]
        return img1, img2, label

    def __len__(self):
        return len(self.data) // 2
        
class MICCDataSet(Dataset):
    def __init__(self, root,transform):
        self.root = root
        self.data = []
        self.transform = transform
        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")
        file_path = os.path.join(root,'selected_file.txt')
        with open(file_path) as f:
            filelist = f.readlines()
        for image in filelist:
            image = image.replace("\n","")
            image_path = os.path.join(root,image)
            identity = int(image.split('/')[-2])
            self.data.append([image_path,identity])
        self.data = self.data[::50]
        print('micc:', len(self.data))


            
            
            
        
        
        # for i in range(12000):
        #     image_path = os.path.join(root,f'{i+1}.jpg')
        #     self.data.append(image_path)
      

    def __getitem__(self, index):
        imgname,id = self.data[index]
        img = Image.open(imgname)
        raw_image = transforms.ToTensor()(img)
        img = self.transform(img)
        return img,  id ,raw_image

    def __len__(self):
        return len(self.data)



class VISDataSet(Dataset):
    def __init__(self, root,transform):
        self.root = root
        self.data = []
        self.transform = transform
        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")
        #types = ['*/*_N_N_0.jpg', '*/*_N_N_0.png']
        types = ['*/bs001*.jpg', '*/bs001*.png']
        #types = ['*/.jpg', '*/*_N_N_0.png']
        image_path_list= []
        for files in types:
            image_path_list.extend(glob(os.path.join(root, files)))
        #file_list_path = root  + '../file_path_list_AFLW2000_align.txt'
        #print(len(image_path_list))
        #count=0
        
        for ori_img_path in image_path_list:
            self.data.append(ori_img_path)
      

    def __getitem__(self, index):
        #index = 0
        imgname = self.data[index]
        img = Image.open(imgname)
        raw_image = transforms.ToTensor()(img)
        img = self.transform(img)
        return img, raw_image, imgname

    def __len__(self):
        return len(self.data)


def test():
    with open('/data/jdq/face_dataset/pg_data/landmark_list.txt','r') as f:
        lines = f.readlines()
    print(len(lines))
    count = 0
    for idx,line in enumerate(lines): 
        landmark_path=line.replace('\n', '')
        a = np.load(landmark_path,allow_pickle=True)
        if len(a.shape) == 0:
            count +=1
        print(f'{count}\{idx+1}')


class YTFDataset(Dataset):
    def __init__(self, root,  transform, resample = 5):
        self.root = root
        self.transform = transform

        self.data = []
        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")

        types = ['*/*/*.jpg', '*/*/*.png']
        image_path_list= []
        for files in types:
            image_path_list.extend(glob(os.path.join(root , files)))
        print(len(image_path_list))

        for imgname in image_path_list:
            self.data.append(imgname)
        self.data = self.data[::resample]
    def __getitem__(self, index):
        imgname = self.data[index]
        img = Image.open(imgname)
        img = self.transform(img)
        #print(type(img))
        #img = img[..., ::-1]
        return  img, imgname

    def __len__(self):
        return len(self.data)
        


        


class BosregDataSet(Dataset):
    def __init__(self, root,transform):
        self.root = root
        self.transform = transform

        self.data = []
        self.data_neutral = []
        self.labels = []
        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")

        types = ['*/*.jpg', '*/*.png']
        for files in types:
            self.data.extend(glob(os.path.join(root , files)))
        types = ['*/*_N_N_0.jpg', '*/*_N_N_0.png']
        for files in types:
            self.data_neutral.extend(glob(os.path.join(root , files)))
        self.data_neutral.sort()
       
        self.data.extend(self.data_neutral)

        for imgname in self.data:
            label = int(imgname.split('/')[-1].split('bs')[1].split('_')[0])
            self.labels.append(label)
        # print(self.labels)    
        # print(len(self.data))

    def __getitem__(self, index):
        imgname = self.data[index]
        label = self.labels[index]
        img = Image.open(imgname)
        img = self.transform(img)
        #print(type(img))
        #img = img[..., ::-1]  
        return  img, label, imgname

    def __len__(self):
        return len(self.data)
    
        


if __name__ == "__main__":
    import cv2
    print("For test.\n")
    

    transform = transforms.ToTensor()   
    dataset = BosregDataSet('/data/jdq/face_dataset/BosphorusDB/', transform=transform)

    ##########################################################################################
    #dataset = VGG2Dataset(transform=transform, indexfile="/data/jdq/face_dataset/pg_data/landmark_list.txt")
    #dataset = AFLW2000DataSet('/data/jdq/dbs/AFLW2000-3D/image_align/', transform=transform)
    # dataset = DataSet_300WLP('/data/jdq/face_dataset/300W_LP/image_align/', transform=transform)


    # ##########################################################################
    # trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=32)
    # print(dataset.__len__())
    # data = dataset.__getitem__(30)
    # img = data['image']
    # img=(img.data.cpu().numpy().transpose(1,2,0)[...,::-1]*255).astype('uint8')
    # img = np.array(img).copy() 
    # landmark=data['landmark'].T + 112
    # print(landmark.shape)
    # print(landmark)

    # point_size = 1
    # point_color = (0, 0, 255) 
    # thickness = 4 
    # for point in landmark:
    #     cv2.circle(img,tuple(point[:2]),point_size,point_color)
    # # print(img.shape)
    # # landmark=data['landmark'].data.cpu().numpy().astype('uint8')
    # # raw_image = Image.fromarray(img)

    # cv2.imwrite("test/2.png",img)


