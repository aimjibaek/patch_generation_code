


import numpy as np
np.random.seed(10)


import os



from matplotlib import pyplot as plt, patches
import openslide 
from PIL import Image
from tqdm.auto import tqdm
import cv2

from classes.PatchFilter import PatchFilter


import time

from datetime import timedelta

class RandomPatchMaker:
    def __init__(self, current, img_name, img_level, patches_num, patch_size, patch_samples_root, patch_coords_root, svs_path, save_coords=True):
        # slide img info
        self.img_level = img_level
        self.img_name = img_name
        self.svs_path = svs_path
        self.slide = openslide.OpenSlide(svs_path)
        self.svs_category_path = svs_path.split('/')[4]
        # print(self.svs_category_path)

        # patch info
        self.patch_size = patch_size
        self.patches_num = patches_num
        self.trial_limit_num = patches_num * 100 # 100의 경우 슬라이드의 1%라는 가정 혹은 무한루프 # 20정도면 배경 많은 슬라이드는 제외됨 # 100번당 한번뽑는 정도면 너무 샘플이 작아서 중복샘플이 너무 많이 될 수 있다. (ex. trial: 135479 for 500 / img_name:TCGA-HT-7483-01Z-00-DX1: 200번당 한번 뽑는 정도)
        self.patches = None
        self.patches_coords = None
        self.patch_samples_root = patch_samples_root
        self.patch_coords_root = patch_coords_root
        self.save_coords = save_coords
        self.patch_samples_path = f'{patch_samples_root}/{self.svs_category_path}/{img_name}'
        self.patch_coords_path = f'{patch_coords_root}/{self.svs_category_path}/{img_name}'

        self.PF = PatchFilter(patch_size=patch_size)
        self.folder_name = f'/Pathology_ImageNet/wlqor98/patch_sampling_coords_img/{current}/{self.svs_category_path}'


        # thumbnail img info
        self.thumbnail_level = 2 # 2000이하로 내려가면 더이상 downscale이 안된다. 그리고 level에 대응하는 downsample scale이 slide마다 다를 수 있다
        if len(self.slide.level_dimensions) <= 2: 
            self.thumbnail_level = -1
        self.thumbnail_downsample_scale = round(self.slide.level_dimensions[0][0] / self.slide.level_dimensions[self.thumbnail_level][0])

        self.img_shape = self.slide.level_dimensions[self.thumbnail_level] # for plotting patches in WSI
        self.img = self.slide.get_thumbnail(self.img_shape) 

        # masking filter test
        self.mean_masking_matrix = None
        self.slide_thumbnail_img = None
        self.masking_const = 16 # for thumbnail
        self.level_0_to_masking_const = self.thumbnail_downsample_scale * self.masking_const


    def is_40x(self):
        properties = self.slide.properties
        # Get the magnification for the base level (level 0)
        base_magnification = properties.get('aperio.AppMag', '0')
        base_magnification = int(float(base_magnification))
        print(f'base_magnification:{base_magnification}')

        return base_magnification == 40

    def random_patch_sampling(self, show_result=True, get_patches=False, use_filters=True):
        
        patches_num = self.patches_num
        img = self.img
        final_patch_size = self.patch_size
        width, height = self.slide.dimensions
        tds = self.thumbnail_downsample_scale

        self.patches = np.zeros((patches_num, final_patch_size, final_patch_size, 3))
        self.patches_coords = np.zeros((patches_num, 2)) # (x,y)

        
        i = 0 
        
        print(f'start {self.img_name}')
        pbar = tqdm(total = patches_num)
        trial = 0

        if show_result:
            print(f'show_result/save img with sample coordinates in {self.folder_name}')   

        if use_filters:  
            start_time = time.time()
            #self.set_masking_matrix()
            if show_result:            
                os.makedirs(self.folder_name, exist_ok=True)
                os.makedirs(self.patch_samples_path, exist_ok=True)
                os.makedirs(self.patch_coords_path, exist_ok=True)
                fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
                plt.title(self.img_name)
                ax.imshow(img)
            while i < patches_num:
                x = np.random.randint(0, width - self.patch_size)
                y = np.random.randint(0, height - self.patch_size)
                
                # x_masking_matrix = x // self.level_0_to_masking_const - 1 # 0부터 시작이니까 -1 해준다
                # y_masking_matrix = y // self.level_0_to_masking_const - 1
                
                # if self.mean_masking_matrix[y_masking_matrix][x_masking_matrix] == None:
                #     continue
                # if self.mean_masking_matrix[y_masking_matrix][x_masking_matrix] == 255:
                #     continue


                # 지금 원본 기준
                region = self.slide.read_region((x, y), self.img_level, (self.patch_size, self.patch_size))

                # Convert the region to a NumPy array
                patch = np.asarray(region)
                patch = patch[:, :, :3]
                
                trial += 1 # is_tissue 통과와 상관없이 count
                if not self.PF.is_tissue(patch):
                    # if show_result:
                    #     ax.add_patch(plt.Rectangle((x//tds, y//tds), self.patch_size//tds, self.patch_size//tds, edgecolor='r', facecolor='none',linewidth=1))
                    continue
                pbar.update(1)


                self.patches[i] = patch
                self.patches_coords[i] = np.array([x, y])
                i += 1
                if show_result:         
                    ax.add_patch(plt.Rectangle((x//tds, y//tds), self.patch_size//tds, self.patch_size//tds, edgecolor='g', facecolor='none',linewidth=1))
                    pass
                
                if trial == self.trial_limit_num: # ex) slide TCGA-DQ-5624-01Z-00-DX1
                    break
            end_time = time.time()  # End time for this iteration
            execution_time = end_time - start_time  # Calculate execution time
            execution_time = int(execution_time)

        else:
            while i < patches_num:
                x = np.random.randint(0, width - self.patch_size)
                y = np.random.randint(0, height - self.patch_size)
                # 지금 원본 기준
                region = self.slide.read_region((x, y), self.img_level, (self.patch_size, self.patch_size))

                # Convert the region to a NumPy array
                patch = np.asarray(region)
                patch = patch[:, :, :3]
                pbar.update(1)

                self.patches[i] = patch
                self.patches_coords[i] = np.array([x, y])
                i += 1
                if show_result:
                    if i == 1:
                        print('show_result')            
                    ax.add_patch(plt.Rectangle((x//tds, y//tds), self.patch_size//tds, self.patch_size//tds, edgecolor='g', facecolor='none'))

        if show_result:
            
            plt.savefig(f"{self.folder_name}/{self.img_name}.png", dpi=200, bbox_inches='tight')
            plt.show()
            plt.close()
        if trial < self.trial_limit_num:
            # Write the execution time of this iteration to the text file
            print('-------')
            print(f"img_name {self.img_name} / {str(timedelta(seconds=execution_time))} / trial: {trial} / sampled: {i}(patches_num)")
            if get_patches:
                return self.patches
            return True
        else:
            print(f"can't sampling / trial: {trial} for {patches_num} / sampled:{i} /img_name:{self.img_name}")
            return False

     
    def set_masking_matrix(self):  
        # Load image
        slide_thumbnail = self.slide.get_thumbnail((self.img_shape[0]//self.masking_const, self.img_shape[1]//self.masking_const)) # Adjust the size as needed     
        self.slide_thumbnail = slide_thumbnail

        img = np.array(slide_thumbnail)
        self.slide_thumbnail_img = img
    
        img = cv2.fastNlMeansDenoisingColored(img,None,10,10,5,21)
        mean_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mean_masking_matrix = cv2.threshold(mean_img, 240, 255, cv2.THRESH_BINARY)
        
        self.mean_masking_matrix = mean_masking_matrix        
        plt.imshow(self.mean_masking_matrix)
        plt.title(f'{self.img_name} / masking matrix')
        plt.show()


        

    def show_patches(self, total_count): 
        for i in range(total_count):
            patch = self.patches[i]
            plt.imshow(patch.astype('uint8'))
            plt.title(self.img_name)
            plt.show()

    def save_patches(self, file_format):

        name = self.img_name
        patch_samples_path = self.patch_samples_path
        patch_coords_path = self.patch_coords_path

        if os.path.isdir(patch_samples_path) == False :
            os.makedirs(patch_samples_path)
        if os.path.isdir(patch_coords_path) == False :
            os.makedirs(patch_coords_path)

        if file_format == 'numpy':
            for i in range(self.patches_num):
                data = self.patches[i].astype(np.uint8)
                np.save(os.path.join(patch_samples_path, f'{name}-{i}.npy'), data)
        elif file_format == 'binary':
            for i in range(self.patches_num):
                data = self.patches[i].astype(np.uint8)
                with open(os.path.join(patch_samples_path, f'{name}-{i}.bin'), 'wb') as f:
                    f.write(data.tobytes())

        if self.save_coords:
            coords = self.patches_coords.astype(np.int32) # int16은 - ~ +32767 까지 표현가능 => int32를 써야함!
            #print(coords.dtype)
            np.save(os.path.join(patch_coords_path, f'{name}-coords.npy'), coords) 
    
    def load_patches_example(self, total_count=4):
        target_dir = self.patch_samples_path
        candidates = os.listdir(target_dir)
        for i, candidate in enumerate(candidates):
            if i == total_count:
                break
            if candidate[-3:] == 'bin':
                with open(f'{target_dir}/{candidate}', 'rb') as f:
                    loaded_data = f.read()
                    loaded_img = np.frombuffer(loaded_data, dtype=np.uint8).reshape(224, 224, 3)
                    print(type(loaded_img))
                    print(loaded_img.dtype)
                    plt.title(candidate)
                    plt.imshow(loaded_img)
                    plt.show()
            

    def load_all_coords_on_slide(self):

        target_dir = self.patch_coords_path
        candidates = os.listdir(target_dir)
        target_file = candidates[0] # 한파일에 모든 coords 저장
        tds = self.thumbnail_downsample_scale
        if not target_file[-3:] == 'npy':
            print('file format error')
        
        coords = np.load(f'{target_dir}/{target_file}')
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.title(self.img_name)
        ax.imshow(self.img)

        for i, coord in enumerate(coords):
            x = coord[0]
            y = coord[1]
            ax.add_patch(plt.Rectangle((x//tds, y//tds), self.patch_size//tds, self.patch_size//tds, edgecolor='g', facecolor='none'))
        plt.show()

