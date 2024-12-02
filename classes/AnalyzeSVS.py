import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, patches
from modules.custom_openslide import get_svs_img_name
from classes.RandomPatchMaker import PatchFilter 
import numpy as np
import openslide
import os

np.random.seed(1)





class AnalyzeSVS:
    def __init__(self, svs_path, filter_version='full'):
        self.filter_version = filter_version
        self.img_name = get_svs_img_name(svs_path)
        self.svs_path = svs_path
        self.slide = openslide.OpenSlide(svs_path)
        self.thumbnail_resolution_level = 2
        self.thumbnail_downsampled_scale = round(self.slide.level_dimensions[0][0] / self.slide.level_dimensions[self.thumbnail_resolution_level][0])
        print(f'thumbnail_downsampled_scale: {self.thumbnail_downsampled_scale} / thumbnail level_dimension:{self.slide.level_dimensions[self.thumbnail_resolution_level]}')
        self.img_shape = self.slide.level_dimensions[self.thumbnail_resolution_level]
        self.slide_thumbnail = self.slide.get_thumbnail(self.img_shape) # Adjust the size as needed # self.slide_thumbnail_masking_matrix 보다 16배 더 큰 사이즈 / 하지만 원본에 비에서는 downsampled_scale만큼 작아진 상태
        self.width, self.height = self.slide.dimensions
    
    def show_svs_thumbnail(self):
        plt.imshow(self.slide_thumbnail)
        plt.title(f'{self.img_name}')
        plt.show()
    
        
    def show_and_get_region_in_svs(self, coord=(0,0), region_size=1000, resolution=1, inner_sample_cnt=0, use_true_scale_for_region_size=False):
        #neon_blue = (0.27, 0.09, 1.0)
        #neon_green = (0.20, 1.0, 0.20)
        neon = (0.20, 1.0, 0.20)

        tds = self.thumbnail_downsampled_scale
    
        coord_x = coord[0]
        coord_y = coord[1]

        fig, ax = plt.subplots(1, figsize=(10, 6), dpi=100)
        ax.imshow(self.slide_thumbnail)

        if use_true_scale_for_region_size:
            region_size = region_size // tds

        rect = patches.Rectangle((coord_x, coord_y), region_size, region_size, linewidth=1, edgecolor=neon, facecolor='none')
        ax.add_patch(rect)
        plt.title(f'{self.img_name} / x: {coord_x}~{coord_x+region_size}, y: {coord_y}~{coord_y+region_size}')
        
        folder_name = "./filter_test_patch_region"
        os.makedirs(folder_name, exist_ok=True)
        plt.savefig(f"./{folder_name}/{self.img_name}_{coord}.png", dpi=200, bbox_inches='tight')
        
        plt.show()

        x = coord_x * tds
        y = coord_y * tds

        if use_true_scale_for_region_size:
            region = self.slide.read_region((x, y), resolution, (region_size * tds, region_size * tds))
        else:
            ds = round(self.slide.level_dimensions[0][0] / self.slide.level_dimensions[resolution][0])
            correction_scale = tds // ds
            region = self.slide.read_region((x, y), resolution, (region_size * correction_scale, region_size * correction_scale))

        region_np = np.array(region)[:, :, :3]  # Convert PIL image to numpy array and remove alpha channel

        # Extract and show random patches
        patch_size = 224

        # Ensure that the region is at least the size of the patch (or larger)
        if region_np.shape[0] < patch_size or region_np.shape[1] < patch_size:
            print("Region size is smaller than the desired patch size.")
            return


        fig, ax = plt.subplots(1, figsize=(9, 9), dpi=100)
        ax.imshow(region)

        if inner_sample_cnt != 0:
            PF = PatchFilter(patch_size=224)
            patches_list = []
            i = 0
            trial = 0
            while inner_sample_cnt > i:
                # Randomly determine the top-left corner coordinates of the patch within the region
                start_x = np.random.randint(0, region_np.shape[1] - patch_size)
                start_y = np.random.randint(0, region_np.shape[0] - patch_size)
                
                # Extract the patch and append to patches list
                patch = region_np[start_y:start_y+patch_size, start_x:start_x+patch_size]


                trial += 1
                if trial == 15000:
                    print(f"can't sampling / trial: {trial}")
                    return region_np, patches_list
                if self.filter_version =='lite':
                    if not PF.is_tissue_lite(patch):
                        continue
                elif self.filter_version == 'full':
                    if not PF.is_tissue(patch):
                        continue
                elif self.filter_version == 'test':
                    if not PF.is_tissue_test(patch):
                        continue

 
                patches_list.append(patch)


                # Draw a rectangle on the main image to show the patch region
                rect_patch = patches.Rectangle((start_x, start_y), patch_size, patch_size, linewidth=1, edgecolor=neon, facecolor='none')
                ax.add_patch(rect_patch)
                
                # Add patch number to the region
                patch_center = (start_x + patch_size/2, start_y + patch_size/2)
                ax.text(patch_center[0], patch_center[1], str(i+1), color=neon, ha='center', va='center')
                i += 1
            
            plt.savefig(f"./{folder_name}/{self.img_name}_{coord}_region.png", dpi=200, bbox_inches='tight')
            plt.show()

            # Display each extracted patch
            for idx, patch in enumerate(patches_list):
                plt.figure(figsize=(5, 5))
                plt.imshow(patch)
                plt.title(f'Patch {idx + 1}')
                plt.show()

            return region_np, patches_list

        return region_np

    def get_filter_info(self, coord_clear=(0,0), coord_noise=(0,0), patch_size=224):

        img_noise = self.show_and_get_region_in_svs(coord_clear, patch_size)
        #plt.imshow(region_noise)
        #plt.show()
        img_clear = self.show_and_get_region_in_svs(coord_noise, patch_size)
        #plt.imshow(region_clear)
        #plt.show()

        filter =  img_clear - img_noise

        img = filter
        print(img/256)
        histograms = []
        for i in range(3):
            histogram, _ = np.histogram(img[:,:,i], bins=256, range=(0, 256))
            histograms.append(histogram)

        # 히스토그램 그리기
        colors = ['r', 'g', 'b']
        labels = ["Red", "Green", "Blue"]
        for i, color in enumerate(colors):
            plt.plot(histograms[i], color=color, label=labels[i])

        plt.legend()
        plt.title(self.img_name + " Pixel Distribution")
        plt.xlabel('Intensity')
        plt.ylabel('Number of pixels')
        plt.show()


