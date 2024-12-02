import os
from tqdm.auto import tqdm
import openslide
import matplotlib.pyplot as plt
import numpy as np




def get_svs_img_name(svs_path):
    return svs_path.split('/')[-1][:23] # 12까지 했을때 가끔씩 똑같은 이름의 슬라이드 존재

def get_svs_path_list(is_for_guri_kirc_tcga_only=True):
    svs_root = '/Pathology_ImageNet/TCGA_data'
    if is_for_guri_kirc_tcga_only:
        svs_root = '/raid/Datasets/Guri_hospital/KIRC_TCGA_Dx_images'

    if is_for_guri_kirc_tcga_only:
        end_count = 519 
    else:
        end_count = 18292 # total pretraining slide count
    pbar = tqdm(total = end_count)
    count = 0
    svs_path_list = []
    print('lets start')
    for foldername, subfolders, filenames in os.walk(svs_root):
        for filename in filenames:
            if filename.endswith('.svs'):
                count += 1
                pbar.update(1)
                full_path = os.path.join(foldername, filename)
                svs_path_list.append(full_path)
                if count == end_count:
                    break
        if count == end_count:
            break

    print(f'len(svs_path_list): {count}')

    # remove slides with errors
    if len(svs_path_list) > 10000 and not is_for_guri_kirc_tcga_only: 
        svs_path_list.remove('/Pathology_ImageNet/TCGA_data/HDD4/BRCA_TCGA_images/TCGA_breast_virture_over_1GB/TCGA-BH-A0AW-01Z-00-DX1.9D50A0D2-B103-411C-831E-8520C3D50173.svs')
        svs_path_list.remove('/Pathology_ImageNet/TCGA_data/HDD4/BRCA_TCGA_images/TCGA_breast_virture_over_1GB/TCGA-BH-A0B3-01Z-00-DX1.90CB0ED5-FBB7-4ABF-93A0-DD88D60D3D55.svs')
        svs_path_list.remove('/Pathology_ImageNet/TCGA_data/HDD3/ACC_TCGA_Dx_tissue_images/92508ebe-31a3-4150-a525-72e2d7245933/TCGA-OU-A5PI-01Z-00-DX5.8D95003F-113E-42A0-BACC-06F42528D4B6.svs')
        
    
    return svs_path_list



# ipynb 파일에서 아래의 주석처리된 코드로 해당 함수를 실행
# from modules.custom_openslide import show_svs_in_grid, get_svs_path_list
# svs_path_list = get_svs_path_list(is_for_guri_kirc_tcga_only=False)
# show_svs_in_grid(svs_path_list, grid_size=(5, 4), downsample_factor=32, start_index=0, cnt=20)
# show_svs_in_grid(svs_path_list, grid_size=(5, 4), downsample_factor=32, start_index=20, cnt=20)

def show_svs_in_grid(svs_path_list, grid_size=(5, 4), downsample_factor=32, start_index=0, cnt=None):
    if start_index >= len(svs_path_list):
        print('out of index')
        return

    # If end_index is not provided or beyond the list length, adjust it
    if cnt is None or start_index + cnt > len(svs_path_list):
        end_index = len(svs_path_list)

    # Initialize the figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))

    for grid_idx, svs_idx in enumerate(range(start_index, start_index + cnt)):
        if grid_idx >= grid_size[0] * grid_size[1]:
            break

        svs_path = svs_path_list[svs_idx]
        try:
            slide = openslide.OpenSlide(svs_path)

            # Get the thumbnail using the constant downsample factor
            thumbnail_dimensions = (slide.dimensions[0] // downsample_factor, slide.dimensions[1] // downsample_factor)
            thumbnail = slide.get_thumbnail(thumbnail_dimensions)

            # Get the current axis for the image
            ax = axes[grid_idx // grid_size[1], grid_idx % grid_size[1]]

            ax.imshow(thumbnail)
            ax.axis('off')  # Turn off axis
            ax.set_title(f"Image {svs_idx}")
        except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
            print('exception: ', e)
            print(f'image {svs_idx} error')

    # Remove any unused subplots
    for i in range(grid_idx + 1, grid_size[0] * grid_size[1]):
        axes.flatten()[i].remove()

    plt.tight_layout()
    plt.show()