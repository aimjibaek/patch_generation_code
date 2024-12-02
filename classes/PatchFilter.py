# one pixel filter와 full pixel filter의 속도 차이가 거의 없었기 때문에, 타일의 개수가 많은 것이 속도저하의 이유라고 생각
# 따라서 속도를 높이려면 타일의 개수를 줄이기 위해 stride를 키우고 cnt_threshold와 loop_cnt를 그에 따라 낮추면 된다 => 하지만 성능은 떨어진다

# np.mean 등을 torch.mean으로 바꿔서 속도향상 가능할 수도 있다

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt



class PatchFilter:
    def __init__(self, patch_size=224):
        self.patch_size = patch_size
        self.ds_patch_size = patch_size//4

        # downscaling for filter too much makes confusion between adipose tissue and white backgroud
        self.edge_tile_size = self.ds_patch_size // 8
        self.edge_tile_filter_loop_cnt = self.ds_patch_size // self.edge_tile_size

        self.total_tile_cnt = 28 # 8*4-4
        
    def is_tissue_lite(self, patch):
        return self.global_std_quick_filter(patch)

    def is_tissue_test(self, patch):
        patch = cv2.resize(patch, (self.ds_patch_size, self.ds_patch_size), interpolation=cv2.INTER_LINEAR)
        return self.global_filter(patch) and self.edge_tile_filter(patch)

    
    def is_tissue(self, patch):
        patch = cv2.resize(patch, (self.ds_patch_size, self.ds_patch_size), interpolation=cv2.INTER_LINEAR)
        return self.global_filter(patch) and self.edge_tile_filter(patch)
    
    #filter for testing
    def global_std_quick_filter(self, patch):
        std = np.std(patch, axis=(0,1,2))
        if std < 5:
            return False
        else:
            return True

    # filter 1
    def global_filter(self, patch):

        # 1. 어두운 노이즈, 흰색 배경 필터
        mean = np.mean(patch, axis=(0,1,2))
        mean = np.round(mean, 2)
        if mean < 50 or mean > 242: # 지방도 239까지도 가능 # slide masking이 사실 mean>242 와 다름없다(다만 샘플링을 안하고 픽셀하나로 바로 확인 가능)
            return False
        
        R, G, B = patch[:,:,0], patch[:,:,1], patch[:,:,2]
        R_mean, G_mean, B_mean = np.mean(R), np.mean(G), np.mean(B)
        R_std, G_std, B_std = np.round(np.std(R), 2), np.round(np.std(G), 2), np.round(np.std(B), 2)
        mean_RGB = (R_mean, G_mean, B_mean)
        std_RGB = (R_std, G_std, B_std)
        std_of_mean = np.round(np.std(mean_RGB),2) # 작다 => 무채색
        mean_of_std = np.round(np.mean(std_RGB),2) # 작다 => 픽셀값 균일

        # 2. 채도 작은 균일한 배경 필터
        if  std_of_mean < 10 and mean_of_std < 7.5: # 가끔 진한 샘플중에 10 이상이면서 std가 적은 경우가 있다
            # pale tissue, adipose의 경우 mean_of_std 8.x 까지도 가끔 존재 (7.x도 존재) # 단 7까지 내려가면 흰부분이 아주 많이 포함된 경우가 된다
            return False

        # 3. 채도 작은 어두운 마킹 필터
        if std_of_mean < 15 and mean < 90: #img_name = 'TCGA-FF-8047-01A-01-BS1' # dark_dense => mean 60 이하인 경우 있지만 채색이 있다(std_of_mean이 18~20이다)
            return False
        
        # 4. 채도가 큰 균일한 마킹 필터
        # # 피로 가득찬 부분도 std_of_mean 60을 넘는다 => mean_of_std이 15~50 다양함
        if std_of_mean > 50 and mean_of_std < 3:
            return False
        
        return True

    # filter 2
    def edge_tile_filter(self, patch): # tile total cnt = 28 

        filter_size = self.edge_tile_size
        stride = filter_size
        loop_cnt = self.edge_tile_filter_loop_cnt

        cnt_background_tile = 0
        cnt_tissue_like_tile = 0

        max_mean_of_std = 0
        max_R_minus_G = 0

        for i in range(loop_cnt):
            for j in range(loop_cnt):
                if i == 0 or i == loop_cnt - 1 or j == 0 or j == loop_cnt - 1: # 가장자리만 보겠다!
                    x_start = i * stride
                    x_end = i * stride + filter_size
                    y_start = j * stride
                    y_end = j * stride + filter_size

                    tile = patch[x_start:x_end,y_start:y_end, :]

                    R, G, B = tile[:,:,0], tile[:,:,1], tile[:,:,2]
                    R_mean, G_mean, B_mean = np.mean(R), np.mean(G), np.mean(B)
                    R_std, G_std, B_std = np.round(np.std(R), 2), np.round(np.std(G), 2), np.round(np.std(B), 2)
                    mean_RGB = (R_mean, G_mean, B_mean)
                    std_RGB = (R_std, G_std, B_std)
                    std_of_mean = np.round(np.std(mean_RGB),2) # 작다 => 무채색
                    mean_of_std = np.round(np.mean(std_RGB),2) # 작다 => 픽셀값 균일

                    #1-A. 채도 크고 균일한 마킹 필터링
                    if std_of_mean > 20 and mean_of_std < 2.5:
                        # std_of_mean > 20 : 채도 큼, mean_of_std < 4 : 색상이 균일함 => 마킹! / 지방의 경우는 mean_of_std > std_of_mean 
                        # mean_of_std < 4 => 균일한 조직?(pink) 배제됨 # 그렇다고 다빼면 마킹 잘 못배제함
                        # mean_of_std < 10 : 이러면 적혈구 지역도 더 많이 배제됨 (# 너무 blood만 차있는 구역도 샘플링에서 배제될 수 있음)
                        return False
                    
                    mean = np.mean(tile)
                    mean = np.round(mean, 2)
                    # 1-B. 밝지 않고 채도 작고 균일한 마킹 필터 
                    # mean 조건 => 지방조직이 배제되는 것을 방지
                    if mean < 170 and std_of_mean < 7 and mean_of_std < 7: # 채도 작고 균일한 편
                        return False

                    R_minus_G = np.round(R_mean - G_mean,2)
                    B_minus_G = np.round(B_mean - G_mean,2)
                    # 2. 배경이 많이 포함된 경우 필터링
                    if mean_of_std < 2.5 or R_minus_G < 1 or B_minus_G < 1: # RGB 값을 안쓰면 노이즈가 많은 배경에 대해서는 필터링이 안된다
                        cnt_background_tile += 1
                    if cnt_background_tile > 23: # threshold가 높아질수록 edge, noise를 잘잡고 지방을 배재하는 trade-off 관계 존재
                        return False
                    
                    
                    # 3. edge에서 샘플링된 조직 필터링  180: 덜 배제, 185: 더 배재됨 # 지방과 조직의 경계에 있는 경우 제외될 수 있음
                    if mean < 180 and mean_of_std > 30: # 지방도 해당하는 경우 있을 수 있기 때문에 tile의 개수에 대해서도 threshold 적용
                        cnt_tissue_like_tile += 1
                    if cnt_background_tile > 11 and cnt_tissue_like_tile > 7: # 좀더 많이 걸쳐있는 경우 # 지방에 걸쳐있는 조직 샘플링에 대한 trade-off 존재
                        return False
                    
                    # 4-A. 
                    if max_mean_of_std < mean_of_std: #그라데이션 마킹도 필터링
                        max_mean_of_std = mean_of_std
                    # 4-B. 
                    if max_R_minus_G < R_minus_G:
                        max_R_minus_G = R_minus_G
        # 4-A. 비균일한 부분이 없는 패치 필터링
        if max_mean_of_std < 7: # 아주 연한(pale) 조직이나 지방이라도 일부분은 10을 넘는 경우가 많음
            return False
        # 4-B. 염색에 의해 R값이 G값보다 커진 부분이 없는 패치 필터링
        if max_R_minus_G < 6: # 7 이상부터는 pale에 걸림
            return False

        return True

    def is_tissue_ash_nash_filter(self,patch):
        R, G, B = patch[:,:,0], patch[:,:,1], patch[:,:,2]
        R_mean, G_mean, B_mean = np.mean(R), np.mean(G), np.mean(B)
        R_std, G_std, B_std = np.round(np.std(R), 2), np.round(np.std(G), 2), np.round(np.std(B), 2)
        mean_RGB = (R_mean, G_mean, B_mean)
        std_RGB = (R_std, G_std, B_std)
        std_of_mean = np.round(np.std(mean_RGB),2) # 작다 => 무채색
        mean_of_std = np.round(np.mean(std_RGB),2) # 작다 => 픽셀값 균일

        return (std_of_mean >= 15) & (mean_of_std <= 55)
    def is_tissue_prev_filter(self,patch):
        patch = patch/255
        R, G, B = patch[:,:,0], patch[:,:,1], patch[:,:,2]
        R_mean, G_mean, B_mean = np.mean(R), np.mean(G), np.mean(B)
        R_std, G_std, B_std = np.round(np.std(R), 2), np.round(np.std(G), 2), np.round(np.std(B), 2)
        mean = (R_mean, G_mean, B_mean)
        std = (R_std, G_std, B_std)

        std_of_mean = np.round(np.std(mean),3)
        mean_of_std = np.round(np.mean(std),3)
        pixel_mean = np.mean(patch)
        return ((std_of_mean >= 0.03) & (mean_of_std >= 0.03) & (mean_of_std <= 0.25)) #and pixel_mean <= 0.80)
        # std_of_mean >= 7.65 and mean_of_std >= 7.65 and mean_of_std <= 0.25

    def deprecated_filters(self):
        pass
        #### GLOBAL FILTERS ####
        # 1.
        # (아래의 필터로 인해 std는 빼도 된다고 판단, std 하는 경우 시간도 더 오래걸린다) => deprecated
        # 아래 필터도 제외하는 경우 생긴다 (1.6인 경우도 있다 (pale))
        # if mean_of_std < 3: #배경에 있는 마킹 필터링
        #     return False
        # 2.
        # 아래와 같은 케이스 존재 => 취소... (TCGA-B0-5694 / adipose)
        # R_minus_G: -0.3763153698979522 / B_minus_G: -1.8517418686224403 
        # if R_mean - G_mean < 0.1 and B_mean - G_mean < 0.1: # g가 더 큰경우 존재함 (-1.8Rkwle)
        #     return False
        # 3.
        # std = np.std(patch, axis=(0,1,2))
        # if std < 5:
        #     return False
        #### EDGE FILTERS #### => 일단 다시 넣기로 함
        # 1. 
        # max_R_minus_G 안 쓸 경우 대안 / 하지만 backgrond count에서 어차피 R_minus_G를 써야 노이즈 더 깔끔하게 제거한다
        # 4-A. 옅은 검은색 마킹 필터링 / 다른 색도 잘됨
        # if max_mean_of_std < 7: # 아주 연한(pale) 조직이나 지방이라도 일부분은 10을 넘는 경우가 많음
        #     return False
        # 2. R_minus_G < -10
        # 다른 필터에서 걸러준다
        # 3.
        # # 특정 지방조직을 배제할 수 있기때문에 최소한의 조건만 남긴다 (조직과 지방의 경계 trade-off를 완화)
        # if cnt_background_tile > 14 and cnt_tissue_like_tile > 4: # 지방은 왠만해서는 1~2개에서 끝난다 # 
        #     return False
   

