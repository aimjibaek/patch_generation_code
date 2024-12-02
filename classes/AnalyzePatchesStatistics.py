import os
import matplotlib.pyplot as plt
import numpy as np


class AnalyzePatchesStatistics:
    def __init__(self):
        pass
        # this class consisted of many functions
     
    def get_patches_statistics(self, patch):
            R = patch[:,:,0]
            G = patch[:,:,1]
            B = patch[:,:,2]

            R_mean, G_mean, B_mean = np.mean(R), np.mean(G), np.mean(B)
            R_std, G_std, B_std = np.std(R), np.std(G), np.std(B)

            # round => 계산 시간 감축
            mean = (np.round(R_mean,2), np.round(G_mean,2), np.round(B_mean,2))
            std = (np.round(R_std,2), np.round(G_std,2), np.round(B_std,2))
            std_of_mean = np.round(np.std(mean),3) # 작다 => R,G,B 값 모두 비슷 => 배경
            mean_of_std = np.round(np.mean(std),3) # 작다 => R,G,B의 각 std이 작다 => 배경

            std = np.std(patch, axis=(0,1,2)) 
            mean = np.mean(patch, axis=(0,1,2)) 

            vertex_4_pixels_patch = np.array([
                [patch[0][0], patch[0][-1]], [patch[-1][0], patch[-1][-1]]
            ])
            R_v = vertex_4_pixels_patch[:,:,0]
            G_v = vertex_4_pixels_patch[:,:,1]
            B_v = vertex_4_pixels_patch[:,:,2]
            
            R_mean_v, G_mean_v, B_mean_v = np.mean(R_v), np.mean(G_v), np.mean(B_v)
            R_std_v, G_std_v, B_std_v = np.std(R_v), np.std(G_v), np.std(B_v)

            # round => 계산 시간 감축
            mean_v = (np.round(R_mean_v,2), np.round(G_mean_v,2), np.round(B_mean_v,2))
            std_v = (np.round(R_std_v,2), np.round(G_std_v,2), np.round(B_std_v,2))
            std_of_mean_vtx = np.round(np.std(mean_v),3) # 작다 => R,G,B 값 모두 비슷 => 배경
            mean_of_std_vtx = np.round(np.mean(std_v),3) # 작다 => R,G,B의 각 std이 작다 => 배경

            R_minus_G_vtx = R_mean_v - G_mean_v
            B_minus_G_vtx = B_mean_v - G_mean_v

            return R_mean, G_mean, B_mean, R_std, G_std, B_std, std_of_mean, mean_of_std, std, mean, std_of_mean_vtx, mean_of_std_vtx, R_minus_G_vtx, B_minus_G_vtx

    def patches_statistics_histogram(self, patches, is_use_quad_split=True):

        std_of_means, mean_of_stds = [], []


        R_minus_G_s, B_minus_G_s = [], []
        
        for patch in patches:
            R_mean, G_mean, B_mean, R_std, G_std, B_std, std_of_mean, mean_of_std, std, mean, std_of_mean_vtx, mean_of_std_vtx, R_minus_G_vtx, B_minus_G_vtx = self.get_patches_statistics(patch)

            h, w, _ = patch.shape
            quad_list = [patch[:h//2, :w//2], patch[:h//2, w//2:], patch[h//2:, :w//2], patch[h//2:, w//2:]]

    
            mean_std_of_mean = 0
            mean_mean_of_std = 0
            for quad in quad_list:
                _, _, _, _, _, _, std_of_mean, mean_of_std, _, _, _, _, _, _ = self.get_patches_statistics(quad)
                mean_std_of_mean += std_of_mean
                mean_mean_of_std += mean_of_std
            mean_std_of_mean = mean_std_of_mean/4
            mean_mean_of_std = mean_mean_of_std/4

            
            std_of_means.append(mean_std_of_mean)
            mean_of_stds.append(mean_mean_of_std)

            R_minus_G_s.append(R_mean - G_mean)
            B_minus_G_s.append(B_mean - G_mean)

                

        
        self._plot_histogram(std_of_means, 'Std Dev of Means', 'm')
        self._plot_histogram(mean_of_stds, 'Mean of Std Devs', 'c')

        self._plot_histogram(R_minus_G_s, 'Red minus Green', 'r')
        self._plot_histogram(B_minus_G_s, 'Blue minus Green', 'b')


    def _plot_histogram(self, data, title, color):
        plt.hist(data, bins=50, color=color, alpha=0.7)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

    def show_patch_statistics(self, patch):
        histograms = []
        for i in range(3):
            histogram, _ = np.histogram(patch[:,:,i], bins=256, range=(0, 256))
            histograms.append(histogram)

        # 히스토그램 그리기
        colors = ['r', 'g', 'b']
        labels = ["Red", "Green", "Blue"]
        for i, color in enumerate(colors):
            plt.plot(histograms[i], color=color, label=labels[i])

        plt.legend()
        plt.title(" Pixel Distribution")
        plt.xlabel('Pixel Value')
        plt.ylabel('Number of pixels')
        plt.show()
        
        R_mean, G_mean, B_mean, R_std, G_std, B_std, std_of_mean, mean_of_std, std, mean, std_of_mean_vtx, mean_of_std_vtx, R_minus_G_vtx, B_minus_G_vtx  = self.get_patches_statistics(patch)

        print(f"Mean Values:\nR: {R_mean:.2f}\nG: {G_mean:.2f}\nB: {B_mean:.2f}")
        print(f"\nStandard Deviation Values:\nR: {R_std:.2f}\nG: {G_std:.2f}\nB: {B_std:.2f}")
        print(f'std_of_mean: {std_of_mean:.2f} / mean_of_std: {mean_of_std:.2f}')
        print(f'std: {std:.2f} / mean: {mean:.2f}')
        #print(f"std_of_mean_vtx: {std_of_mean_vtx:.2f} / mean_of_std_vtx: {mean_of_std_vtx:.2f}")

        print(f"R_minus_G: {R_mean - G_mean} / B_minus_G: {B_mean - G_mean} / R_minus_B: {R_mean - B_mean}")

        return R_mean < B_mean