import numpy as np
import matplotlib.pyplot as plt
import cv2
from noise import pnoise2
import random
from skimage.measure import label, regionprops
import h5py

def generate_perlin_noise_2d(shape, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    if seed is None:
        seed = np.random.randint(0, 100)
    
    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            noise[i][j] = pnoise2(i / scale, 
                                  j / scale, 
                                  octaves=octaves, 
                                  persistence=persistence, 
                                  lacunarity=lacunarity, 
                                  repeatx=shape[0], 
                                  repeaty=shape[1], 
                                  base=seed)
    return noise

num_maps = 10000
image_shape = (512, 512)    
save_path = "dataset_3/testtest.h5"
with h5py.File(save_path, 'w') as h5file:
    # Create datasets for images and ground truth masks
    images_dataset = h5file.create_dataset("images", (num_maps, *image_shape), dtype='float32')
    masks_dataset = h5file.create_dataset("ground_truth_masks", (num_maps, *image_shape), dtype='float32')

    # Create synthetic difference maps
    for i in range(num_maps):
        #512x512 image with random values between -1 and 1,
        image = (np.random.rand(512, 512) * 2 - random.uniform(0.85, 1.15))
        image = cv2.GaussianBlur(image, (7,7), 0)
        
        #generate a perlin noise map
        perlin_noise_map = generate_perlin_noise_2d((512, 512), scale=random.randint(10,60), octaves=4, persistence=0.5, lacunarity=2.0, seed=None)
        #scale intensity of perlin noise map
        perlin_noise_map = 2 * (perlin_noise_map - np.min(perlin_noise_map)) / (np.max(perlin_noise_map) - np.min(perlin_noise_map)) - 1
        #set values below 0.5 to -1 and above 0.5 to 1 and between 0.5 and -0.5 to 0
        perlin_noise_map[perlin_noise_map < -0.5] = -1
        perlin_noise_map[perlin_noise_map > 0.5] = 1
        perlin_noise_map[(perlin_noise_map >= -0.5) & (perlin_noise_map <= 0.5)] = 0

        labeled_map = label(perlin_noise_map != 0)
        regions = regionprops(labeled_map)
        #set 50-90% of the regions with value -1 and 1 to 0
        for region in regions:
            if random.random() < random.uniform(0.8,1):
                perlin_noise_map[region.coords[:,0], region.coords[:,1]] = 0
        
        ground_truth_mask = perlin_noise_map.copy()
        #remove small regions
        for region in regions:
            if region.area < 20:
                ground_truth_mask[region.coords[:,0], region.coords[:,1]] = 0

        mask_neg = (ground_truth_mask == -1).astype(np.uint8)
        mask_pos = (ground_truth_mask == 1).astype(np.uint8)

        # Dilate the binary masks
        dilated_neg = cv2.dilate(mask_neg, np.ones((3, 3), np.uint8), iterations=1)
        dilated_pos = cv2.dilate(mask_pos, np.ones((3, 3), np.uint8), iterations=1)

        # Combine the dilated masks with the original output
        ground_truth_mask[dilated_neg == 1] = -1
        ground_truth_mask[dilated_pos == 1] = 1

        seed = np.random.randint(0, 100)
        perlin_noise_map_background = generate_perlin_noise_2d((512, 512), scale=random.uniform(100,400), octaves=int(random.uniform(4,8)), persistence=0.5, lacunarity=2.0, seed=seed)

        perlin_noise_map = cv2.GaussianBlur(perlin_noise_map, (31,31), 0)

        image_scale = random.uniform(0.5, 1)

        val = random.random()
        if val < 0.33:
            scale_background = random.uniform(0,0.8)
        elif val < 0.66:
            scale_background = random.uniform(0,0.3)
        else:
            scale_background = random.uniform(0,0.1)
        #print(scale_background)
        if scale_background > 0.4:
            scale_target = random.uniform(1,1.1)
        elif scale_background > 0.25:
            scale_target = random.uniform(0.9,1.1)
        else:
            scale_target = random.uniform(0.65,1)
        #print(scale_target)
        image = image*image_scale + perlin_noise_map_background*scale_background + perlin_noise_map*scale_target

        #calculate mean intensity of "image" in the coordinates of -1 and 1 regions of perlin_noise_map

    
        #normalize image between -1 and 1
        image = 2 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1

        hist, bins = np.histogram(image.flatten(), bins=100, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if True:
            plt.figure(figsize=(15,5))
            plt.subplot(1,3,1)
            plt.imshow(image, cmap="coolwarm", vmin=-1, vmax=1)
            plt.title("Image")
            plt.axis("off")
            plt.subplot(1,3,2)
            plt.imshow(ground_truth_mask, cmap="coolwarm", vmin=-1, vmax=1)
            plt.title("Ground Truth Mask")
            plt.axis("off")
            plt.subplot(1,3,3)
            plt.plot(bin_centers, hist)
            plt.title("Histogram")
            plt.show()
        
        #print(scale_target)
        images_dataset[i] = image
        masks_dataset[i] = ground_truth_mask

        #save image and ground truth mask as npz
        #np.savez(f"dataset_3/synthetic_diff_map_{i}.npz", image=image, ground_truth_mask=ground_truth_mask)
        print(f"{i}/{num_maps} done.") 