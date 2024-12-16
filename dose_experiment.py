import cv2
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from models import UNet
import torch
from skimage.measure import label, regionprops

sample_paths = ["data/Dose/spot_5",
                "data/Dose/spot_6",
                "data/Dose/spot_7",
                "data/Dose/spot_8",
                "data/Dose/spot_9",]

UNet = UNet(1,1)
UNet.load_state_dict(torch.load("model_10k.pth"))
UNet.eval()


movement = {}
for sample_path in sample_paths:
    #
    movement[f"{sample_path}"] = []

    data_dir = f"{sample_path}"
    # Find all unique time points in data_dir, the file names are in the format "time_first_20.tif"
    time_points = set()
    for file in os.listdir(data_dir):
        # if file name follows the format "time_first_20.tif" or "time_last_20.tif"
        if file.endswith("_first_20.tif") or file.endswith("_last_20.tif"):
            time = file.split("_")[0]
            time_points.add(time)
    time_points = list(time_points)
    time_points.sort()
    print(f"Time points for {data_dir}: {time_points}")

    areas = []
    for time in time_points:
        im_before = f"{time}_first_20.tif"
        im_after = f"{time}_last_20.tif"

        im_before_path = os.path.join(data_dir, im_before)
        im_after_path = os.path.join(data_dir, im_after)

        # Load the images
        image1 = cv2.imread(im_before_path, cv2.IMREAD_GRAYSCALE).astype(float)
        image2 = cv2.imread(im_after_path, cv2.IMREAD_GRAYSCALE).astype(float)

        # Resize the images to 512x512
        image1 = cv2.resize(image1, (512, 512))
        image2 = cv2.resize(image2, (512, 512))

        # Filter the images with a Gaussian filter
        image1 = cv2.GaussianBlur(image1, (7, 7), 0)
        image2 = cv2.GaussianBlur(image2, (7, 7), 0)

        # Normalize the images
        image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
        image2 = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))

        # Get the difference map including negative values
        diff = image2 - image1

        # Fit Gaussian to the histogram of the difference map
        diff_flat = diff.flatten()
        hist, bins = np.histogram(diff_flat, bins=100, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        def gaussian(x, mean, std, amplitude):
            return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

        # Fit the Gaussian
        popt, _ = curve_fit(gaussian, bin_centers, hist)

        # Extract fitted parameters
        mean, std, amplitude = popt

        if mean < 0:
            im_old = image1
            image1 = np.clip(image1 - abs(mean), 0, 1)
        elif mean > 0:
            im_old = image2
            image2 = np.clip(image2 - abs(mean), 0, 1)
        else:
            print("Images are equally bright")

        diff = image2 - image1

        diff_normalized = 2 * (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) - 1
        diff_normalized = torch.tensor(diff_normalized).unsqueeze(0).unsqueeze(0).float()
        output = UNet(diff_normalized)
        output = output.squeeze().detach().numpy()
        # Set values below -0.5 to -1 and above 0.5 to 1 and between 0.5 and -0.5 to 0
        output[output < -0.5] = -1
        output[output > 0.5] = 1
        output[(output >= -0.5) & (output <= 0.5)] = 0

        # Create binary masks
        mask_neg = (output == -1).astype(np.uint8)
        mask_pos = (output == 1).astype(np.uint8)

        # Dilate the binary masks
        #dilated_neg = cv2.dilate(mask_neg, np.ones((5, 5), np.uint8), iterations=1)
        #dilated_pos = cv2.dilate(mask_pos, np.ones((5, 5), np.uint8), iterations=1)

        # Combine the dilated masks with the original output
        # output[dilated_neg == 1] = -1
        # output[dilated_pos == 1] = 1

        # Remove small regions
        labeled_map = label(output != 0)
        regions = regionprops(labeled_map)
        for region in regions:
            if region.area < 50:
                output[region.coords[:, 0], region.coords[:, 1]] = 0

        diff_normalized = diff_normalized.squeeze().numpy()
        
        if False:
            plt.figure(figsize=(10, 5))
            plt.suptitle(f"{sample_path} - {time}")
            plt.subplot(1, 3, 1)
            plt.imshow(im_old, cmap='gray')
            plt.title("Original Image")
            plt.subplot(1, 3, 2)
            plt.imshow(diff_normalized, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Difference Map")
            plt.subplot(1, 3, 3)
            plt.imshow(output, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Output")
            plt.show()

        # Calculate the mean and total area of positive regions for each sample and gas environment
        labeled_map_pos = label(output == 1)
        regions_pos = regionprops(labeled_map_pos)
        tot_area_pos = sum(region.area for region in regions_pos)

        labeled_map_neg = label(output == -1)
        regions_neg = regionprops(labeled_map_neg)
        tot_area_neg = sum(region.area for region in regions_neg)

        tot_area = tot_area_pos + tot_area_neg

        areas.append(tot_area)
        movement[f"{sample_path}"] = areas

# Define the exponential decay function
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# Corresponding currents for each dose
currents = [280, 144, 69, 35, 20]

# Collecting mean movements for each sample path
mean_movements = []
for sample_path in sample_paths:
    print(f"{sample_path}: {movement[f'{sample_path}']}")
    mean_movement = np.mean(movement[f"{sample_path}"])
    mean_movements.append(mean_movement)
    plt.plot(currents[sample_paths.index(sample_path)], mean_movement, "x", color="red", markersize=8, markeredgewidth=2)

initial_guess = [max(mean_movements), 0.01, min(mean_movements)]
popt, _ = curve_fit(exp_decay, currents, mean_movements, p0=initial_guess, maxfev=10000)
x_fit = np.linspace(min(currents), max(currents), 100)
y_fit = exp_decay(x_fit, *popt)
plt.plot(x_fit, y_fit, "k--")
plt.xlabel("Current (pA)")
plt.ylabel("Mean Movement (pixels)")
plt.gca().invert_xaxis()
plt.title("Movement vs. Beam Current")
plt.savefig("dose_experiment.png")
plt.show()

print(f"Mean movements: {mean_movements}")

# save the fitted curve parameters
with open("dose_experiment.txt", "w") as f:
    f.write(f"Mean movements: {mean_movements}\n")
    f.write(f"Curve parameters: {popt}")

print(f" Function: {popt[0]} * exp(-{popt[1]} * x) + {popt[2]}")