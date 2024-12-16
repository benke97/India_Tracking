import cv2
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from models import UNet
import matplotlib.animation as animation
import torch
from skimage.measure import label, regionprops
import csv


sample_paths = ["data/CPInOZrO",
                "data/InOYSZ",
                "data/InOmZrO",
                "data/FSPInOZrO",
                "data/InOZrOSBET20",
                "data/InOtZrO",
                "data/PdInOZrO",
                "data/InOTiO",]



gas_environments = ["Air",
                    "N2",
                    "CO2H2",
                    "N2H2"]

UNet = UNet(1,1)
UNet.load_state_dict(torch.load("model_10k.pth"))
UNet.eval()


movement = {}
positive_areas = {}
negative_areas = {}
pos_region_sizes = {}
neg_region_sizes = {}

for sample_path in sample_paths:
    for gas_environment in gas_environments:
        #
        #if gas exists for given sample
        if not os.path.exists(f"{sample_path}/{gas_environment}"):
            continue
        movement[f"{sample_path}_{gas_environment}"] = []

        data_dir = f"{sample_path}/{gas_environment}"
        # Find all unique time points in data_dir, the file names are in the format "time_first_20.tif"
        time_points = set()
        for file in os.listdir(data_dir):
            # if file name follows the format "time_first_20.tif" or "time_last_20.tif"
            if file.endswith("_first_20.tif") or file.endswith("_last_20.tif") or file.endswith("_first_5.tif") or file.endswith("_last_5.tif"):
                time = file.split("_")[0]
                time_points.add(time)
        time_points = list(time_points)
        time_points.sort()
        print(f"Time points for {data_dir}: {time_points}")
        areas = []
        pos_areas = []
        neg_areas = []
        pos_areas_sizes = []
        neg_areas_sizes = []
        for time in time_points:
            if sample_path == "data/InOmZrO":
                im_before = f"{time}_first_5.tif"
                im_after = f"{time}_last_5.tif"
            else:
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

            # Remove small regions
            labeled_map = label(output != 0)
            regions = regionprops(labeled_map)
            for region in regions:
                if region.area < 50:
                    output[region.coords[:, 0], region.coords[:, 1]] = 0

            diff_normalized = diff_normalized.squeeze().numpy()
            
            if False:
                plt.figure(figsize=(10, 5))
                plt.suptitle(f"{sample_path} - {gas_environment} - {time}")
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
            if sample_path == "data/InOZrOSBET20afsasdf":
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                #time as title
                fig.suptitle(f"{time}")
                # Animation subplot
                ax1.axis('off')
                img_display = ax1.imshow(image1, cmap='gray')

                # Difference image subplot
                ax2.imshow(diff_normalized, cmap='coolwarm', vmin=-1, vmax=1)
                ax2.set_title("Difference Map")

                ax3.imshow(output, cmap='coolwarm', vmin=-1, vmax=1)
                ax3.imshow(image1, cmap='gray', alpha=0.5)
                ax3.set_title("Output")

                def update(frame):
                    if frame % 2 == 0:
                        img_display.set_data(image1)
                    else:
                        img_display.set_data(image2)
                    return [img_display]

                ani = animation.FuncAnimation(fig, update, frames=np.arange(20), interval=200, blit=True)

                plt.show()
            # Calculate the mean and total area of positive regions for each sample and gas environment
            labeled_map_pos = label(output == 1)
            regions_pos = regionprops(labeled_map_pos)
            tot_area_pos = sum(region.area for region in regions_pos)

            labeled_map_neg = label(output == -1)
            regions_neg = regionprops(labeled_map_neg)
            tot_area_neg = sum(region.area for region in regions_neg)

            pos_areas_sizes.append([region.area for region in regions_pos])
            neg_areas_sizes.append([region.area for region in regions_neg])            

            tot_area = tot_area_pos + tot_area_neg

            areas.append(tot_area)
            pos_areas.append(tot_area_pos)
            neg_areas.append(tot_area_neg)

        positive_areas[f"{sample_path}_{gas_environment}"] = pos_areas
        negative_areas[f"{sample_path}_{gas_environment}"] = neg_areas
        movement[f"{sample_path}_{gas_environment}"] = areas

#print num investigated images per sample and gas



def plot_metric(metric, title, ylabel, std_dev=None):
    #set font size
    plt.rcParams.update({'font.size': 14})
    gases = ['Air', 'N2', 'N2H2', 'CO2H2']
    colors = {'Air': 'r', 'N2': 'g', 'N2H2': 'b', 'CO2H2': 'c'}
    bar_width = 0.2  # Width of individual bars
    indices = np.arange(len(metric))  # Indices for sample locations

    fig, ax = plt.subplots(figsize=(14, 9))

    for i, gas in enumerate(gases):
        gas_values = [metric[sample].get(gas, 0) for sample in metric.keys()]
        std_errors = [std_dev[sample].get(gas, 0) if std_dev and sample in std_dev else 0 for sample in metric.keys()]
        std_errors = [min(err, val) for err, val in zip(std_errors, gas_values)]  # Ensure error bar does not extend below zero
        bar_positions = indices + i * bar_width - bar_width  # Adjust bar positions for grouping
        # [Air, N2, CO2H2] should be ['Residual Air', 'N$_2$', 'CO$_2$+H$_2$']
        gas_latex = "Residual Air, 400°C" if gas == "Air" else "N$_2$, 280°C" if gas == "N2" else "CO$_2$+H$_2$, 280°C" if gas == "CO2H2" else "N$_2$+H$_2$, 280°C"
        bars = ax.bar(bar_positions, gas_values, bar_width, label=gas_latex, color=colors[gas], yerr=std_errors, capsize=5)
    # Customizing the plot
    names_latex = ["CP InO$_x$/m-ZrO$_2$","InO$_x$/YSZ", "InO$_x$/m-ZrO$_2$", "FSP InO$_x$/t-ZrO$_2$", "InO$_x$/m-ZrO$_2$ \n (S$_{BET}$ = 20 m$^2$g$^{-1}$)", "InO$_x$/t-ZrO$_2$","Pd/InO$_x$/m-ZrO$_2$", "InO$_x$/TiO$_2$"]
    ax.set_xticks(indices)
    ax.set_xticklabels(names_latex, rotation=45, ha='center',multialignment='center')
    ax.set_ylabel(ylabel)
    #ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(ylabel + ".png", dpi=300)
    plt.show()

#Scatter positive_areas+negative_areas against positive_areas-negative_areas for InOmZrO, InOZrOSBET20, InOtZrO, InOTiO in Air, N2, CO2H2 and save figs
#save positive_areas and negative_areas to csv

# Writing the positive areas
with open('positive_areas_231024.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Sample", "Gas", "Positive Area"])  # Adjust header if needed
    for sample in sample_paths:
        for gas in gas_environments:
            #if gas does not exist for given sample
            if f"{sample}_{gas}" not in positive_areas:
                continue
            # Unpack the list so each element has its own column
            writer.writerow([sample.split("/")[1], gas, *positive_areas[f"{sample}_{gas}"]])

# Writing the negative areas
with open('negative_areas_231024.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Sample", "Gas", "Negative Area"])  # Adjust header if needed
    for sample in sample_paths:
        for gas in gas_environments:
            if f"{sample}_{gas}" not in positive_areas:
                continue
            # Unpack the list so each element has its own column
            writer.writerow([sample.split("/")[1], gas, *negative_areas[f"{sample}_{gas}"]])





for gas in gas_environments:
    plt.figure(figsize=(10, 10))
    sample_colors = {'data/InOmZrO': 'r', 'data/InOZrOSBET20': 'g', 'data/InOtZrO': 'b', 'data/InOTiO': 'y'}
    sample_markers = {'data/InOmZrO': 'o', 'data/InOZrOSBET20': 's', 'data/InOtZrO': 'D', 'data/InOTiO': '^'}
    sample_latex_labels  = {'data/InOmZrO': 'HS-InO$_x$/m-ZrO$_2$', 'data/InOZrOSBET20': 'LS-InO$_x$/m-ZrO$_2$', 'data/InOtZrO': 'InO$_x$/t-ZrO$_2$', 'data/InOTiO': 'InO$_x$/TiO$_2$'}
    min_A_tot = 1e6
    max_A_tot = -1e6
    min_A_diff = 1e6
    max_A_diff = -1e6
    for sample in ["data/InOmZrO", "data/InOZrOSBET20", "data/InOtZrO", "data/InOTiO"]:
        #if sample does not exist for given gas
        if f"{sample}_{gas}" not in positive_areas:
            continue
        pos_area = positive_areas[f"{sample}_{gas}"]
        neg_area = negative_areas[f"{sample}_{gas}"]
        A_tot = [pos + neg for pos, neg in zip(pos_area, neg_area)]
        A_diff = [pos - neg for pos, neg in zip(pos_area, neg_area)]
        min_A_tot = min(min(A_tot), min_A_tot)
        max_A_tot = max(max(A_tot), max_A_tot)
        min_A_diff = min(min(A_diff), min_A_diff)
        max_A_diff = max(max(A_diff), max_A_diff)
        # Plot scatter points
        plt.scatter(A_diff, A_tot, c=sample_colors[sample], marker=sample_markers[sample], label=sample_latex_labels[sample])
        #plt.scatter(pos_area, neg_area, c=sample_colors[sample], marker=sample_markers[sample], label=sample_latex_labels[sample])
        # Calculate and plot mean (centroid) of the cluster
        mean_A_diff = np.mean(A_diff)
        mean_A_tot = np.mean(A_tot)
        plt.scatter(mean_A_diff, mean_A_tot, c=sample_colors[sample], marker='X', s=200, edgecolor='black')
        #plt.scatter(np.mean(pos_area), np.mean(neg_area), c=sample_colors[sample], marker='X', s=200, edgecolor='black')

    plt.xlim(-(max(abs(min_A_diff), abs(max_A_diff))), max(abs(min_A_diff), abs(max_A_diff)))
    plt.gca().set_aspect('equal', adjustable='box')
    #label x latex "A_diff" and y latex "A_tot"
    plt.xlabel("A$_{diff}$ = A$_{increase}$ - A$_{decrease}$ [pixels]", fontsize=12)
    plt.ylabel("A$_{tot}$ = A$_{increase}$ + A$_{decrease}$ [pixels]", fontsize=12)
    x_vals_neg = np.linspace(-max_A_tot, 0, 100)
    x_vals = np.linspace(0, max_A_diff, 100)
    plt.plot(x_vals, x_vals, '--', color='black',alpha=0.5)
    plt.plot(x_vals_neg, -x_vals_neg, '--', color='black',alpha=0.5)
    # Add a grid for clarity
    plt.grid(True, linestyle='--', alpha=0.6)
    # Add a background color to the entire plot for emphasis
    plt.gca().set_facecolor('#f0f0f0')
    # Adjust legend to show both sample points and their means
    plt.legend(loc='lower right')
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"Total_Area_vs_Positive_Area_Negative_Area_{gas}.png", dpi=300)
    plt.show()


print(positive_areas["data/InOmZrO_Air"])
print(movement["data/InOmZrO_N2"])
print(movement["data/InOmZrO_CO2H2"])
# calc mean and std deviation per sample+gas
mean_movement = {}
std_dev_movement = {}
for sample in sample_paths:
    mean_movement[sample] = {}
    std_dev_movement[sample] = {}
    for gas in gas_environments:
        if f"{sample}_{gas}" not in movement:
            continue
        mean_movement[sample][gas] = np.mean(movement[f"{sample}_{gas}"])
        std_dev_movement[sample][gas] = np.std(movement[f"{sample}_{gas}"])

#calc standard error of the mean
std_err_movement = {}
for sample in sample_paths:
    std_err_movement[sample] = {}
    for gas in gas_environments:
        if f"{sample}_{gas}" not in movement:
            continue
        std_err_movement[sample][gas] = std_dev_movement[sample][gas] / np.sqrt(len(movement[f"{sample}_{gas}"]))

mean_positive_areas = {}
std_dev_positive_areas = {}
for sample in sample_paths:
    mean_positive_areas[sample] = {}
    std_dev_positive_areas[sample] = {}
    for gas in gas_environments:
        if f"{sample}_{gas}" not in positive_areas:
            continue
        mean_positive_areas[sample][gas] = np.mean(positive_areas[f"{sample}_{gas}"])
        std_dev_positive_areas[sample][gas] = np.std(positive_areas[f"{sample}_{gas}"])

std_err_positive_areas = {}
for sample in sample_paths:
    std_err_positive_areas[sample] = {}
    for gas in gas_environments:
        if f"{sample}_{gas}" not in positive_areas:
            continue
        std_err_positive_areas[sample][gas] = std_dev_positive_areas[sample][gas] / np.sqrt(len(positive_areas[f"{sample}_{gas}"]))

mean_negative_areas = {}
std_dev_negative_areas = {}
for sample in sample_paths:
    mean_negative_areas[sample] = {}
    std_dev_negative_areas[sample] = {}
    for gas in gas_environments:
        if f"{sample}_{gas}" not in negative_areas:
            continue
        mean_negative_areas[sample][gas] = np.mean(negative_areas[f"{sample}_{gas}"])
        std_dev_negative_areas[sample][gas] = np.std(negative_areas[f"{sample}_{gas}"])

std_err_negative_areas = {}
for sample in sample_paths:
    std_err_negative_areas[sample] = {}
    for gas in gas_environments:
        if f"{sample}_{gas}" not in negative_areas:
            continue
        std_err_negative_areas[sample][gas] = std_dev_negative_areas[sample][gas] / np.sqrt(len(negative_areas[f"{sample}_{gas}"]))
# Calculate the difference between positive and negative areas
pos_neg_diff = {}
for sample in sample_paths:
    pos_neg_diff[sample] = {}
    for gas in gas_environments:
        if f"{sample}_{gas}" not in positive_areas or f"{sample}_{gas}" not in negative_areas:
            continue
        pos_neg_diff[sample][gas] = mean_positive_areas[sample][gas] - mean_negative_areas[sample][gas]

std_err_pos_neg_diff = {}
for sample in sample_paths:
    std_err_pos_neg_diff[sample] = {}
    for gas in gas_environments:
        if f"{sample}_{gas}" not in positive_areas or f"{sample}_{gas}" not in negative_areas:
            continue
        std_err_pos_neg_diff[sample][gas] = np.sqrt(std_err_positive_areas[sample][gas]**2 + std_err_negative_areas[sample][gas]**2)


# Plot the metrics


plot_metric(mean_movement, "Mean Movement of Samples in Different Gas Environments", "Mean Movement [pixels per series]", std_dev=std_err_movement)
plot_metric(mean_positive_areas, "Mean Positive Area of Samples in Different Gas Environments", "Mean Positive Area [pixels per series]", std_dev=std_err_positive_areas)
plot_metric(mean_negative_areas, "Mean Negative Area of Samples in Different Gas Environments", "Mean Negative Area [pixels per series]", std_dev=std_err_negative_areas)

def plot_metric_neg(metric, title, ylabel, std_dev=None):
    # Set font size
    plt.rcParams.update({'font.size': 14})
    gases = ['Air', 'N2', 'CO2H2']
    colors = {'Air': 'r', 'N2': 'g', 'CO2H2': 'b'}
    bar_width = 0.2  # Width of individual bars
    indices = np.arange(len(metric))  # Indices for sample locations

    fig, ax = plt.subplots(figsize=(14, 9))

    for i, gas in enumerate(gases):
        gas_values = [metric[sample].get(gas, 0) for sample in metric.keys()]
        std_errors = [std_dev[sample].get(gas, 0) if std_dev and sample in std_dev else 0 for sample in metric.keys()]

        # Ensure error bar does not extend below zero for positive values
        std_errors_neg = [min(err, val - 0) if val >= 0 else min(err, abs(val)) for err, val in zip(std_errors, gas_values)]
        std_errors_pos = [err for err in std_errors]

        bar_positions = indices + i * bar_width - bar_width  # Adjust bar positions for grouping
        gas_latex = "Residual Air, 400°C" if gas == "Air" else "N$_2$, 280°C" if gas == "N2" else "CO$_2$+H$_2$, 280°C"
        
        bars = ax.bar(bar_positions, gas_values, bar_width, label=gas_latex, color=colors[gas], 
                      yerr=[std_errors_neg, std_errors_pos], capsize=5)

    # Customizing the plot
    names_latex = ["CP InO$_x$/m-ZrO$_2$","InO$_x$/YSZ", "InO$_x$/m-ZrO$_2$", "FSP InO$_x$/t-ZrO$_2$", "InO$_x$/m-ZrO$_2$ \n (S$_{BET}$ = 20 m$^2$g$^{-1}$)", "InO$_x$/t-ZrO$_2$","Pd/InO$_x$/m-ZrO$_2$", "InO$_x$/TiO$_2$"]
    ax.set_xticks(indices)
    ax.set_xticklabels(names_latex, rotation=45, ha='center', multialignment='center')
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(ylabel + ".png", dpi=300)
    plt.show()



plot_metric_neg(pos_neg_diff, "Difference between Mean Positive and Negative Areas of Samples in Different Gas Environments", "Difference [pixels per series]", std_dev=std_err_pos_neg_diff)

#save data to csv
import csv

with open('movement.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Sample", "Gas", "Mean Movement", "Standard Error"])
    for sample in sample_paths:
        for gas in gas_environments:
            if f"{sample}_{gas}" not in movement:
                continue
            writer.writerow([sample, gas, mean_movement[sample][gas], std_err_movement[sample][gas]])

with open('positive_areas.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Sample", "Gas", "Mean Positive Area", "Standard Error"])
    for sample in sample_paths:
        for gas in gas_environments:
            if f"{sample}_{gas}" not in positive_areas:
                continue
            writer.writerow([sample, gas, mean_positive_areas[sample][gas], std_err_positive_areas[sample][gas]])

with open('negative_areas.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Sample", "Gas", "Mean Negative Area", "Standard Error"])
    for sample in sample_paths:
        for gas in gas_environments:
            if f"{sample}_{gas}" not in negative_areas:
                continue
            writer.writerow([sample, gas, mean_negative_areas[sample][gas], std_err_negative_areas[sample][gas]])

with open('pos_neg_diff.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Sample", "Gas", "Mean Difference", "Standard Error"])
    for sample in sample_paths:
        for gas in gas_environments:
            if gas == "N2H2" and sample != "data/InOmZrO":
                continue
            #if f"{sample}_{gas}" not in pos_neg_diff:
            #    continue
            writer.writerow([sample, gas, pos_neg_diff[sample][gas], std_err_pos_neg_diff[sample][gas]])

print("Data saved to csv files")