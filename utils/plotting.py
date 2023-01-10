import pickle
import sys
import os
import statistics
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]
red = (0, 0, 255)
green = (0, 255, 0)
cyan = (255, 255, 0)
random.seed(42)
np.random.seed(42)

def process_results_simulation(error_m, save_folder):
    res_error_m = error_m
    #res_error_m = [e for gt, e in zip(res_ground_truth, res_error_m) if mask[gt[1], gt[0]]>100]
    #res_ground_truth = [gt for gt in res_ground_truth if mask[gt[1], gt[0]]>100]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    save_filename = os.path.join(save_folder, 'matching_results.txt')
    f = open(save_filename, "a")

    total_tested = len(res_error_m)
    error_0 = res_error_m.count(0)
    f.write("Perfect matches: %d of %d (%.2f%%) \n" % (error_0, total_tested, 100*error_0/total_tested))

    error_25 = sum(x <= 25 for x in res_error_m)
    f.write("Mismatch less or equal to 25m: %d of %d (%.2f%%) \n" % (error_25, total_tested, 100*error_25/total_tested))

    error_50 = sum(x <= 50 for x in res_error_m)
    f.write("Mismatch less or equal to 50m: %d of %d (%.2f%%) \n" % (error_50, total_tested, 100*error_50/total_tested))

    error_100 = sum(x <= 100 for x in res_error_m)
    f.write("Mismatch less or equal to 100m: %d of %d (%.2f%%) \n" % (error_100, total_tested, 100*error_100/total_tested))

    error_150 = sum(x <= 150 for x in res_error_m)
    f.write("Mismatch less or equal to 150m: %d of %d (%.2f%%) \n" % (error_150, total_tested, 100*error_150/total_tested))

    f.write("Mean error: %.2fm \n" % (np.mean(res_error_m)))
    print(f"Mean error: {np.mean(res_error_m)}")

    f.close()

    plt.hist(res_error_m, histtype='step', bins=130)
    plt.title('Histogram of localization error')
    plt.xlabel("error")
    plt.ylabel("No. occurences")
    plt.savefig(os.path.join(save_folder, 'hist_error_localization.pdf'))

def save_heatmap_simulation(pos, err, database_image_path, config, save_folder, index=None):
    basemap_img = cv2.imread(database_image_path)
    basemap_img = basemap_img[config[0]:config[2], config[1]:config[3], :]

    for i in range(len(pos)):
        pos_single = list(map(int, pos[i]))
        err_single = err[i]
        if err_single < 50:
            cv2.circle(basemap_img, (pos_single[1] - config[1], pos_single[0] - config[0]), 9, green, 2)
        elif err_single < 100:
            cv2.circle(basemap_img, (pos_single[1] - config[1], pos_single[0] - config[0]), 9, cyan, 2)
        else:
            cv2.circle(basemap_img, (pos_single[1] - config[1], pos_single[0] - config[0]), 9, red, 2)

    if index is None:
        cv2.imwrite(os.path.join(save_folder, "heatmap.png"), basemap_img)
    else:
        cv2.imwrite(os.path.join(save_folder, f"heatmap{index}.png"), basemap_img)