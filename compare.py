import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.signal import find_peaks

def compare_images(dir, index1, index2, index3):
    images = []
    # Load and median filter images
    for filename in sorted(os.listdir(dir)):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff"):
            img = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_GRAYSCALE)
            filtered_img = cv2.medianBlur(img, 5)
            images.append(filtered_img)
    
    # Calculate average images
    avg_img1 = np.mean(images[index1:index2], axis=0)
    avg_img2 = np.mean(images[index2:index3], axis=0)
    
    # Calculate horizontal sums
    horizontal_sum1 = np.sum(avg_img1, axis=0)
    horizontal_sum2 = np.sum(avg_img2, axis=0)

    # Find local peaks
    peaks1, _ = find_peaks(horizontal_sum1, prominence=1000)
    peaks2, _ = find_peaks(horizontal_sum2, prominence=1000)

    plt.figure(figsize=(20, 4))  # Doubled the horizontal size

    # Find positions of maximums
    max_pos1 = np.argmax(horizontal_sum1)
    max_pos2 = np.argmax(horizontal_sum2)
    print(f"Max position for first range: {max_pos1}")
    print(f"Max position for second range: {max_pos2}")
    
   # Plot first range with peaks
    plt.subplot(1, 2, 1)
    plt.plot(horizontal_sum1)
    plt.plot(peaks1, horizontal_sum1[peaks1], "x")
    plt.title("from " + str(index1) + " to " + str(index2))
    plt.vlines(x=peaks1, ymin=0, ymax=max(horizontal_sum1), color="C1")
    plt.xticks(peaks1)

    # Plot second range with peaks
    plt.subplot(1, 2, 2)
    plt.plot(horizontal_sum2)
    plt.plot(peaks2, horizontal_sum2[peaks2], "x")
    plt.title("from " + str(index2) + " to " + str(index3))
    plt.vlines(x=peaks2, ymin=0, ymax=max(horizontal_sum2), color="C1")
    plt.xticks(peaks2)

    plt.show()

    for peak in peaks1:
        cv2.line(avg_img1, (peak, 0), (peak, avg_img1.shape[0]-1), (255, 0, 0), 3)
        cv2.putText(avg_img1, str(peak), (peak, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    for peak in peaks2:
        cv2.line(avg_img2, (peak, 0), (peak, avg_img2.shape[0]-1), (255, 0, 0), 3)
        cv2.putText(avg_img2, str(peak), (peak, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    
    # Combine images for display
    combined_image = np.vstack((avg_img1, avg_img2))
    cv2.imshow('Average Images', combined_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare images from a directory.")
    parser.add_argument("dir", default='imgs', type=str, help="Directory containing the images")
    parser.add_argument("index1", default=0, type=int, help="Start index for the first range")
    parser.add_argument("index2", default=6, type=int, help="End index for the first range and start index for the second range")
    parser.add_argument("index3", default=12, type=int, help="End index for the second range")
    
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
    compare_images(args.dir, args.index1, args.index2, args.index3)