import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

"""
Extract Key Points
"""

def extract_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        raise ValueError("Feature detection failed.")

    return keypoints, descriptors

"""
Image Stitching
Use the extracted key points to align and stitch images into a panorama
"""

def stitch_image_pair(image1, image2, keypoints1, descriptors1, keypoints2, descriptors2):
    # BruteForce Matcher
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda m: m.distance)

    best_matches = matches[:50]

    # Matching points
    source_pts = np.float32([keypoints1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dest_pts = np.float32([keypoints2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

    # Homography
    homography_matrix, status = cv2.findHomography(source_pts, dest_pts, cv2.RANSAC, 5.0)
    if homography_matrix is None:
        raise ValueError("Homography Fail.")

    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Corners Transformation
    corners_image1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_image1, homography_matrix)

    # Panorama dimensions
    all_corners = np.concatenate((
        transformed_corners,
        np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
    ), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation matrix
    translation_offset = [-x_min, -y_min]
    translation_matrix = np.array([
        [1, 0, translation_offset[0]],
        [0, 1, translation_offset[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Panorama
    stitched_width = x_max - x_min
    stitched_height = y_max - y_min

    # Perspective transform
    result = cv2.warpPerspective(
        image1,
        translation_matrix.dot(homography_matrix),
        (stitched_width, stitched_height)
    )
    result[
        translation_offset[1]:translation_offset[1]+height2,
        translation_offset[0]:translation_offset[0]+width2
    ] = image2

    return result

def visualize_panorama(panorama):
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Final Panorama')
    plt.show()

def main():
    images = []
    # Folder where the images are located
    img_folder = 'panorama_images'
    
    # Construct the full paths to the images
    image_paths = [os.path.join(img_folder, 'img_01.png'), os.path.join(img_folder, 'img_02.png')]

    # Check if the images exist in the specified directory
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Error: The file {path} does not exist.")
            return
        else:
            print(f"Found image: {path}")
    
    # Read the images
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Error: Could not read image {path}")
        images.append(img)

    try:
        print("Extracting keypoints...")
        keypoints_list = []
        descriptors_list = []
        for img in images:
            kp, desc = extract_keypoints(img)
            keypoints_list.append(kp)
            descriptors_list.append(desc)

        print("Stitching images...")
        panorama = stitch_image_pair(
            images[0], images[1],
            keypoints_list[0], descriptors_list[0],
            keypoints_list[1], descriptors_list[1]
        )

        # Save the final panorama
        cv2.imwrite('stitched_result.jpg', panorama)
        print("Panorama saved as 'stitched_result.jpg'")

        visualize_panorama(panorama)

    except Exception as e:
        print(f"Error during panorama creation: {str(e)}")

if __name__ == "__main__":
    main()
