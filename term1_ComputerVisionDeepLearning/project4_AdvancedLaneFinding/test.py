import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


nx = 9#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

objpoints=[]
imgpoints=[]

objp=np.zeros((ny*nx,3),np.float32)
objp[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2)


cwd = os.getcwd()
cameral_cal_dir='camera_cal'
cameral_cal_files=os.listdir(cwd+'/'+cameral_cal_dir)
# print(cameral_cal_files)
for jpg_file in cameral_cal_files:
    # img = mpimg.imread(cwd+'/camera_cal/calibration3.jpg')
    print(jpg_file)
    img = mpimg.imread(cwd+'/'+cameral_cal_dir+'/'+jpg_file)
    # plt.imshow(img)
    # plt.show()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    print("ret=",ret)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        # Draw and display the corners
        print("draw and display the corners")
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # plt.imshow(img)
        # plt.show()



# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
#    undist = np.copy(img)  # Delete this line
    return undist

#   Use color transforms, gradients, etc., to create a thresholded binary image.
def pipeline(img):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 30
    thresh_max = 150
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 175
    s_thresh_max = 250
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary,color_binary

# Define a function that takes an image, number of x and y points,
def corners_unwarp(undist, area_of_interest):
    # Choose an offset from image corners to plot detected corners
    offset1 = 200  # offset for dst points x value
    offset2 = 0  # offset for dst points bottom y value
    offset3 = 0  # offset for dst points top y value
    # Grab the image shape
    # img_size = (gray.shape[1], gray.shape[0])
    img_size = (gray.shape[1], gray.shape[0])

    # For source points I'm grabbing the outer four detected corners
    src = np.float32(area_of_interest)
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    dst = np.float32([[offset1, offset3],
                      [img_size[0] - offset1, offset3],
                      [img_size[0] - offset1, img_size[1] - offset2],
                      [offset1, img_size[1] - offset2]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)
    # Return the resulting image and matrix
    return warped, M, Minv

def poly_fit(binary_warped):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit,right_fit,out_img,left_lane_inds,right_lane_inds,nonzeroy,nonzerox

# Read in an image
test_img_folder="test_images"
output_folder="output_images"
output_folder_undist="undist"
output_folder_threshold_binary_image="threshold"
output_folder_birdview="birdview"

test_img_names=os.listdir('./'+test_img_folder)
for test_img_name in test_img_names:
    img = cv2.imread(cwd+'/'+test_img_folder+'/'+test_img_name)
    plt.imshow(img)
    # plt.show()
    undistorted = cal_undistort(img, objpoints, imgpoints)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()
    directory="./" + output_folder + "/" + output_folder_undist
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    plt.savefig(directory+"/"+test_img_name[:-4]+"undist.jpg")
    plt.close()



    # thresholded images
    combined_binary,color_binary=pipeline(undistorted)
    # Plotting thresholded images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Stacked thresholds')
    ax1.imshow(color_binary)
    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined_binary, cmap='gray')


    directory_threshold="./" + output_folder + "/" + output_folder_threshold_binary_image


    try:
        os.stat(directory_threshold)
    except:
        os.mkdir(directory_threshold)
    plt.savefig(directory_threshold+"/"+test_img_name[:-4]+"thresholded.jpg")
    plt.close()

    #   Apply a perspective transform to rectify binary image ("birds-eye view").
    # M = cv2.getPerspectiveTransform(src, dst)
    # warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

# # Define image shape
# image_shape = image.shape
# print("image shape:", image_shape)

    # Define the region
    # area_of_interest = [[150 + 430, 460], [1150 - 440, 460], [1150, 720], [150, 720]]
    area_of_interest = [[150 + 430, 460], [1150 - 440, 460], [1050, 683], [257, 683]]
    plt.imshow(combined_binary)
    plt.plot(area_of_interest[1][0],area_of_interest[1][1],'.')
    plt.plot(area_of_interest[2][0],area_of_interest[2][1],'.')
    plt.plot(area_of_interest[3][0],area_of_interest[3][1],'.')
    plt.plot(area_of_interest[0][0],area_of_interest[0][1],'.')
    # plt.show()
    plt.close()

    binary_warped,M,Minv=corners_unwarp(combined_binary, area_of_interest)
    plt.imshow(binary_warped)
    # plt.show()

    directory_threshold="./" + output_folder + "/" + output_folder_birdview
    try:
        os.stat(directory_threshold)
    except:
        os.mkdir(directory_threshold)
    plt.savefig(directory_threshold+"/"+test_img_name[:-4]+"birdview.jpg")
    plt.close()

#   Detect lane pixels and fit to find the lane boundary.
#     print("binary_warped.shape[0]=",binary_warped.shape[0])
    left_fit, right_fit, out_img, left_lane_inds, right_lane_inds, nonzeroy, nonzerox=poly_fit(binary_warped)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    # plt.show()
    output_folder_ploy_fit="ploy_fit"
    directory_threshold="./" + output_folder + "/" + output_folder_ploy_fit
    try:
        os.stat(directory_threshold)
    except:
        os.mkdir(directory_threshold)
    plt.savefig(directory_threshold+"/"+test_img_name[:-4]+"ploy_fit.jpg")
    plt.close()
#   Determine the curvature of the lane and vehicle position with respect to center.
#   Warp the detected lane boundaries back onto the original image.
#   Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.