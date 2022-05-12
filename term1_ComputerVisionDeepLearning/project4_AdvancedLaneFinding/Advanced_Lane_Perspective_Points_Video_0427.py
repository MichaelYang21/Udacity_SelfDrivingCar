import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



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

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]
    b_thresh_min = 175
    b_thresh_max = 250
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    Luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    L_channel = Luv[:, :, 0]
    L_thresh_min = 60
    L_thresh_max = 255
    L_binary = np.zeros_like(L_channel)
    L_binary[(L_channel >= L_thresh_min) & (L_channel <= L_thresh_max)] = 1


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
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary,b_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1) | (b_binary==1) ] = 1

    return combined_binary,color_binary

# Define a function that takes an image, number of x and y points,
def corners_unwarp(undist, area_of_interest):
    # Choose an offset from image corners to plot detected corners
    offset1 = 200  # offset for dst points x value
    offset2 = 0  # offset for dst points bottom y value
    offset3 = 0  # offset for dst points top y value
    # Grab the image shape
    # img_size = (gray.shape[1], gray.shape[0])
    img_size = (undist.shape[1], undist.shape[0])

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



def curvature(ploty,leftx,rightx):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    y_eval = np.mean(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    return left_curverad,right_curverad

def find_perspective_points(combined_binary):
    from math import ceil,pi,atan2,cos,sin
    edges = combined_binary

    # Computing perspective points automatically
    rho = 10              # distance resolution in pixels of the Hough grid
    theta = 0.1*np.pi/180  # angular resolution in radians of the Hough grid
    threshold = 5       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5 # minimum number of pixels making up a line
    max_line_gap = 30    # maximum gap in pixels between connectable line segments

    angle_min_mag = 20*pi/180
    angle_max_mag = 60*pi/180

    lane_markers_x = [[], []]
    lane_markers_y = [[], []]

    masked_edges = np.copy(edges)
    masked_edges[:edges.shape[0]*8//10,:] = 0
    # masked_edges[edges.shape[0] * 6 // 10:, :] = 0
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            theta = atan2(y1-y2, x2-x1)
            rho = ((x1+x2)*cos(theta) + (y1+y2)*sin(theta))/2
            if (abs(theta) >= angle_min_mag and abs(theta) <= angle_max_mag):
                if theta > 0: # positive theta is downward in image space?
                    i = 0 # Left lane marker
                else:
                    i = 1 # Right lane marker
                lane_markers_x[i].append(x1)
                lane_markers_x[i].append(x2)
                lane_markers_y[i].append(y1)
                lane_markers_y[i].append(y2)

    if len(lane_markers_x[0]) < 1 or len(lane_markers_x[1]) < 1:
        # Failed to find two lane markers
        return None

    p_left  = np.polyfit(lane_markers_y[0], lane_markers_x[0], 1)
    p_right = np.polyfit(lane_markers_y[1], lane_markers_x[1], 1)

    # Find intersection of the two lines
    apex_pt = np.linalg.solve([[p_left[0], -1], [p_right[0], -1]], [-p_left[1], -p_right[1]])
    top_y = ceil(apex_pt[0] + 0.075*edges.shape[0])

    bl_pt = ceil(np.polyval(p_left, edges.shape[0]))
    tl_pt = ceil(np.polyval(p_left, top_y))

    br_pt = ceil(np.polyval(p_right, edges.shape[0]))
    tr_pt = ceil(np.polyval(p_right, top_y))

    src = np.array([[tl_pt, top_y],
                    [tr_pt, top_y],
                    [br_pt, edges.shape[0]],
                    [bl_pt, edges.shape[0]]], np.float32)

    area_of_interest=[[tl_pt, top_y],
                    [tr_pt, top_y],
                    [br_pt, edges.shape[0]],
                    [bl_pt, edges.shape[0]]]

    return src,area_of_interest

def draw_poly(image,undist,warped,left_fitx,right_fitx,ploty,curvature,Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))


    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    pts=np.argwhere(newwarp[:, :, 1])
    position=find_position(pts, image)
    print("vehicle position=",position)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: {} m".format(int(curvature))
    cv2.putText(result,text,(400,100), font, 1,(255,255,255),2)
    if position < 0:
        text = "Vehicle is {:.2f} m left of center".format(-position)
    else:
        text = "Vehicle is {:.2f} m right of center".format(position)
    cv2.putText(result,text,(400,150),font, 1,(255,255,255),2)
    plt.imshow(result)
    # plt.show()
    return result

def find_position(pts,img):
    # Find the position of the car from the center
    # It will show if the car is 'x' meters from the left or right
    position = int(img.shape[1]/2)
    # print(position)
    y_position_remove=img.shape[0]/2
    left  = np.min(pts[(pts[:,1] < position) & (pts[:,0] > y_position_remove)][:,1])
    right = np.max(pts[(pts[:,1] > position) & (pts[:,0] > y_position_remove)][:,1])
    center = (left + right)/2
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/img.shape[1] # meteres per pixel in x dimension
    return (position - center)*xm_per_pix

# # Read in an image
# test_img_folder="test_images"
# output_folder="output_images"
# output_folder_undist="undist"
# output_folder_threshold_binary_image="threshold"
# output_folder_birdview="birdview"
def frames(test_img_folder,output_folder,plotting):
    output_folder_undist = "undist"
    output_folder_threshold_binary_image = "threshold"
    output_folder_birdview = "birdview"
    test_img_names=os.listdir('./'+test_img_folder)
    for frame_i, test_img_name in enumerate(test_img_names):
        img = cv2.imread(cwd+'/'+test_img_folder+'/'+test_img_name)
        plt.imshow(img)
        # plt.show()
        undistorted = cal_undistort(img, objpoints, imgpoints)

        if plotting:
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
        if plotting:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            # ax1.set_title('Stacked thresholds')
            # ax1.imshow(color_binary)
            ax1.set_title('Original')
            ax1.imshow(img)
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

        # Define the region
        # area_of_interest = [[150 + 430, 460], [1150 - 440, 460], [1150, 720], [150, 720]]
        # src, area_of_interest=find_perspective_points(combined_binary)
        try:
            src, area_of_interest=find_perspective_points(combined_binary)
        except:
            print("frame_i=",frame_i)
            print("area_of_interest could not be automatically found")
            area_of_interest = [[150 + 430, 460], [1150 - 440, 460], [1050, 683], [257, 683]]

        if plotting:
            plt.imshow(combined_binary)
            plt.plot(area_of_interest[1][0],area_of_interest[1][1],'.')
            plt.plot(area_of_interest[2][0],area_of_interest[2][1],'.')
            plt.plot(area_of_interest[3][0],area_of_interest[3][1],'.')
            plt.plot(area_of_interest[0][0],area_of_interest[0][1],'.')
            # plt.show()
            output_folder_area_of_interest="area_of_interest"
            directory_threshold="./" + output_folder + "/" + output_folder_area_of_interest
            try:
                os.stat(directory_threshold)
            except:
                os.mkdir(directory_threshold)
            plt.savefig(directory_threshold+"/"+test_img_name[:-4]+"area_of_interest.jpg")
            plt.close()

        binary_warped,M,Minv=corners_unwarp(combined_binary, area_of_interest)
        plt.imshow(binary_warped)
        # plt.show()
        if plotting:
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
        if plotting:
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
        print(test_img_name)
        left_curverad, right_curverad=curvature(ploty, left_fitx, right_fitx)
        mean_curverad=(left_curverad+right_curverad)/2.0
        print("mean curverature=",mean_curverad)

    #   Warp the detected lane boundaries back onto the original image.
        result=draw_poly(img, undistorted, binary_warped, left_fitx, right_fitx, ploty,mean_curverad,Minv)

        if plotting:
        #   Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
            output_folder_ploy_fit="lane"
            directory_threshold="./" + output_folder + "/" + output_folder_ploy_fit
            try:
                os.stat(directory_threshold)
            except:
                os.mkdir(directory_threshold)
            plt.savefig(directory_threshold+"/"+test_img_name[:-4]+"_lane.jpg")
            plt.close()
    return result


# Define a class to receive the characteristics of each line detection
import collections
class Line():
    def __init__(self,cache_length):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = collections.deque(maxlen=cache_length)
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = collections.deque(maxlen=cache_length)
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

def process_image(img, key_frame_interval=20, cache_length=10):

    if process_image.cache is None:

        left_line = Line(cache_length=cache_length)
        right_line = Line(cache_length=cache_length)

        cache = {'warp_m': None,
                 'warp_minv': None,
                 'frame_ctr': 0,
                 'left': left_line,
                 'right': right_line}
    else:
        cache = process_image.cache

    left_line = cache['left']
    right_line = cache['right']

    # img = cv2.imread(cwd+'/'+test_img_folder+'/'+test_img_name)
    plt.imshow(img)
    # plt.show()
    undistorted = cal_undistort(img, objpoints, imgpoints)

    # thresholded images
    combined_binary,color_binary=pipeline(undistorted)

    #   Apply a perspective transform to rectify binary image ("birds-eye view").

    # Define the region
    # area_of_interest = [[150 + 430, 460], [1150 - 440, 460], [1150, 720], [150, 720]]
    # src, area_of_interest=find_perspective_points(combined_binary)
    try:
        src, area_of_interest=find_perspective_points(combined_binary)
    except:
        print("area_of_interest could not be automatically found")
        area_of_interest = [[150 + 430, 460], [1150 - 440, 460], [1050, 683], [257, 683]]

    binary_warped,M,Minv=corners_unwarp(combined_binary, area_of_interest)
    plt.imshow(binary_warped)
    # plt.show()

#   Detect lane pixels and fit to find the lane boundary.
#     print("binary_warped.shape[0]=",binary_warped.shape[0])
    left_fit, right_fit, out_img, left_lane_inds, right_lane_inds, nonzeroy, nonzerox=poly_fit(binary_warped)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


    #   Determine the curvature of the lane and vehicle position with respect to center.

    left_curverad, right_curverad=curvature(ploty, left_fitx, right_fitx)
    mean_curverad=(left_curverad+right_curverad)/2.0
    print("mean curverature=",mean_curverad)

    mean_curverad_best=np.average(right_line.radius_of_curvature)

    mean_curv_dev=abs(mean_curverad-mean_curverad_best)<0.4*mean_curverad_best
    left_right_curv_dev= abs(left_curverad-right_curverad)< 0.4*left_curverad
    Normal=mean_curv_dev and left_right_curv_dev
    Normal=Normal or cache['frame_ctr']==0

    if Normal:
        pass
    else:
        mean_curverad=mean_curverad_best
        right_fitx=right_line.recent_xfitted[-1]
        left_fitx=left_line.recent_xfitted[-1]

    #   Warp the detected lane boundaries back onto the original image.
    right_line.radius_of_curvature.append(mean_curverad)
    left_line.radius_of_curvature.append(mean_curverad)
    right_line.recent_xfitted.append(right_fitx)
    left_line.recent_xfitted.append(left_fitx)
    result=draw_poly(img, undistorted, binary_warped, left_fitx, right_fitx, ploty,mean_curverad,Minv)

    cache['frame_ctr'] = cache['frame_ctr'] + 1
    process_image.cache = cache

    return result

from moviepy.editor import VideoFileClip
# import imageio
# imageio.plugins.ffmpeg.download()


# Camera Calibration
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

# Collecting object points and image points for camera calibration
for jpg_file in cameral_cal_files:
    # img = mpimg.imread(cwd+'/camera_cal/calibration3.jpg')
    # print(jpg_file)
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


# Apply undistortion to chess board
output_folder="output_images"
output_folder_chess_undist="chess_undist"
directory_threshold = "./" + output_folder + "/" + output_folder_chess_undist

for jpg_file in cameral_cal_files:
    img = mpimg.imread(cwd+'/'+cameral_cal_dir+'/'+jpg_file)
    undistorted = cal_undistort(img, objpoints, imgpoints)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    try:
        os.stat(directory_threshold)
    except:
        os.mkdir(directory_threshold)
    plt.savefig(directory_threshold + "/" + jpg_file[:-4] + "Chess_undist.jpg")
    plt.close()



# Single Images Processing
test_img_folder="test_images"
plotting=True
result=frames(test_img_folder,output_folder,plotting)
        # videofolder = "/video"
        # output_folder="output_images"
        # test_img_folder=output_folder+videofolder+"/test_images"
        # output_folder=output_folder+videofolder
        # plotting=False
        # frames(test_img_folder,output_folder)

# Video Processing

def clear_cache():
    process_image.cache = None
clear_cache()
white_output = 'video.mp4'
clip1 = VideoFileClip("project_video.mp4").subclip(35,41)
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
