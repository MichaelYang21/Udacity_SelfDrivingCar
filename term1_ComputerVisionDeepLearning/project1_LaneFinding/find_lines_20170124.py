# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
subdir="test_images/"
Figure_Files_All=os.listdir(subdir)

Figure_Files_Original=[]

for Figure_Files_i in Figure_Files_All:
    if "Lines" not in Figure_Files_i and ".jpg" in Figure_Files_i:
        Figure_Files_Original+=[Figure_Files_i]
print(Figure_Files_Original)

for i,Figure_Name in enumerate(Figure_Files_Original):
# i=1
#     Figure_Name=Figure_Files_Original[i]
    print(Figure_Name)
    img=mpimg.imread(subdir+Figure_Name)
    plt.imshow(img)
    plt.show()

    gray=grayscale(img)
    # plt.imshow(gray,cmap='gray')
    # plt.show()

    kernel_size=5
    blur_gray=gaussian_blur(gray, kernel_size)
    # plt.imshow(blur_gray,cmap='gray')
    # plt.show()

    low_threshold=150
    high_threshold=250
    edges=canny(blur_gray, low_threshold, high_threshold)
#     plt.imshow(edges)
#     plt.show()

    #imshape=img.shape
    #print(imshape)
    #vertices=np.array([[30,30],[40,30],[30,90]],dtype=np.int32)
    #masked_edges=region_of_interest(edgs, vertices)

    mask = np.zeros_like(edges)

    if len(edges.shape) > 2:
        channel_count = edges.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = image.shape

    vertices = np.array([[(480,310), (900, 550), (100,550)]], dtype=np.int32)

    # if i==0 :
    #     vertices = np.array([[(480,310), (900, 550), (100,550)]], dtype=np.int32)
    # if i==1:
    #     vertices = np.array([[(225,150), (400, 230), (100,230)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

#     plt.imshow(masked_edges)
#     plt.show()

    rho=9
    theta=10*np.pi/180
    threshold=15
    min_line_len=15
    max_line_gap=20
    line_img=np.copy(img)*0
    lines=hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    # plt.imshow(lines)
    # plt.show()

    lines_edges=weighted_img(lines, img, α=0.7, β=1.2, λ=0.)
    plt.imshow(lines_edges)
    plt.show()
    plt.savefig(subdir+Figure_Name[:-4]+"_Lines.jpg")
