import numpy as np
import cv2
import scipy.misc # For saving images as needed

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.sin(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def ground_extract(img):
    return color_threshold(img, (160,160,160), (255,255,255))
    # return color_threshold(img, (165,165,165), (255,255,255))

def obstacles_extract(img):
    # return color_threshold(img, (1,1,1), (160,160,160))
    return color_threshold(img, (1,1,1), (150,150,150))
    # return color_threshold(img, (1,1,1), (70,60,50))
    # return color_threshold(img, (1,1,1), (60,60,60))
    # return color_threshold(img, (1,1,1), (30,30,30))


def rocks_extract(img):
    return color_threshold(img, (130,100,0), (200,200,70))

def color_threshold(img, lower_thresh, upper_thresh):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
   
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]
    r_binary = np.zeros_like(img[:,:,0])
    g_binary = np.zeros_like(img[:,:,0])
    b_binary = np.zeros_like(img[:,:,0])
    r_binary[(r_channel >= lower_thresh[0]) & (r_channel <= upper_thresh[0])] = 1
    g_binary[(g_channel >= lower_thresh[1]) & (g_channel <= upper_thresh[1])] = 1
    b_binary[(b_channel >= lower_thresh[2]) & (b_channel <= upper_thresh[2])] = 1
    
    color_select[(r_binary == 1) & (g_binary == 1) & (b_binary == 1)] = 1
    # Return the binary image
    return color_select

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    image = Rover.img
    # 1) Define source and destination points for perspective transform
    # source = np.float32([[12, 141], [118, 96], [200,96], [300,141]])
    # destination = np.float32([[150, 150], [150,140] ,[160,140], [160, 150]])
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                      [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      ])
    # 2) Apply perspective transform
    persp_transf = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # colorsel = color_thresh(persp_transf, rgb_thresh=(160, 160, 160))
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    obs_sel = obstacles_extract(persp_transf)
    rock_sel = rocks_extract(persp_transf)
    grnd_sel = ground_extract(persp_transf)
    Rover.vision_image[:,:,0] = obs_sel * 255
    Rover.vision_image[:,:,1] = rock_sel * 255
    Rover.vision_image[:,:,2] = grnd_sel * 255
    # print(np.sum(colorsel))
    # scipy.misc.imsave('../output/colorsel.jpg', colorsel*255)
    # 5) Convert map image pixel values to rover-centric coords
    xpix_obs, ypix_obs = rover_coords(obs_sel)
    xpix_rock, ypix_rock = rover_coords(rock_sel)
    xpix_grnd, ypix_grnd = rover_coords(grnd_sel)

    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10
    world_size = Rover.worldmap.shape[0]
    obstacle_x_world, obstacle_y_world = pix_to_world(xpix_obs, ypix_obs, Rover.pos[0], Rover.pos[1], 
                                    Rover.yaw, world_size, scale)
    rock_x_world, rock_y_world = pix_to_world(xpix_rock, ypix_rock, Rover.pos[0], Rover.pos[1], 
                                    Rover.yaw, world_size, scale)
    navigable_x_world, navigable_y_world = pix_to_world(xpix_grnd, ypix_grnd, Rover.pos[0], Rover.pos[1], 
                                    Rover.yaw, world_size, scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    limit = 0.5#1.0
    if (((abs(Rover.pitch) < limit)| (abs(Rover.pitch) > (360.0 - limit))) & 
        ((abs(Rover.roll) < limit)| (abs(Rover.roll) > (360.0 - limit)))):
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
        Rover.worldmap[navigable_y_world, navigable_x_world, 0] = 0 # remove obstacles
    # Rover.worldmap[y_world, x_world,2] += 1
    # 8) Convert rover-centric pixel positions to polar coordinates
    rover_centric_pixel_distances, rover_centric_angles = to_polar_coords(xpix_grnd, ypix_grnd)
    # Update Rover pixel distances and angles
    Rover.nav_dists = rover_centric_pixel_distances
    Rover.nav_angles = rover_centric_angles
   
    
    return Rover