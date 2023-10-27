import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def generate_gradient(colors):
    # Create an array to store our gradient
    gradient = np.zeros((256, 1, 3), dtype=np.uint8)

    # For each provided color
    for i in range(1, len(colors)):
        # Calculate the start and end of this gradient segment
        start = int((i - 1) / (len(colors) - 1) * 255)
        end = int(i / (len(colors) - 1) * 255)

        # Calculate the color at each point in this segment
        for j in range(start, end):
            ratio = (j - start) / (end - start)
            gradient[j, 0] = [
                int((1 - ratio) * colors[i - 1][k] + ratio * colors[i][k]) for k in range(3)
            ]

    return gradient

def apply_gradient(image_path, output_path, gradient):
    # Load the image
    image = Image.open(image_path).convert('L')
    pixels = np.array(image)

    # Map each grayscale value to the corresponding gradient color
    new_pixels = gradient[pixels]

    #remove uinnecessary dimensions
    new_pixels = np.squeeze(new_pixels)

    # Save the output
    new_image = Image.fromarray(new_pixels)
    new_image.save(output_path)

def visualize_gradient(gradient):
    # Convert the list of colors into the shape expected by apply_gradient and imshow
    gradient_array = np.array(gradient).reshape(256, 1, 3)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.imshow(gradient_array, aspect='auto')
    plt.show()

def get_image_palette(image_path, n_colors):
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')
    image = image.resize((50, 50))  # optional: reducing size to speed up computation
    pixel_data = np.array(image)

    # Reshape the data to 1D and apply K-means clustering
    pixel_data = pixel_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixel_data)

    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_

    # Ensure the color values are integers
    colors = colors.round(0).astype(np.uint8)
    
    # Convert colors to grayscale and sort by the grayscale value
    grayscale = [0.2989 * R + 0.5870 * G + 0.1140 * B for R, G, B in colors]
    colors = [color for _, color in sorted(zip(grayscale, colors))]
    
    # Convert the list of colors into the shape expected by apply_gradient and imshow
    return np.array(colors).reshape(256, 1, 3)

def create_color_scheme_PCA(image_path):
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')
    pixel_data = np.array(image)

    # Reshape the data to 2D and normalize to range 0-1
    pixel_data = pixel_data.reshape(-1, 3) / 255

    # Apply PCA to the data
    pca = PCA(n_components=3)
    pca.fit(pixel_data)

    # Subtract the mean color from each pixel
    centered_pixels = pixel_data - pca.mean_

    # Get the first two principal components
    principal_components = pca.components_[:2]

    # Project each pixel onto the plane defined by the first two principal components
    projected_pixels = np.dot(centered_pixels, principal_components.T)

    # Map the projected pixels to the range 0-1
    min_val = projected_pixels.min(axis=0)
    max_val = projected_pixels.max(axis=0)
    scaled_pixels = (projected_pixels - min_val) / (max_val - min_val)

    # Convert the projected pixels to RGB values and reshape to match the output of generate_gradient
    color_palette = (scaled_pixels * 255).astype(np.uint8).reshape(-1, 1, 3)

    return color_palette



# Generate and visualize the gradient
#new_gradient = get_image_palette('/Users/nicklatina/Desktop/CLIP Prompts/Prompt Tunnel.jpg', 256)


#new_gradient = generate_gradient([new_gradient[i][0] for i in range(0, 256, 100)])
# colors=[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),
#         (255, 173, 173),(255, 214, 165),(253, 255, 182),(202, 255, 191),
#         (155, 246, 255),(160, 196, 255),(189, 178, 255),(255, 198, 255)]*7
#colors = [(25, 100, 171), (0, 0, 0), (25, 100, 171), (255, 255, 255)]*9
colors = [(0,0,0), (0,0,0), (0,0,0), (200,255,200)]*10
#colors = [color if color in ((0,0,0),(20,20,20)) else (255,255,255) for color in colors]


#colors = [(0, 0, 0), (0, 20, 54), (187, 23, 100)]
new_gradient = generate_gradient(colors)

visualize_gradient(new_gradient)

# Apply the gradient to the image
apply_gradient('/Users/nicklatina/Desktop/CLIP Prompts/Prompt LC.jpeg', 
            '/Users/nicklatina/Desktop/output_image.jpg', new_gradient)



# Complexification

for i in range(1,len(colors)-1):
    gradient = generate_gradient(colors[0:-i])
    apply_gradient('/Users/nicklatina/Desktop/depth.jpg', 
             f'/Users/nicklatina/Desktop/depth/output_image_{1000+i}.jpg', gradient)


# Banding    
# for i in range(int(len(colors)/2)):
#     temp_colors = colors.copy()
#     temp_colors[i*2] = tuple([255, 0, 0])
#     gradient = generate_gradient(temp_colors)
#     apply_gradient('/Users/nicklatina/Desktop/depth.jpg',
#              f'/Users/nicklatina/Desktop/band/output_image_{1000+i}.jpg', gradient)


#complexification but with k-means
# new_gradient = get_image_palette('/Users/nicklatina/Desktop/CLIP Prompts/Prompt Scrib5.JPG', 256)
# for grain in range(1, 20):
#     temp_gradient = generate_gradient([new_gradient[i][0] for i in range(0, 256, grain)])
#     apply_gradient('/Users/nicklatina/Desktop/CLIP Prompts/Prompt BackStairs.PNG',
#                     f'/Users/nicklatina/Desktop/palette_transfer/(scrib5 bStairs) output_image_{grain}.jpg', temp_gradient)

