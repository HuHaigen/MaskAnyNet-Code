import random
import numpy as np
import cv2


def extract_and_stitch_single_patches(image, patch_scale_size):
    # Ensure the input is a NumPy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Get image dimensions
    height, width, _ = image.shape
    mask = np.ones((height, width), dtype=np.uint8) * 255

    scale = patch_scale_size

    # Calculate patch size
    patch_height = int(height * scale)
    patch_width = int(width * scale)

    # Randomly determine the start point
    y_start = random.randint(0, height - patch_height)
    x_start = random.randint(0, width - patch_width)

    y_end = y_start + patch_height
    x_end = x_start + patch_width

    # Set the corresponding area in the mask to 0 (unselected)
    mask[y_start:y_end, x_start:x_end] = 0

    # Extract the patch from the image
    patch = image[y_start:y_end, x_start:x_end, :]

    # Since only one patch is extracted, directly assign it to stitched_image
    stitched_image = patch

    # Apply the mask to the original image (this will affect areas outside the patch)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image, stitched_image


def extract_and_stitch_random_patches(image, n, scale=0.25):
    # Ensure the input is a NumPy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    h, w, _ = image.shape
    block_height = h // n
    block_width = w // n

    # Create a binary mask matrix (n x n)
    mask = np.zeros((n, n), dtype=np.uint8)

    # Each block has a certain probability of being masked
    for i in range(n):
        for j in range(n):
            # Each block has (scale * scale) chance to be masked
            mask[i, j] = 1 if random.random() < (scale * scale) else 0

    # Construct the final mask
    final_mask = np.zeros_like(image)

    for i in range(n):
        for j in range(n):
            if mask[i, j] == 1:
                final_mask[i * block_height:(i + 1) * block_height,
                           j * block_width:(j + 1) * block_width] = 255
            else:
                final_mask[i * block_height:(i + 1) * block_height,
                           j * block_width:(j + 1) * block_width] = 0

    # Apply mask to image
    masked_image = cv2.bitwise_and(image, image, mask=(255 - final_mask[:, :, 0]))

    # Initialize the stitched image (e.g., resized image from masked regions)
    stitched_image = np.zeros((int(h * scale), int(w * scale), 3), dtype=np.uint8)

    # Start position for stitched image
    stitch_row = 0
    stitch_col = 0

    # Only stitch masked blocks
    for i in range(n):
        for j in range(n):
            if mask[i, j] == 1:
                block = image[i * block_height:(i + 1) * block_height,
                              j * block_width:(j + 1) * block_width]

                stitched_row_start = stitch_row * block_height
                stitched_row_end = (stitch_row + 1) * block_height
                stitched_col_start = stitch_col * block_width
                stitched_col_end = (stitch_col + 1) * block_width

                if stitched_row_end <= stitched_image.shape[0] and stitched_col_end <= stitched_image.shape[1]:
                    stitched_image[stitched_row_start:stitched_row_end,
                                   stitched_col_start:stitched_col_end] = block

                stitch_col += 1
                if stitch_col >= int(n * scale):
                    stitch_col = 0
                    stitch_row += 1

    return masked_image, stitched_image


def extract_and_stitch_grid_patches(image, grid_size, stride):
    # grid_size and stride are lists containing height and width respectively.
    # This allows handling non-square images.

    # Ensure the input is a NumPy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Get image dimensions
    height, width, _ = image.shape
    mask = np.ones((height, width), dtype=np.uint8) * 255

    # Calculate number of rows and columns in the grid
    num_rows = max(1, (height - grid_size[0] + int(stride[0])) // int(stride[0]))
    num_cols = max(1, (width - grid_size[1] + int(stride[1])) // int(stride[1]))

    # Initialize the stitched image
    stitched_height = grid_size[0] * num_rows
    stitched_width = grid_size[1] * num_cols
    stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

    # Loop through each grid cell and extract patches
    for i in range(num_rows):
        for j in range(num_cols):
            y_start = i * int(stride[0])
            y_end = min(y_start + grid_size[0], height)
            x_start = j * int(stride[1])
            x_end = min(x_start + grid_size[1], width)

            # Set the selected region in the mask to 0 (unselected)
            mask[y_start:y_end, x_start:x_end] = 0

            patch = image[y_start:y_end, x_start:x_end, :]

            # Copy the patch into the correct location in stitched_image
            row_start = i * grid_size[0]
            col_start = j * grid_size[1]
            stitched_image[row_start:row_start + (y_end - y_start),
                           col_start:col_start + (x_end - x_start), :] = patch

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image, stitched_image
