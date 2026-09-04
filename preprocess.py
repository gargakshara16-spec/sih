import rasterio
import numpy as np
from PIL import Image


def preprocess_geotiff(path, output_path):

    with rasterio.open(path) as src:

        # Read RGB bands
        image = src.read([1, 2, 3]).astype(np.float32)

        # Convert:
        # (3, height, width)
        # to:
        # (height, width, 3)
        image = np.transpose(image, (1, 2, 0))

        # Normalize pixel values
        min_value = image.min()
        max_value = image.max()

        image = (image - min_value) / (max_value - min_value)

        # Convert to 8-bit image
        image = (image * 255).astype(np.uint8)

        # Save as PNG
        Image.fromarray(image).save(output_path)

        print("Preprocessing complete!")
        print("Input:", path)
        print("Output:", output_path)
        print("Image shape:", image.shape)


if __name__ == "__main__":

    input_path = input("Enter GeoTIFF path: ")
    output_path = "preprocessed.png"

    preprocess_geotiff(input_path, output_path)