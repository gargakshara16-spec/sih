import rasterio
import matplotlib.pyplot as plt
import numpy as np


def visualize_geotiff(path):

    with rasterio.open(path) as src:

        print("Bands:", src.count)
        print("Size:", src.width, "x", src.height)
        print("CRS:", src.crs)

        # Read the first three bands
        image = src.read([1, 2, 3]).astype(float)

        # Move bands to the last dimension
        image = image.transpose(1, 2, 0)

        # Normalize each band separately
        for i in range(3):

            band = image[:, :, i]

            min_value = np.percentile(band, 2)
            max_value = np.percentile(band, 98)

            image[:, :, i] = np.clip(
                (band - min_value) /
                (max_value - min_value),
                0,
                1
            )

        plt.figure(figsize=(10, 8))

        plt.imshow(image)

        plt.title("GeoTIFF RGB Visualization")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":

    path = input("Enter GeoTIFF path: ")

    visualize_geotiff(path)