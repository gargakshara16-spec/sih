import rasterio
import numpy as np


def inspect_geotiff(path):

    print("\n===================================")
    print("       GeoTIFF INFORMATION")
    print("===================================")

    with rasterio.open(path) as src:

        print("File:", path)
        print("Width:", src.width)
        print("Height:", src.height)
        print("Bands:", src.count)
        print("CRS:", src.crs)

        print("Transform:")
        print(src.transform)

        print("Resolution:", src.res)

        print("Bounds:")
        print(src.bounds)

        print("Data type:", src.dtypes[0])

        print("===================================")

        return {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": src.crs,
            "transform": src.transform,
            "resolution": src.res,
            "bounds": src.bounds
        }


def read_rgb_geotiff(path):

    with rasterio.open(path) as src:

        image = src.read([1, 2, 3])

        image = np.transpose(image, (1, 2, 0))

        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds
        }

    return image, metadata


if __name__ == "__main__":

    path = input("Enter the path to your GeoTIFF file: ")

    inspect_geotiff(path)