
from typing import Literal
import cv2
import numpy as np


class ImageDecompressor:
    """
    Decompress CompressedImage messages using OpenCV.
    """

    Format = Literal['bayer_rggb8', 'bayer_grbg8', 'bayer_gbrg8', 'bayer_bggr8',
                     'bayer_rggb', 'bayer_grbg', 'bayer_gbrg', 'bayer_bggr',
                     'rgb8']
    
    

    def __init__(self, format:Format='bayer_rggb8'):
        """
        Initialize the decompressor with specified input
        
        Args:
            format (str): Input format (e.g., 'bayer_rggb').
        """

        self.format_from = format.lower()
    
    def __call__(self, img: np.ndarray):
        self.decompress(img)
        

    def decompress(self, img: np.ndarray) -> np.ndarray:
        """
        Decompress CompressedImage message using OpenCV.
        
        Args:
            compressed_msg: CompressedImage message
        
        Returns:
            numpy.ndarray: OpenCV image (BGR format)
        """
        
        cv_image = img
        # print(f"Input image shape: {cv_image.shape}, dtype: {cv_image.dtype}")


        # Handle different channel configurations
        # if len(cv_image.shape) == 2:
        #     # Grayscale image, convert to 3-channel BGR
        #     cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        # elif len(cv_image.shape) == 3:
        #     if cv_image.shape[2] == 4:
        #         # RGBA/BGRA image, convert to BGR
        #         cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)
        #     elif cv_image.shape[2] == 3:
        #         pass
        

        # Check for Bayer pattern in format strin
        match self.format_from:
            case 'bayer_rggb8' | 'bayer_rggb':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_RG2RGB)
            case 'bayer_grbg8' | 'bayer_grbg':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_GR2RGB)
            case 'bayer_gbrg8' | 'bayer_gbrg':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_GB2RGB)
            case 'bayer_bggr8' | 'bayer_bggr':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_BG2RGB)
            case 'rgb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            case 'bgr8':
                # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pass
            case _:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_RG2RGB)  # Default

        return cv_image
            


        

if __name__ == "__main__":
    decompressor = ImageDecompressor(format='bayer_rggb8')

    file = '/media/morita/SSD_B0084/visual-odometry/dataset/vslam-tunnel/pycuvslam/camera/front/1751261439759080017.jpg'

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    print(f"Input image shape: {img.shape}")
    out = decompressor.decompress(img)
    print(f"Output image shape: {out.shape}")
    cv2.imshow('decompressed', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
