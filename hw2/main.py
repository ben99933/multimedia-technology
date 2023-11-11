import cv2
import numpy as np
import tqdm


def convolution(image, kernel, stride=1):
    image = np.array(image, dtype=float)
    # [h, w, channel] = image.shape
    h = image.shape[0]
    w = image.shape[1]
    k = kernel.shape[0]
    r = int(k // 2)

    # padding 0
    image_padding = np.zeros([h + k - 1, w + k - 1], dtype=float)
    image_padding[r:h + r, r:w + r] = image

    result = np.zeros(image.shape)
    for i in tqdm.tqdm(range(r, h + r, stride)):
        for j in range(r, w + r, stride):
            split = image_padding[i - r:i + r + 1, j - r:j + r + 1]
            result[i - r, j - r] = np.sum(split * kernel)
    return result


def gaussian_blur(image, kernel_size: int, sigma: float = 1.0):
    kernel = np.outer(cv2.getGaussianKernel(kernel_size, sigma), cv2.getGaussianKernel(kernel_size, sigma))
    # print("kernel=", kernel)
    # result = convolution(image, kernel)
    r = convolution(image[:, :, 0], kernel)
    g = convolution(image[:, :, 1], kernel)
    b = convolution(image[:, :, 2], kernel)
    # 其實實際上是BGR
    result = np.dstack((r,g,b))
    return result


def toGray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def unsharp(image):
    gray = toGray(image)
    # print(gray)
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    result = convolution(gray, kernel)
    # print(result)
    # cv2.imshow("unsharp", result)
    # if cv2.waitKey(0) == 27: cv2.destroyAllWindows()
    return result

# using sobel
def edgeDetection(image):
    gray = toGray(image)
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    result1 = convolution(gray, kernel)
    kernel = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    result2 = convolution(gray, kernel)
    result = np.sqrt(result1**2 + result2**2)
    return result

def main():
    image = cv2.imread("./柴犬飛飛.jpg")
    # print(image)
    if image is None:
        print("No image 柴犬飛飛.jpg")
        return -1

    # 輸出圖片
    cv2.imwrite("3x3.jpg", gaussian_blur(image, 3))
    cv2.imwrite("7x7.jpg", gaussian_blur(image, 7))
    cv2.imwrite("11x11.jpg", gaussian_blur(image, 11))

    cv2.imwrite("sigma1.jpg", gaussian_blur(image, 3, 1))
    cv2.imwrite("sigma10.jpg", gaussian_blur(image, 3, 10))
    cv2.imwrite("sigma30.jpg", gaussian_blur(image, 3, 30))

    # sharp
    cv2.imwrite("unsharp.jpg", unsharp(image))
    cv2.imwrite("unsharp_gaussian3.jpg", unsharp(cv2.imread("3x3.jpg")))
    cv2.imwrite("unsharp_gaussian7.jpg", unsharp(cv2.imread("7x7.jpg")))
    cv2.imwrite("unsharp_gaussian11.jpg", unsharp(cv2.imread("11x11.jpg")))

    cv2.imwrite("edgeDetection.jpg", edgeDetection(image))
    cv2.imwrite("edgeDetection_gaussian3.jpg", edgeDetection(cv2.imread("3x3.jpg")))
    cv2.imwrite("edgeDetection_gaussian7.jpg", edgeDetection(cv2.imread("7x7.jpg")))
    cv2.imwrite("edgeDetection_gaussian11.jpg", edgeDetection(cv2.imread("11x11.jpg")))

    # PSNR
    print(f'3x3.png PSNR: {cv2.PSNR(image, cv2.imread("3x3.jpg"))}')
    print(f'7x7.png PSNR: {cv2.PSNR(image, cv2.imread("7x7.jpg"))}')
    print(f'11x11.png PSNR: {cv2.PSNR(image, cv2.imread("11x11.jpg"))}')

    print(f'sigma1.png PSNR: {cv2.PSNR(image, cv2.imread("sigma1.jpg"))}')
    print(f'sigma10.png PSNR: {cv2.PSNR(image, cv2.imread("sigma10.jpg"))}')
    print(f'sigma30.png PSNR: {cv2.PSNR(image, cv2.imread("sigma30.jpg"))}')

    print(f'unsharp.png PSNR: {cv2.PSNR(image, cv2.imread("unsharp.jpg"))}')
    print(f'edgeDetection.png PSNR: {cv2.PSNR(image, cv2.imread("edgeDetection.jpg"))}')


if __name__ == "__main__":
    main()
