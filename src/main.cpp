#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

int load_image(const std::string name, cv::Mat &image, int flags)
{
  image = cv::imread(name, flags);
  if (image.empty())
  {
    std::cout << "error loading " << name << std::endl;
    return -1;
  }
  return 0;
}

cv::Mat gaussian_blur_kernel(int size, float sigma)
{
  int N = size;
  if (N % 2 == 0)
  {
    N++;
  }
  cv::Mat kernel(N, N, CV_32F);
  float sum = 0;
  for (int i = 0; i < kernel.rows; i++)
  {
    for (int j = 0; j < kernel.cols; j++)
    {
      float squareDist = pow((-(N - 1) * 0.5) + i, 2) + pow((-(N - 1) * 0.5) + j, 2);
      kernel.at<float>(i, j) = exp(-squareDist / (2.0 * sigma * sigma));
      sum += kernel.at<float>(i, j);
    }
  }
  kernel /= sum;

  return kernel;
}

std::vector<cv::Mat> gaussian_histogram(const cv::Mat &image, int focus)
{
  std::vector<cv::Mat> hist(256);
  float size = 20;
  float gap = size / (focus + 1);
  for (int i = 0; i < focus; i++)
  {
    hist[i] = gaussian_blur_kernel(size - i * gap, sqrt(size - i * gap));
  }
  for (int i = focus; i < 256; i++)
  {
    hist[i] = gaussian_blur_kernel(1, 0.01);
  }
  return hist;
}

void filter(const cv::Mat &src, const cv::Mat &src_depth,
            cv::Mat &dst, int focus)
{

  dst = cv::Mat(src.rows, src.cols, CV_32FC3, cv::Scalar(0., 0., 0.));
  auto hist = gaussian_histogram(src_depth, focus);
  for (int x = 0; x < src.rows; x++)
  {
    for (int y = 0; y < src.cols; y++)
    {
      uchar current_depth = src_depth.at<uchar>(x, y);
      cv::Mat kernel = hist[current_depth];
      int N = kernel.cols;
      float sum = 0.;

      for (int i = x - N / 2; i <= x + N / 2; i++)
      {
        for (int j = y - N / 2; j <= y + N / 2; j++)
        {
          if (i > 0 && i < src.rows && j > 0 && j < src.cols)
          {
            if (src_depth.at<uchar>(i, j) <= current_depth)
            {
              dst.at<cv::Vec3f>(x, y) += src.at<cv::Vec3f>(i, j) * kernel.at<float>(i - x + N / 2, j - y + N / 2);
            }
            else
            {
              sum += kernel.at<float>(i - x + N / 2, j - y + N / 2);
            }
          }
          else
          {
            sum += kernel.at<float>(i - x + N / 2, j - y + N / 2);
          }
        }
      }
      if (sum > 0)
      {
        dst.at<cv::Vec3f>(x, y) += src.at<cv::Vec3f>(x, y) * sum;
        //dst.at<cv::Vec3f>(x, y) /= 1- sum;
      }
      else if (sum > 0.5)
      {
        dst.at<cv::Vec3f>(x, y) += dst.at<cv::Vec3f>(x, y) * sum;
      }
    }
  }
}
struct Data
{
  cv::Mat image;
  cv::Mat image_depth;
  cv::Mat blured_image;
};

void callBackKeyboard(int event, int x, int y, int flags, void *userdata)
{
  Data *data = (Data *)userdata;
  uint8_t focus = data->image_depth.at<uint8_t>(y, x);
  switch (event)
  {
  case cv::EVENT_LBUTTONDOWN:
    filter(data->image, data->image_depth, data->blured_image, focus);
    break;
  case cv::EVENT_RBUTTONDOWN:
  case cv::EVENT_MBUTTONDOWN:
  case cv::EVENT_MOUSEMOVE:
    break;
  }
}

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cout << "usage: " << argv[0] << " image image-depth" << std::endl;
    return -1;
  }

  Data data;
  if (load_image(argv[1], data.image, 1) == -1)
  {
    return -1;
  }
  if (load_image(argv[2], data.image_depth, cv::IMREAD_GRAYSCALE) == -1)
  {
    return -1;
  }

  cv::Mat float_image;
  data.image.convertTo(float_image, CV_32FC3, 1. / 255.);
  data.image = float_image;

  std::string window_name = "RGBD image";
  cv::namedWindow(window_name);
  cv::setMouseCallback(window_name, callBackKeyboard, &data);

  filter(data.image, data.image_depth, data.blured_image, 0);

  bool quit = false;
  while (!quit)
  {
    cv::imshow(window_name, data.blured_image);
    int key = cv::waitKey(500) % 256;
    if (key == 27 || key == 'q')
      quit = true;
  }
  cv::waitKey();

  return 0;
}
