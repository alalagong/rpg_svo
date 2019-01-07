// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <stdexcept>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/config.h>
#include <boost/bind.hpp>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/performance_monitor.h>
#include <fast/fast.h>

namespace svo {

int Frame::frame_counter_ = 0; //静态成员变量

/********************************
 * @ function: frame构造函数
 * 
 * @ param:   cam   相机模型
 *            img   图像
 *            timestamp   时间戳
 * @ note:
 *******************************/
Frame::Frame(vk::AbstractCamera* cam, const cv::Mat& img, double timestamp) :
    id_(frame_counter_++), // 帧独一无二的ID
    timestamp_(timestamp),  
    cam_(cam), 
    key_pts_(5), //5个关键特征点及相应的3D点，用于判断是否两帧具有重叠视野
    is_keyframe_(false),
    v_kf_(NULL) //g2o的节点的临时指针
{
  initFrame(img);
}

Frame::~Frame()
{
  // 帧上的特征点要删除
  std::for_each(fts_.begin(), fts_.end(), [&](Feature* i){delete i;});
}

/********************************
 * @ function: 初始化frame，检查图像，建立图像金字塔
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void Frame::initFrame(const cv::Mat& img)
{
  // check image
  // 判断图像有效性
  if(img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height())
    throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");

  // Set keypoints to NULL
  // 初始化为空指针
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature* ftr){ ftr=NULL; });

  // Build Image Pyramid
  // 创建金字塔，得到一系列图像
  frame_utils::createImgPyramid(img, max(Config::nPyrLevels(), Config::kltMaxLevel()+1), img_pyr_);
}
// 设置为关键帧
void Frame::setKeyframe()
{
  is_keyframe_ = true;
  setKeyPoints();
}
// 增加图像特征点
void Frame::addFeature(Feature* ftr)
{
  fts_.push_back(ftr);
}

/********************************
 * @ function: 设置最合适的key_pts_
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void Frame::setKeyPoints()
{
//[***step 1***] 删除没有3D点的feature
  for(size_t i = 0; i < 5; ++i)
    if(key_pts_[i] != NULL)
      if(key_pts_[i]->point == NULL)
        key_pts_[i] = NULL;
//[***step 2***] 从fts_中挑出合适的5个点放入key_pts_
  std::for_each(fts_.begin(), fts_.end(), [&](Feature* ftr){ if(ftr->point != NULL) checkKeyPoints(ftr); });
}

/********************************
 * @ function: 挑选距离图像中心较远的四个象限里的4个点
 *              和图像中心较近的一个点
 * 
 * @ param: 
 * 
 * @ note: 感觉代码在2,4象限上的点的计算上有问题，不能是负的 ？？？
 *         以及比较的对象有问题，更改 By gong
 *******************************/
void Frame::checkKeyPoints(Feature* ftr)
{
  const int cu = cam_->width()/2;
  const int cv = cam_->height()/2;

  // center pixel
  // 如果空，则把当前的赋值
  if(key_pts_[0] == NULL)
    key_pts_[0] = ftr;
//[***step 1***] 挑选距离中心近的
  else if(std::max(std::fabs(ftr->px[0]-cu), std::fabs(ftr->px[1]-cv))
        < std::max(std::fabs(key_pts_[0]->px[0]-cu), std::fabs(key_pts_[0]->px[1]-cv)))
    key_pts_[0] = ftr;
//[***step 2***] 挑选右下角的点
  if(ftr->px[0] >= cu && ftr->px[1] >= cv)
  {
    if(key_pts_[1] == NULL)
      key_pts_[1] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[1]->px[0]-cu) * (key_pts_[1]->px[1]-cv))
      key_pts_[1] = ftr;
  }
//[***step 3***] 挑选右上角的
  // bug:  (ftr->px[1]-cv) 会小于零
  if(ftr->px[0] >= cu && ftr->px[1] < cv)
  {
    if(key_pts_[2] == NULL)
      key_pts_[2] = ftr;
    // else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
    else if((ftr->px[0]-cu) * (cv-ftr->px[1])
          // > (key_pts_[2]->px[0]-cu) * (key_pts_[2]->px[1]-cv))
          > (key_pts_[2]->px[0]-cu) * (cv-key_pts_[2]->px[1]))
      key_pts_[2] = ftr;
  }
//[***step 4***] 挑选左上角
  // bug：应该与cu比较
  // if(ftr->px[0] < cv && ftr->px[1] < cv)
  if(ftr->px[0] < cu && ftr->px[1] < cv)
  {
    if(key_pts_[3] == NULL)
      key_pts_[3] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[3]->px[0]-cu) * (key_pts_[3]->px[1]-cv))
      key_pts_[3] = ftr;
  }
//[***step 5***] 挑选左下角
  // bug1: 应该与cu比较
  // bug2：(ftr->px[0]-cu)会小于零
  if(ftr->px[0] < cu && ftr->px[1] >= cv)
  // if(ftr->px[0] < cv && ftr->px[1] >= cv)
  {
    if(key_pts_[4] == NULL)
      key_pts_[4] = ftr;
    // else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
    //       > (key_pts_[4]->px[0]-cu) * (key_pts_[4]->px[1]-cv))
    else if(cu-(ftr->px[0]) * (ftr->px[1]-cv)
          > (cu-key_pts_[4]->px[0]) * (key_pts_[4]->px[1]-cv))      
      key_pts_[4] = ftr;
  }
}

/********************************
 * @ function: 从5个keyPoints中删除指定的feature
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void Frame::removeKeyPoint(Feature* ftr)
{
  bool found = false;
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature*& i){
    if(i == ftr) {
      i = NULL;
      found = true;
    }
  });
  if(found)
    setKeyPoints();
}

/********************************
 * @ function: 给定的3D点在当前帧是否可见
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
bool Frame::isVisible(const Vector3d& xyz_w) const
{
  Vector3d xyz_f = T_f_w_*xyz_w;
  // 深度为正
  if(xyz_f.z() < 0.0)
    return false; // point is behind the camera
  Vector2d px = f2c(xyz_f);
  // 在图像内
  if(px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width() && px[1] < cam_->height())
    return true;
  return false;
}


/// Utility functions for the Frame class
namespace frame_utils {

void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;
  // 每次下采样1/4
  for(int i=1; i<n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
    vk::halfSample(pyr[i-1], pyr[i]);
  }
}

/********************************
 * @ function: 得到frame帧上特征点深度的中位数和最小深度
 * 
 * @ param: 
 * 
 * @ note: 
 *******************************/
bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min)
{
  vector<double> depth_vec;
  depth_vec.reserve(frame.fts_.size());
  depth_min = std::numeric_limits<double>::max(); //最大的可存储
  for(auto it=frame.fts_.begin(), ite=frame.fts_.end(); it!=ite; ++it)
  {
    if((*it)->point != NULL) //特征点有对应的3d点
    {
      // 3D点转换到当前帧坐标系下，并取深度
      const double z = frame.w2f((*it)->point->pos_).z();
      depth_vec.push_back(z);
      depth_min = fmin(z, depth_min); //返回z和depth_min中小的那个
    }
  }
  if(depth_vec.empty())
  {
    SVO_WARN_STREAM("Cannot set scene depth. Frame has no point-observations!");
    return false;
  }
  depth_mean = vk::getMedian(depth_vec);
  return true;
}

} // namespace frame_utils
} // namespace svo
