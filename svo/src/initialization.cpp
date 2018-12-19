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

#include <svo/config.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo {
namespace initialization {

/********************************
 * @ function: 插入第一帧，特征足够则作为参考帧，并且得到跟踪的特征点
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  reset();
  //[***step 1***] 把第一帧作为参考帧，检测(跟踪)特征点px_ref_，得到它的归一化平面上的向量f_ref_
  detectFeatures(frame_ref, px_ref_, f_ref_);
  //特征少则失败
  if(px_ref_.size() < 100)
  {
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }
  //[***step 2***]特征足够，作为参考帧，并且插入其特征点到当前跟踪的特征点
  // px_ref_参考帧上要跟踪的点
  // px_cur_当前帧已经跟踪上的点
  frame_ref_ = frame_ref;
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
  return SUCCESS;
}

/********************************
 * @ function:
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
  //[***step 1***] 使用KLT在参考帧与当前帧之间进行角点跟踪
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");

  // 跟踪的角点太少
  if(disparities_.size() < Config::initMinTracked())
    return FAILURE;
  //特征点移动的距离的太小，则不是关键帧
  double disparity = vk::getMedian(disparities_); //运动方向中值，方向由当前帧特征点指向参考帧特征点
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
  if(disparity < Config::initMinDisparity())
    return NO_KEYFRAME;

  //[***step 2***] 根据两帧计算单应矩阵，恢复T_cur_from_ref_，以及三角化的点xyz_in_cur_
  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);
  SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");
  // 内点数量少则初始化失败
  if(inliers_.size() < Config::initMinInliers())
  {
    SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
    return FAILURE;
  }

  // 计算尺度这里不是很理解？？？
  // Rescale the map such that the mean scene depth is equal to the specified scale
  // 把深度提取出来
  vector<double> depth_vec;
  for(size_t i=0; i<xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  
  double scene_depth_median = vk::getMedian(depth_vec); //深度的中值
  double scale = Config::mapScale()/scene_depth_median; //得到尺度？？？
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_; //求得当前帧的变换矩阵，这里的ref位姿如何得到的？？？
  // 求得平移变量
  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));

  // For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();
  for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
  {
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
    if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
    {
      Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
      Point* new_point = new Point(pos);

      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
      frame_cur->addFeature(ftr_cur);
      new_point->addFrameRef(ftr_cur);

      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
      frame_ref_->addFeature(ftr_ref);
      new_point->addFrameRef(ftr_ref);
    }
  }
  return SUCCESS;
}

void KltHomographyInit::reset()
{
  px_cur_.clear();
  frame_ref_.reset();
}

/********************************
 * @ function: 对新的一帧提取特征点
 * 
 * @ param:   FramePtr frame,
 *            vector<cv::Point2f>& px_vec,
 *            vector<Vector3d>& f_vec
 * 
 * @ note: 这里的px_vec是减去cx, cy的
 *******************************/
void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec)
{
  Features new_features;
  feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

  // - f_vec是特征点经过相机光心反投影cam2world()
  // - (X,Y,Z) = ((u - cx)/fx, (v - cy)/fy, 1.0)
  // now for all maximum corners, initialize a new seed
  px_vec.clear(); px_vec.reserve(new_features.size());
  f_vec.clear(); f_vec.reserve(new_features.size());
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
    f_vec.push_back(ftr->f); 
    delete ftr;
  });
}

/********************************
 * @ function:
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void trackKlt(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities)
{
  const double klt_win_size = 30.0; //搜索窗口的大小
  const int klt_max_iter = 30;
  const double klt_eps = 0.001;
  vector<uchar> status;
  vector<float> error;
  vector<float> min_eig_vec;
  // 迭代终止条件
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
  // 参数：status为特征点是否被找到，error为找到点的测量误差
  cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                           px_ref, px_cur,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size),
                           4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  f_cur.clear(); f_cur.reserve(px_cur.size());
  disparities.clear(); disparities.reserve(px_cur.size());

  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    // 没找到则删除
    if(!status[i])
    {
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }
    // 转化为归一化平面上的点
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
  // 计算ref和cur之间移动的大小，光流向量大小
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
  }
}

/********************************
 * @ function: 通过H矩阵分解得到SE3，并三角化剔除外点
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void computeHomography(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref)
{
  vector<Vector2d > uv_ref(f_ref.size());
  vector<Vector2d > uv_cur(f_cur.size());
  //[***step 1***] 三维世界坐标投影到归一化平面，这里已经减去偏移量？？？
  for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
  {
    uv_ref[i] = vk::project2d(f_ref[i]);
    uv_cur[i] = vk::project2d(f_cur[i]);
  }

  //[***step 2***] 构造单应类，从单应矩阵恢复R,t
  //单应矩阵类构造
  vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
  // 通过分解H得到T
  Homography.computeSE3fromMatches();
  
  //[***step 3***] 把特征点三角化计算重投影误差，决定内点还是外点
  // 注意这里的点的顺序变了，先cur后ref
  vector<int> outliers;
  vk::computeInliers(f_cur, f_ ref,
                     Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                     reprojection_threshold, focal_length,
                     xyz_in_cur, inliers, outliers);
  T_cur_from_ref = Homography.T_c2_from_c1;
}


} // namespace initialization
} // namespace svo
