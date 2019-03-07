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
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <svo/depth_filter.h>
#ifdef USE_BUNDLE_ADJUSTMENT // 这玩意在哪定义的呢
#include <svo/bundle_adjustment.h>
#endif

namespace svo {
// 构造函数
FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam) :
  FrameHandlerBase(),  //基类
  cam_(cam), //相机模型
  reprojector_(cam_, map_), //重投影类, map_来自基类
  depth_filter_(NULL)  //深度滤波器
{
  initialize();
}
//* 初始化函数
void FrameHandlerMono::initialize()
{
//[ ***step 1*** ] 初始化fast角点检测类共享指针, 子类的fastdetector赋值给基类共享指针
  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
//[ ***step 2*** ] 声明 depthfilter 的回调函数, 用来构建地图点
  //* depth_filter_cb（point*, sigma）参数传入map_.point_candidates_->newCandidatePoint(_1, _2 )
  //* bind的第一个参数为对象成员函数, 第二个为该对象, 后面是参数
  //? 多线程常用回调函数?
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
//[ ***step 3*** ] 创建深度滤波类, 并启动深度滤波
  depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
  depth_filter_->startThread();
}
//* 析构函数
FrameHandlerMono::~FrameHandlerMono()
{
  delete depth_filter_;
}

/********************************
 * @ function: 增加一帧, 根据状态进行处理[第一帧, 第二帧, 重定位, 默认帧]
 * 
 * @ param: 
 * 
 * @ note:  关于 stage_ 和 set_start_ 需要看一看
 *******************************/
void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp)
{
//[ ***step 1*** ] 判断 stage_, 开始计时, 并且清空trash
  if(!startFrameProcessingCommon(timestamp))
    return;

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();      //!< 相近的关键帧
  overlap_kfs_.clear();   //!< 第一个参数具有一定共视关系的关键帧, 第二个参数是该帧观察到多少个地图点

  // create new frame
//[ ***step 2*** ] 创建一个新的帧, 会构建图像金字塔
  SVO_START_TIMER("pyramid_creation");
  //* 这个 reset 是共享指针的
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp)); // 共享指针的 reset
  SVO_STOP_TIMER("pyramid_creation");

//[ ***step 3*** ] 根据stage_确定如何处理
  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    res = processFrame();
  else if(stage_ == STAGE_SECOND_FRAME)
    res = processSecondFrame();
  else if(stage_ == STAGE_FIRST_FRAME)
    res = processFirstFrame();
  else if(stage_ == STAGE_RELOCALIZING)
    res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));

  // set last frame
  last_frame_ = new_frame_;
  new_frame_.reset();
  // finish processing
//[ ***step 4*** ] 完成一帧的处理,记录参数, 根据返回的状态进行设置
//* last_frame_->nObs() 就是 fts.size()
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
}

//@ 处理第一帧
FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
//[ ***step 1*** ] 在初始化类中加入一帧, 若初始化不成功则返回无关键帧状态
  if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;
//[ ***step 2*** ] 初始化成功, 加入关键加入地图, 等待第二帧 
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_SECOND_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}

//@ 处理第二帧
FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
//[ ***step 1*** ] 第二帧加入到初始化类
  initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);
  if(res == initialization::FAILURE)
    return RESULT_FAILURE;
  else if(res == initialization::NO_KEYFRAME)
    return RESULT_NO_KEYFRAME;
//[ ***step 2*** ] 把第一帧和第二帧一起做BA优化
  // two-frame bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
#endif
//[ ***step 3*** ] 设置第二帧为关键帧, 并计算中位深度, 最小深度, 加入到深度滤波中
  new_frame_->setKeyframe();
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);
//[ ***step 4*** ] 将第二帧作为关键帧加入到地图中, 重置初始化的类, 
  // add frame to map
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  klt_homography_init_.reset();
  SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
  return RESULT_IS_KEYFRAME;
}

//@ 处理正常的一帧
FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
//[ ***step 1*** ] 将上一帧的位姿设置为这一帧初始位姿  
  // Set initial pose TODO use prior
  new_frame_->T_f_w_ = last_frame_->T_f_w_;
//[ ***step 2*** ] 配置稀疏图像对齐, 并运行得到粗略位姿
  // sparse image align
  SVO_START_TIMER("sparse_img_align");
  //* 设置了从 kltMaxLevel 到 kltMinLevel 的图像对齐(粗到精)
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_); 
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);
//[ ***step 3*** ] 将关键帧上的3D点, 候选地图点投影到当前帧, 并进行特征匹配得到精确地特征点位置(每个cell里一个)
  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references = reprojector_.n_matches_; // 匹配的特征点
  const size_t repr_n_mps = reprojector_.n_trials_;              // 尝试匹配的次数(点数)
  SVO_LOG2(repr_n_mps, repr_n_new_references);
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
  //* 如果匹配的点少, 则还是使用上一帧的位姿(匹配点少, 则不可信)
  if(repr_n_new_references < Config::qualityMinFts())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;
    return RESULT_FAILURE;
  }
//[ ***step 4*** ] 对当前帧进行位姿优化 (该帧上的重投影误差)
//*经过之前的特征对齐, 特征点的位置变了, 因此要进行这个单帧位姿的优化
  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final; // 误差观测数目
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
  //* 如果观测的点过少也不可信
  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;
//[ ***step 5*** ] 优化了当前帧位姿位姿之后, 再向其他的观测帧进行重投影误差优化3D位置点
  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  // select keyframe
//[ ***step 6*** ] 加入到 core_kfs 中用来 localBA, 根据 1.特征点数目 2.特征减少的数目, 判断跟踪质量, 筛选关键帧
  core_kfs_.insert(new_frame_);  //!!!
  setTrackingQuality(sfba_n_edges_final);
  //* 跟踪质量不满足则用上一帧, 直接返回
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    return RESULT_FAILURE;
  }
  //* 通过位姿变换到当前帧, 计算深度
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  //* 如果共视的特征点与平均深度的距离都小于一点数值则不需要关键帧( 离得比较近 )
  if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  {
    // 加入到深度滤波线程
    depth_filter_->addFrame(new_frame_);
    return RESULT_NO_KEYFRAME;
  }
//[ ***step 7*** ] 满足上面的条件就设置成关键帧(提取5个关键点), 把该帧加入到point的参考帧, 以及把地图点加入参考帧
  new_frame_->setKeyframe();
  SVO_DEBUG_STREAM("New keyframe selected.");

  // new keyframe selected
  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
      (*it)->point->addFrameRef(*it);
  //? 这个怎么又加入到 frame 里投影时候不是加过???
  //?答: 投影时候增加的是原来的地图点, 加入深度滤波之后又重新提取了特征点, 因此这是新的
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

//[ ***step 8*** ] 如果定义了BA, 就使用 localBA 对关键帧们进行优化
  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif
//[ ***step 9*** ] 关键帧加入到深度滤波线程, 会重新提取特征点, 进行计算深度直到收敛
  // init new depth-filters
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  //* 如果地图中关键帧过多, 则要删除一些, 清理深度滤波的种子点, 地图中的点
  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

//[ ***step 10*** ] 增加到关键帧中
  // add keyframe to map
  map_.addKeyframe(new_frame_);

  return RESULT_IS_KEYFRAME;
}

//@ 重定位, 参数: 初始位姿, 参考关键帧(离上一帧最近的关键帧)
FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
    const SE3& T_cur_ref,
    FramePtr ref_keyframe)
{
  SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
  if(ref_keyframe == nullptr)
  {
    SVO_INFO_STREAM("No reference keyframe.");
    return RESULT_FAILURE;
  }
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
//[ ***step 1*** ] 进行图像对齐, 返回特征点数
  size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
//[ ***step 2*** ] 特征点数大于30, 就把最近帧作为上一帧进行处理
  if(img_align_n_tracked > 30)
  {
    SE3 T_f_w_last = last_frame_->T_f_w_;
    last_frame_ = ref_keyframe;

    FrameHandlerMono::UpdateResult res = processFrame();
//[ ***step 3*** ] 定位成功则进入常规的跟踪阶段, 否则新的一帧位姿设置为参考帧的位姿
    if(res != RESULT_FAILURE)
    {
      stage_ = STAGE_DEFAULT_FRAME;
      SVO_INFO_STREAM("Relocalization successful.");
    }
    else
      new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
    return res;
  }
  return RESULT_FAILURE;
}

bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
  FramePtr ref_keyframe;
  //* 通过ID在map中找出 frame
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  //* 根据输入的图像和时间戳重置图像  
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  //* 用这一帧去重定位
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  //* 重定位成功, 则下一帧
  if(res != RESULT_FAILURE) {
    last_frame_ = new_frame_;
    return true;
  }
  return false;
}

//* 重置
void FrameHandlerMono::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  core_kfs_.clear();
  overlap_kfs_.clear();
  depth_filter_->reset();
}

//* 设置第一帧
void FrameHandlerMono::setFirstFrame(const FramePtr& first_frame)
{
  resetAll();
  last_frame_ = first_frame;
  last_frame_->setKeyframe();
  map_.addKeyframe(last_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
}

//* 根据共视的点与场景平均深度的距离大小, 判断是否需要关键帧
bool FrameHandlerMono::needNewKf(double scene_depth_mean)
{
  for(auto it=overlap_kfs_.begin(), ite=overlap_kfs_.end(); it!=ite; ++it)
  {
    Vector3d relpos = new_frame_->w2f(it->first->pos());
    if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist()*0.8 &&
       fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist()*1.3)
      return false;
  }
  return true;
}

//* 从 overlap_kfs 中挑出最好的几帧作为 core_kfs
void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size()-1);
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
}

} // namespace svo
