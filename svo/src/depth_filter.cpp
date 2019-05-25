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

#include <algorithm>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <svo/global.h>
#include <svo/depth_filter.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/matcher.h>
#include <svo/config.h>
#include <svo/feature_detection.h>

namespace svo {

//* static量
int Seed::batch_counter = 0;
int Seed::seed_counter = 0;

//* seed的构造函数, 注意使用的是逆深度
Seed::Seed(Feature* ftr, float depth_mean, float depth_min) :
    batch_id(batch_counter),
    id(seed_counter++),           // 每次新建一个种子就加一
    live_time(0),
    ftr(ftr),
    a(10),
    b(10),
    mu(1.0/depth_mean),           // 逆深度的均值
    z_range(1.0/depth_min),       // 逆深度的最大值
    sigma2(z_range*z_range/36)    // 99% 的概率在这个区间的协方差? 为啥这么求
{
}
Seed::~Seed() 
{
  if(live_time != 0)
  {
    // if(!DepthFilter::f.is_open())
    //   std::cout<<"!!!!!!"<<std::endl;   
    // DepthFilter::f << "No. "<< id << "seed live " << live_time << " frames"<<endl;
    std::cout<< "No. "<< id << " seed live " << live_time << " frames"<<endl;
  }
}

//* 深度滤波器的构造函数
DepthFilter::DepthFilter(feature_detection::DetectorPtr feature_detector, callback_t seed_converged_cb) :
    feature_detector_(feature_detector),
    seed_converged_cb_(seed_converged_cb),      //? 收敛
    seeds_updating_halt_(false),
    thread_(NULL),
    new_keyframe_set_(false),
    new_keyframe_min_depth_(0.0),
    new_keyframe_mean_depth_(0.0)             //? 最大值? 均值?
{
  std::string file = "/home/gong/catkin_ws/src/rpg_svo/seeds.txt";
  f.open(file.c_str());
}
//* 析构函数 
DepthFilter::~DepthFilter()
{
  f.close();
  stopThread();
  SVO_INFO_STREAM("DepthFilter destructed.");
}

void DepthFilter::startThread()
{
  //* 启动方式类似bind  !参数列表(成员函数, 对象实例, 参数)
  thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
}

void DepthFilter::stopThread()
{
  SVO_INFO_STREAM("DepthFilter stop thread invoked.");
  if(thread_ != NULL)
  {
    SVO_INFO_STREAM("DepthFilter interrupt and join thread... ");
    seeds_updating_halt_ = true; // 种子点更新暂停
    thread_->interrupt(); // 线程暂停
    thread_->join();      // 等待子线程执行完成返回
    thread_ = NULL;       // 删除线程
  }
}

//* 增加一帧到队列, 保证只有两帧
void DepthFilter::addFrame(FramePtr frame)
{
  if(thread_ != NULL)
  {
  {
      lock_t lock(frame_queue_mut_);
      if(frame_queue_.size() > 2)
        frame_queue_.pop();  // 大于两个则弹出,队首
      frame_queue_.push(frame);  //放入队尾
    }
    seeds_updating_halt_ = false;  // 种子点更新不停止
    frame_queue_cond_.notify_one(); // 启用, 其 wait 的线程
  }
  else
    updateSeeds(frame); //! 不使用多线程时候的更新
}

//* 增加一新的关键帧
//? 这里的传入参数 depth 是哪里来的???
//?答: 这里的两个深度是根据关键帧上其它点的深度计算得到, 
//?    因为SVO是面向无人机下视, 因此场景深度基本是单一平面,使用平均深度可以作为较好的初值
void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min)
{
  new_keyframe_min_depth_ = depth_min;
  new_keyframe_mean_depth_ = depth_mean;
  if(thread_ != NULL)
  {
    new_keyframe_ = frame;
    new_keyframe_set_ = true;  // 有新的关键帧要处理
    seeds_updating_halt_ = true;  // 种子点更新停止
    frame_queue_cond_.notify_one();  // 唤醒挂起线程
  }
  else
    initializeSeeds(frame);  //! 不使用多线程
}

/********************************
 * @ function: 初始化种子点
 * 
 * @ param:  关键帧的共享指针
 * 
 * @ note:
 *******************************/
void DepthFilter::initializeSeeds(FramePtr frame)
{
  Features new_features;
  //[ ***step 1*** ] 将已经有特征点的网格设置为占据 
  feature_detector_->setExistingFeatures(frame->fts_); // 将有点的网格设置为占据
  //[ ***step 2*** ] 提取新的特征点
  feature_detector_->detect(frame.get(), frame->img_pyr_,
                            Config::triangMinCornerScore(), new_features);

  // initialize a seed for every new feature
  //[ ***step 3*** ] 暂停更新种子点, 上线程锁
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_); // by locking the updateSeeds function stops
  ++Seed::batch_counter;  // batch计数增加
  //[ ***step 4*** ] 增加种子点到列表中, 种子点都是新提取的点!!!
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    seeds_.push_back(Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
  });

  if(options_.verbose)
    SVO_INFO_STREAM("DepthFilter: Initialized "<<new_features.size()<<" new seeds");
  //[ ***step 5*** ] 继续更新种子点
  seeds_updating_halt_ = false;
}

//* 当关键被移除,对应的种子点也要被移除
void DepthFilter::removeKeyframe(FramePtr frame)
{
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_);
  std::list<Seed>::iterator it=seeds_.begin();
  size_t n_removed = 0;
  while(it!=seeds_.end())
  {
    if(it->ftr->frame == frame.get())
    {
      if(!f.is_open())
        std::cout<<"!!!!!!"<<std::endl;  
      f <<it->id << ", " << it->live_time << ", 0"<<endl;
      it = seeds_.erase(it);
      ++n_removed;
    }
    else
      ++it;
  }
  seeds_updating_halt_ = false;
}

//* 重置深度滤波器, 清空seed frame, 停止更新
void DepthFilter::reset()
{
  seeds_updating_halt_ = true;
  {
    lock_t lock(seeds_mut_);
    seeds_.clear();
  }
  //bug: 应该给frame_queue_mut_上锁
  lock_t lock(frame_queue_mut_);  //! 更改by gong
  // 清空frame队列
  while(!frame_queue_.empty())
    frame_queue_.pop();
  seeds_updating_halt_ = false; // 停止更新

  if(options_.verbose)
    SVO_INFO_STREAM("DepthFilter: RESET.");
}

/********************************
 * @ function: 更新种子点的大循环
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void DepthFilter::updateSeedsLoop()
{
  while(!boost::this_thread::interruption_requested()) //当有请求interrupt时终止
  {
    FramePtr frame;
    {
  //[ ***step 1*** ] 检测是否有线程锁frame_queue_
      lock_t lock(frame_queue_mut_);  
  //[ ***step 2*** ] 没有帧和关键帧加入, 就上锁, 等待 notify
      while(frame_queue_.empty() && new_keyframe_set_ == false)
        frame_queue_cond_.wait(lock);
  //[ ***step 3*** ] 若是新加入关键帧, 则重新处理  
      if(new_keyframe_set_)
      {
        new_keyframe_set_ = false;  // 处理了新的关键帧就置false
        seeds_updating_halt_ = false; // 增加的时候给暂停了
        clearFrameQueue(); //清楚frame队列
        frame = new_keyframe_;
      }
      else
  //[ ***step 4*** ] 没有关键帧加入, 就把最前边的帧拿出来进行更新地图, 并删除这一帧
      {
        frame = frame_queue_.front();
        frame_queue_.pop();
      }
    }
  //[ ***step 5*** ] 利用最新帧, 更新种子点
    updateSeeds(frame);
    if(frame->isKeyframe())
  //[ ***step 6*** ] 如果是关键帧就初始化上面的特征点为种子点  
      initializeSeeds(frame);
  }
}

/********************************
 * @ function: 更新种子点的高斯分布值
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void DepthFilter::updateSeeds(FramePtr frame)
{
  // update only a limited number of seeds, because we don't have time to do it
  // for all the seeds in every frame!
  // 
  size_t n_updates=0, n_failed_matches=0, n_seeds = seeds_.size();
  //[ ***step 1*** ] 检查是否有线程在访问seeds_
  lock_t lock(seeds_mut_);
  std::list<Seed>::iterator it=seeds_.begin();
    //* 计算1 pixel 对应的角度(计算协方差用)
  const double focal_length = frame->cam_->errorMultiplier2(); // 焦距
  double px_noise = 1.0;  // 噪声对应的像素值, (参考笔记单目深度估计)
  double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)

  while( it!=seeds_.end())
  {
    // set this value true when seeds updating should be interrupted
    if(seeds_updating_halt_) //* 如果设置了停止, 就直接返回
      return;
  //[ ***step 2*** ] 剔除时间太久的种子点, 当前batch_id - seed_batch_id
    // check if seed is not already too old
    if((Seed::batch_counter - it->batch_id) > options_.max_n_kfs) {
      if(!f.is_open())
        std::cout<<"!!!!!!"<<std::endl;  
      f << it->id << ", " << it->live_time << ", 0"<<endl;
      it = seeds_.erase(it);
      continue;
    }

  //[ ***step 3*** ] 求当前帧 到 种子点特征点所在帧的变换
    // check if point is visible in the current image
    SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
  //[ ***step 4*** ] 投影到当前帧, 并判断是否在正确位置
    const Vector3d xyz_f(T_ref_cur.inverse()*(1.0/it->mu * it->ftr->f) );
    if(xyz_f.z() < 0.0)  {
      ++it; // behind the camera
      continue;
    }
    if(!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>())) {
      ++it; // point does not project in image
      continue;
    }

  //[ ***step 5*** ] 根据均值方差计算深度的搜索范围, 进行极线搜索得到参考帧的深度
    // we are using inverse depth coordinates
    float z_inv_min = it->mu + sqrt(it->sigma2);
    float z_inv_max = max(it->mu - sqrt(it->sigma2), 0.00000001f);
    double z;
    if(!matcher_.findEpipolarMatchDirect(
        *it->ftr->frame, *frame, *it->ftr, 1.0/it->mu, 1.0/z_inv_min, 1.0/z_inv_max, z))
    {
      it->b++; // increase outlier probability when no match was found
      ++it;
      ++n_failed_matches;
      continue;
    }
  //[ ***step 6*** ] 根据新计算出来的值, 计算方差, 并更新种子点
    // compute tau
    double tau = computeTau(T_ref_cur, it->ftr->f, z, px_error_angle);
    double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

    // update the estimate
    updateSeed(1./z, tau_inverse*tau_inverse, &*it);
    ++n_updates;

    //* 如果是关键帧, 把对齐得到的位置占据
    if(frame->isKeyframe())
    {
      // The feature detector should not initialize new seeds close to this location
      feature_detector_->setGridOccpuancy(matcher_.px_cur_);
    }
  //[ ***step 7*** ] 判断收敛, 将新的深度转化为新的3D点, 给种子点, 传给回调函数, 并删除种子点
  //? 这个收敛条件有些不懂
    // if the seed has converged, we initialize a new candidate point and remove the seed
    if(sqrt(it->sigma2) < it->z_range/options_.seed_convergence_sigma2_thresh)
    {
      assert(it->ftr->point == NULL); // TODO this should not happen anymore
      Vector3d xyz_world(it->ftr->frame->T_f_w_.inverse() * (it->ftr->f * (1.0/it->mu)));
      Point* point = new Point(xyz_world, it->ftr);
      //! 在这之前 feature 都没有指向的point, 只有frame.
      it->ftr->point = point;

      /* FIXME it is not threadsafe to add a feature to the frame here.
      if(frame->isKeyframe())
      {
        Feature* ftr = new Feature(frame.get(), matcher_.px_cur_, matcher_.search_level_);
        ftr->point = point;
        point->addFrameRef(ftr);
        frame->addFeature(ftr);
        it->ftr->frame->addFeature(it->ftr);
      }
      else
      */
    //* 把新的3D点, 和方差, 传给回调函数, 赋值给map->point_candidates
      {
        seed_converged_cb_(point, it->sigma2); // put in candidate list
      }
      //
      if(!f.is_open())
        std::cout<<"!!!!!!"<<std::endl;  
      f << it->id << ", " << it->live_time << ", 1"<<endl;
      it = seeds_.erase(it);
    }
    else if(isnan(z_inv_min))
    {
      SVO_WARN_STREAM("z_min is NaN");
      if(!f.is_open())
        std::cout<<"!!!!!!"<<std::endl;  
      f << it->id << ", " << it->live_time << ", 0"<<endl;
      it = seeds_.erase(it);
    }
    else
      ++it;
  }
}

//* 删除 frame 队列所以值 
void DepthFilter::clearFrameQueue()
{
  while(!frame_queue_.empty())
    frame_queue_.pop();
}

//* 从种子点中找出在 frame 上的点, 并复制给seeds
void DepthFilter::getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds)
{
  lock_t lock(seeds_mut_);
  for(std::list<Seed>::iterator it=seeds_.begin(); it!=seeds_.end(); ++it)
  {
    if (it->ftr->frame == frame.get())
      seeds.push_back(*it);
  }
}

//* 公式更新后验参数, 公式见笔记(单目稠密重建)
//* 论文: Video-based, Real-Time Multi View Stereo 
void DepthFilter::updateSeed(const float x, const float tau2, Seed* seed)
{
  seed->live_time++;
  float norm_scale = sqrt(seed->sigma2 + tau2);
  if(std::isnan(norm_scale))
    return;
  //! N (mu, sigma^2 + tau^2)  
  boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
  //! 1/s^2=1/sigma^2 + 1/tau^2
  float s2 = 1./(1./seed->sigma2 + 1./tau2);
  //! s2 * (mu/sigma^2 + x/tau^2)
  float m = s2*(seed->mu/seed->sigma2 + x/tau2);
  //! a/(a+b)*N ( x | mu, sigma^2 + tau^2)  
  float C1 = seed->a/(seed->a+seed->b) * boost::math::pdf(nd, x);
  //! b/(a+b)*U(x)
  float C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range;
  //! C=C1+C2 , 归一化  
  float normalization_constant = C1 + C2;
  C1 /= normalization_constant;
  C2 /= normalization_constant;

  float f = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
  float e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
          + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));
  //* 更新参数
  // update parameters
  float mu_new = C1*m+C2*seed->mu;
  seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;
  seed->mu = mu_new;
  seed->a = (e-f)/(f-e/f);
  seed->b = seed->a*(1.0f-f)/f;
}

//* 公式求 标准差(\tau) , 公式见笔记(单目稠密重建)
//* 对应论文: REMODE: Probabilistic, Monocular Dense Reconstruction in Real Time
double DepthFilter::computeTau(
      const SE3& T_ref_cur,
      const Vector3d& f,
      const double z,
      const double px_error_angle)
{
  Vector3d t(T_ref_cur.translation());
  Vector3d a = f*z-t;
  double t_norm = t.norm();
  double a_norm = a.norm();
  double alpha = acos(f.dot(t)/t_norm); // dot product
  double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
  double beta_plus = beta + px_error_angle;
  double gamma_plus = PI-alpha-beta_plus; // triangle angles sum to PI
  double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
  return (z_plus - z); // tau
}

} // namespace svo
