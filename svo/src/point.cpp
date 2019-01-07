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
#include <vikit/math_utils.h>
#include <svo/point.h>
#include <svo/frame.h>
#include <svo/feature.h>
 
namespace svo {

int Point::point_counter_ = 0;


/********************************
 * @ function: 管理三角化的3D点的类，就是MapPoint
 *              构造函数
 * @ param:    pos   该3D点坐标
 * 
 * @ note:
 *******************************/
Point::Point(const Vector3d& pos) :
  id_(point_counter_++), //3D点的ID
  pos_(pos), //3D点的坐标
  normal_set_(false), //是否估计表面法向量
  n_obs_(0), //能观测该点的帧数，包括关键帧和投影到的帧
  v_pt_(NULL), //作为g2o BA优化时的点的临时指针 
  last_published_ts_(0), //上一次的发布的时间戳
  last_projected_kf_id_(-1), //上一次投影的关键帧的ID，防止投影两次
  type_(TYPE_UNKNOWN), //3D点的质量（删除、候选、未知、好）
  n_failed_reproj_(0), //重投影失败的次数，用来衡量点的质量
  n_succeeded_reproj_(0), //重投影成功次数，同上
  last_structure_optim_(0) //上一次优化的时间戳
{}

Point::Point(const Vector3d& pos, Feature* ftr) :
  id_(point_counter_++),
  pos_(pos),
  normal_set_(false),
  n_obs_(1),
  v_pt_(NULL),
  last_published_ts_(0),
  last_projected_kf_id_(-1),
  type_(TYPE_UNKNOWN),
  n_failed_reproj_(0),
  n_succeeded_reproj_(0),
  last_structure_optim_(0)
{
  obs_.push_front(ftr); //增加一个该点的观测帧
}

Point::~Point()
{}

/********************************
 * @ function: 增加一个该点的观测帧，以feature形式表示
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void Point::addFrameRef(Feature* ftr)
{
  obs_.push_front(ftr);
  ++n_obs_;
}

/********************************
 * @ function: 查找该点是否被frame观测
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
Feature* Point::findFrameRef(Frame* frame)
{
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
    if((*it)->frame == frame)
      return *it;
  return NULL;    // no keyframe found
}

/********************************
 * @ function: 删除该点的一个观测帧
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
bool Point::deleteFrameRef(Frame* frame)
{
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
  {
    if((*it)->frame == frame)
    {
      obs_.erase(it);
      return true;
    }
  }
  return false;
}
/********************************
 * @ function: 计算该点的法向量（点到相机方向），以及它的协方差
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void Point::initNormal()
{
  assert(!obs_.empty());
//[***step 1***] 得到最近的关键帧上该点的特征点（投影）
  const Feature* ftr = obs_.back();
  assert(ftr->frame != NULL);
//[***step 2***] 将该点在归一化平面上的坐标转到世界坐标下，作为法向量，方向由该点指向摄影机中心
  normal_ = ftr->frame->T_f_w_.rotation_matrix().transpose()*(-ftr->f);
//[***step 3***] 计算该法向量的协方差矩阵，为什么这么求？？？，3D点的坐标到归一化平面的距离越大协方差越小？？？ 
  normal_information_ = DiagonalMatrix<double,3,3>(pow(20/(pos_-ftr->frame->pos()).norm(),2), 1.0, 1.0);
  // 标志位置1
  normal_set_ = true;
}

/********************************
 * @ function: 找到最近的一帧，夹角最小
 * 
 * @ param: 
 * 
 * @ note: 都是世界坐标
 *******************************/
bool Point::getCloseViewObs(const Vector3d& framepos, Feature*& ftr) const
{
  // TODO: get frame with same point of view AND same pyramid level!
//[***step 1***] 得到framepose与当前3D点之间的向量，并归一化
  Vector3d obs_dir(framepos - pos_); obs_dir.normalize();
  auto min_it=obs_.begin();
  double min_cos_angle = 0;
//[***step 2***] 从当前点的可观测帧中找到夹角最小的
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
  {
    // 得到每个观测帧到该点的向量并归一化
    Vector3d dir((*it)->frame->pos() - pos_); dir.normalize();
    // 由于归一化可以求得夹角
    double cos_angle = obs_dir.dot(dir);
    // 找最小
    if(cos_angle > min_cos_angle)
    {
      min_cos_angle = cos_angle;
      min_it = it;
    }
  }
  // 返回最近的feature（帧）
  ftr = *min_it;
  // 大于60°视为无用
  if(min_cos_angle < 0.5) // assume that observations larger than 60° are useless
    return false;
  return true;
}


/********************************
 * @ function: 最小化投影的坐标，优化3D点的坐标
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void Point::optimize(const size_t n_iter)
{
  Vector3d old_point = pos_; //待优化的量
  double chi2 = 0.0; 
  // A'Ax=A'b 
  Matrix3d A;
  Vector3d b;

  for(size_t i=0; i<n_iter; i++)
  {
    A.setZero();
    b.setZero();
    double new_chi2 = 0.0;

    // compute residuals
    for(auto it=obs_.begin(); it!=obs_.end(); ++it)
    {
      Matrix23d J; //Jacobians
      const Vector3d p_in_f((*it)->frame->T_f_w_ * pos_); //世界系转换为相机系下
      Point::jacobian_xyz2uv(p_in_f, (*it)->frame->T_f_w_.rotation_matrix(), J); //计算Jacobians
      const Vector2d e(vk::project2d((*it)->f) - vk::project2d(p_in_f));//计算投影误差，是坐标的误差，而不是光度
      new_chi2 += e.squaredNorm();//归一化
      A.noalias() += J.transpose() * J;
      b.noalias() -= J.transpose() * e;
      // 如果不用noalias，内部操作是这样的：m = m+ m
      // 如果使用noalias，内部操作是这样的：tmp = m+ m, m = tmp
    }

    // solve linear system
    // 求解AX=b
    const Vector3d dp(A.ldlt().solve(b));

    // check if error increased
    // 判断chi2是否增加，以及是否有解
    if((i > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dp[0]))
    {
#ifdef POINT_OPTIMIZER_DEBUG
      cout << "it " << i
           << "\t FAILURE \t new_chi2 = " << new_chi2 << endl;
#endif
      pos_ = old_point; // roll-back 回滚操作（想到了西部世界 -_-! ，roll back）
      break;
    }

    // 迭代优化
    // update the model
    Vector3d new_point = pos_ + dp;
    old_point = pos_;
    pos_ = new_point;
    chi2 = new_chi2;
#ifdef POINT_OPTIMIZER_DEBUG
    cout << "it " << i
         << "\t Success \t new_chi2 = " << new_chi2
         << "\t norm(b) = " << vk::norm_max(b)
         << endl;
#endif

    // stop when converged
    if(vk::norm_max(dp) <= EPS)
      break;
  }
#ifdef POINT_OPTIMIZER_DEBUG
  cout << endl;
#endif
}

} // namespace svo
