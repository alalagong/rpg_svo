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
#include <svo/pose_optimizer.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <vikit/robust_cost.h>
#include <vikit/math_utils.h>

namespace svo {
namespace pose_optimizer {


void optimizeGaussNewton(
    const double reproj_thresh,
    const size_t n_iter,
    const bool verbose,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs)
{
  // init
  double chi2(0.0);
  vector<double> chi2_vec_init, chi2_vec_final;
  vk::robust_cost::TukeyWeightFunction weight_function; // Tukey权重函数(1-x^2/b^2)^2, 0
  SE3 T_old(frame->T_f_w_); // 原位姿(SE3) !  
  Matrix6d A;
  Vector6d b;

  // compute the scale of the error for robust estimation
  std::vector<float> errors; errors.reserve(frame->fts_.size()); // 特征点的位置误差
//[ ***step 1*** ] 计算frame上每个特征点位置与3D点投影位置的误差(单位平面上)
  for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point == NULL) //? 为什么会出现这个情况
      continue;
  //* 特征位置和3D点投影的位置误差(在单位平面上!!!)
    Vector2d e = vk::project2d((*it)->f)
               - vk::project2d(frame->T_f_w_ * (*it)->point->pos_);
  // 转换到相应的金字塔层, 层数越高的占比越小, 高层的误差噪声大,则缩小               
    e *= 1.0 / (1<<(*it)->level); 
    errors.push_back(e.norm()); //xy平方和, 像素距离, 误差大小
  }
  if(errors.empty())
    return;
//[ ***step 2*** ] 根据总errors计算中位数绝对误差, 误差的尺度, 确定误差整体是比较大还是比较小
  vk::robust_cost::MADScaleEstimator scale_estimator; // 中位数绝对偏差估计
  estimated_scale = scale_estimator.compute(errors); // 返回估计的标准差

  num_obs = errors.size();
  chi2_vec_init.reserve(num_obs); //初始卡方误差
  chi2_vec_final.reserve(num_obs); // 最终卡方误差
  double scale = estimated_scale;
  //* 迭代优化位姿
  for(size_t iter=0; iter<n_iter; iter++)
  {
    // overwrite scale
    // 第五次迭代会重新改变一次scale 
    if(iter == 5)
    //* 之前求得J里面不包括fx, 因此阈值也要除掉fx 
    //* 越往后由于迭代, 误差的标准差会越小(这是个估计值把)
    //? 这个尺度有什么作用 , 师兄注释说越往后误差可靠性越高?
      scale = 0.85/frame->cam_->errorMultiplier2();// fabs(fx_)

    b.setZero();
    A.setZero();
    double new_chi2(0.0);

    // compute residual
//[ ***step 3*** ] 计算最小二乘相关参数
    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
      if((*it)->point == NULL)
        continue;
      //*step3.1 计算对T的雅克比矩阵J, 先平移后旋转
      Matrix26d J;
      Vector3d xyz_f(frame->T_f_w_ * (*it)->point->pos_);
      Frame::jacobian_xyz2uv(xyz_f, J);
      //*step3.2 计算残差
      Vector2d e = vk::project2d((*it)->f) - vk::project2d(xyz_f);
      //*step3.3 计算的是信息矩阵? ∑^(1/2)
      double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
      //*step3.4 求和计算整个最小二乘 A 和 b
      //! ∑^(1/2)*b
      e *= sqrt_inv_cov;
      if(iter == 0)
        // e的平方和
        chi2_vec_init.push_back(e.squaredNorm()); // just for debug
      //! J*∑^(1/2)
      J *= sqrt_inv_cov;
      //* robust处理
      //! x_square <= b_square ---> const float tmp = 1.0f - x_square / b_square; return tmp * tmp;
      //! x_square > b_square  ---> return 0;
      //* 权重可以使得, 大的误差系数是0, 小的误差权重大一些, 大的误差权重小一些. 去除外点更加鲁棒
      double weight = weight_function.value(e.norm()/scale); // 权重函数
      //! A=J^T*∑^(-1)*J
      A.noalias() += J.transpose()*J*weight;
      //! b=J^T*∑^(-1)*e
      b.noalias() -= J.transpose()*e*weight;
      new_chi2 += e.squaredNorm()*weight;
    }
//[ ***step 4*** ] 求解最小二乘问题得到 dT
    // solve linear system
    const Vector6d dT(A.ldlt().solve(b));

    // check if error increased
    if((iter > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dT[0]))
    {
      if(verbose)
        std::cout << "it " << iter
                  << "\t FAILURE \t new_chi2 = " << new_chi2 << std::endl;
      frame->T_f_w_ = T_old; // roll-back
      break;
    }
//[ ***step 5*** ] 更新图像的位姿
    // update the model
    SE3 T_new = SE3::exp(dT)*frame->T_f_w_;
    T_old = frame->T_f_w_;
    frame->T_f_w_ = T_new;
    chi2 = new_chi2;
    if(verbose)
      std::cout << "it " << iter
                << "\t Success \t new_chi2 = " << new_chi2
                << "\t norm(dT) = " << vk::norm_max(dT) << std::endl;

    // stop when converged
    if(vk::norm_max(dT) <= EPS)
      break;
  }

  // Set covariance as inverse information matrix. Optimistic estimator!
//[ ***step 6*** ] 计算图像位姿的协方差, cov=(J^T*∑^(-1)*J)^+, 伪逆
  const double pixel_variance=1.0;
  //* 由于之前求的J里面没有fx, 这里A要乘上fx^2
  frame->Cov_ = pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();

  // Remove Measurements with too large reprojection error
  //* 同理阈值要除掉fx
  double reproj_thresh_scaled = reproj_thresh / frame->cam_->errorMultiplier2();
  size_t n_deleted_refs = 0;
//[ ***step 7*** ] 计算优化后的误差, 如果大于阈值, 则把该点删除
  for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point == NULL)
      continue;
    Vector2d e = vk::project2d((*it)->f) - vk::project2d(frame->T_f_w_ * (*it)->point->pos_);
    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
    e *= sqrt_inv_cov;
    chi2_vec_final.push_back(e.squaredNorm());
    if(e.norm() > reproj_thresh_scaled)
    {
      // we don't need to delete a reference in the point since it was not created yet 
      //? 上面这句什么意思??? 不需要删除点的参考,因为还没建立?
      (*it)->point = NULL;
      ++n_deleted_refs;
    }
  }
  // 计算优化前后误差的中位数
  error_init=0.0;
  error_final=0.0;
  if(!chi2_vec_init.empty())
    error_init = sqrt(vk::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
  if(!chi2_vec_final.empty())
    error_final = sqrt(vk::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();
  //? 这是什么尺度
  //?答 误差的尺度吧, 反应误差大小
  estimated_scale *= frame->cam_->errorMultiplier2();
  if(verbose)
    std::cout << "n deleted obs = " << n_deleted_refs
              << "\t scale = " << estimated_scale
              << "\t error init = " << error_init
              << "\t error end = " << error_final << std::endl;
  num_obs -= n_deleted_refs; // 从观测中减去删除的点
}

} // namespace pose_optimizer
} // namespace svo
