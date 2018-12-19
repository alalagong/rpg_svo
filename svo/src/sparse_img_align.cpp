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
#include <svo/sparse_img_align.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/config.h>
#include <svo/point.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>

namespace svo {
/********************************
 * @ function: 构造函数
 * 
 * @ param:   int n_levels,    // = max_level = max_level_  粗金字塔
              int min_level,   // = min_level_  细金字塔 
              int n_iter,      // 迭代次数, 继承自vk::NLLSSolver
              Method method,   // GN 和 LM
              bool display,    // = display_
              bool verbose);   // = verbose_ 继承来, 输出统计信息
 * 
 * @ note:  对继承来的一些参数赋值
 *******************************/
SparseImgAlign::SparseImgAlign(
    int max_level, int min_level, int n_iter,
    Method method, bool display, bool verbose) :
        display_(display),
        max_level_(max_level),
        min_level_(min_level)
{
  //继承来的
  n_iter_ = n_iter;
  n_iter_init_ = n_iter_;
  method_ = method;
  verbose_ = verbose;
  eps_ = 0.000001;  //精度
}
/********************************
 * @ function: 利用稀疏图像对齐, 
 *             对当前帧和参考帧之间的T进行由粗到精的优化
 * 
 * @ param: 参考帧 ref_frame
 *          当前帧 cur_frame
 * 
 * @ note:
 *******************************/
size_t SparseImgAlign::run(FramePtr ref_frame, FramePtr cur_frame)
{ 
  //[***step 1***] 复位NLLSSolver的变量值
  reset();

  if(ref_frame->fts_.empty())
  {
    SVO_WARN_STREAM("SparseImgAlign: no features to track!");
    return 0;
  }

  ref_frame_ = ref_frame;
  cur_frame_ = cur_frame;

  //[***step 2***] 初始化cache
  //存储每个特征点的patch, 大小 (features_size*patch_area(4*4))
  ref_patch_cache_ = cv::Mat(ref_frame_->fts_.size(), patch_area_, CV_32F);
  //存储每个所有patch的雅克比, 大小 (6*ref_patch_cache_.size)
  jacobian_cache_.resize(Eigen::NoChange, ref_patch_cache_.rows*patch_area_);
  //存储可见的特征点, 类型vector<bool> 大小 feature_size, 默认都为false
  visible_fts_.resize(ref_patch_cache_.rows, false); // TODO: should it be reset at each level?

  //[***step 3***] 获得从参考帧到当前帧之间的变换 
  // T_cur_from_world = T_cur_from_ref * T_ref_from_world
  // T_[to]_[from]  T_A_C = T_A_B * T_B_C
  SE3 T_cur_from_ref(cur_frame_->T_f_w_ * ref_frame_->T_f_w_.inverse());

  //[***step 4***] 在不同的金字塔层对T_c_r进行稀疏图像对齐优化, 由粗到精, 具有更好的初值
  // 在4level到2level之间
  for(level_=max_level_; level_>=min_level_; --level_)
  {
    mu_ = 0.1;
    jacobian_cache_.setZero();
    have_ref_patch_cache_ = false;
    if(verbose_)
      printf("\nPYRAMID LEVEL %i\n---------------\n", level_);
    optimize(T_cur_from_ref);
  }

  //[***step 5***] 利用求得的T_c_r 求得 T_c_w
  cur_frame_->T_f_w_ = T_cur_from_ref * ref_frame_->T_f_w_;

  // n_meas_表示前一帧所有特征点块(feature patch)像素投影后在cur_frame中的像素个数。
  // n_meas_/patch_area_表示特征点数
  return n_meas_/patch_area_;
}

Matrix<double, 6, 6> SparseImgAlign::getFisherInformation()
{
  double sigma_i_sq = 5e-4*255*255; // image noise
  Matrix<double,6,6> I = H_/sigma_i_sq; //Hessian矩阵除以噪声？
  return I;
}

/********************************
 * @ function: 逆向组合法的预计算阶段，计算J
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void SparseImgAlign::precomputeReferencePatches()
{
  const int border = patch_halfsize_+1;
  //[***step 1***] 获取level_层图像金字塔的图像
  const cv::Mat& ref_img = ref_frame_->img_pyr_.at(level_);
  const int stride = ref_img.cols; //当前层图像的列数
  const float scale = 1.0f/(1<<level_); //金字塔的尺度, 每层2倍
  const Vector3d ref_pos = ref_frame_->pos(); //当前帧位置
  const double focal_length = ref_frame_->cam_->errorMultiplier2();  //返回 fx_ ??
  size_t feature_counter = 0;
  std::vector<bool>::iterator visiblity_it = visible_fts_.begin();

  for(auto it=ref_frame_->fts_.begin(), ite=ref_frame_->fts_.end();
      it!=ite; ++it, ++feature_counter, ++visiblity_it)
  {
    // check if reference with patch size is within image
    const float u_ref = (*it)->px[0]*scale; // 原图像的像素坐标变换到对应金字塔层
    const float v_ref = (*it)->px[1]*scale;

//[***step 2***] 将特征点变换到金字塔图像上, 并判断feature的patch是否在其内(不包括边缘)
    const int u_ref_i = floorf(u_ref); //变换后坐标下取整
    const int v_ref_i = floorf(v_ref);
    //特征点patch是否在变换后的图像内(不包括边缘), 并且具有三维坐标则可见
    if((*it)->point == NULL || u_ref_i-border < 0 || v_ref_i-border < 0 || u_ref_i+border >= ref_img.cols || v_ref_i+border >= ref_img.rows)
      continue;
    *visiblity_it = true;

//[***step 3***] 得到相机坐标系下的特征点, 并求得对变换矩阵的雅克比矩阵
    // cannot just take the 3d points coordinate because of the reprojection errors in the reference image!!!  这句话啥意思???
    const double depth(((*it)->point->pos_ - ref_pos).norm()); //点的三维坐标减去ref的相机坐标得到深度z
    const Vector3d xyz_ref((*it)->f*depth); //为单位平面上的值(x/z, y/z, 1) * z 得到camera坐标
    // evaluate projection jacobian
    Matrix<double,2,6> frame_jac;
    Frame::jacobian_xyz2uv(xyz_ref, frame_jac); // 雅克比 du/dp_c * dp_c/dzeta(se3)

////[***step 4***] 利用该层金字塔图像对每个patch中像素进行插值，计算得到图像对像素的导数。
    // compute bilateral interpolation weights for reference image
    const float subpix_u_ref = u_ref-u_ref_i;
    const float subpix_v_ref = v_ref-v_ref_i;
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    size_t pixel_counter = 0;
    //首指针 + 特征点的索引*patch大小, 即遍历访问ref_patch_cache_, cache内每个特征点patch的访问指针
    float* cache_ptr = reinterpret_cast<float*>(ref_patch_cache_.data) + patch_area_*feature_counter; 
    for(int y=0; y<patch_size_; ++y)
    {
      //计算出该层金字塔图像这个特征的patch的第一个像素指针
      uint8_t* ref_img_ptr = (uint8_t*) ref_img.data + (v_ref_i+y-patch_halfsize_)*stride + (u_ref_i-patch_halfsize_);
      for(int x=0; x<patch_size_; ++x, ++ref_img_ptr, ++cache_ptr, ++pixel_counter)
      {
        /***************x***************|
         * \  #     #     #     #     #     
         * \      。    。    。    。
         * \  #     #     #     #     #   
         * \      。    。    。    。      
         * y  #     #     #     #     #
         * \      。    。    @     。
         * \  #     #     #     #     #
         * \      。    。    。    。
         * \  #     #     #     #     #
         * ****************************/
        // @为计算出来的金字塔上图像的点，向下取正得到#，。为插值得到的点
        // 我们是向下取值的, 因此这么插值得到patch中第一个点的值
        // 每次都插值一次得到要求的点
        // precompute interpolated reference patch color
        *cache_ptr = w_ref_tl*ref_img_ptr[0] + w_ref_tr*ref_img_ptr[1] + w_ref_bl*ref_img_ptr[stride] + w_ref_br*ref_img_ptr[stride+1];

        // we use the inverse compositional: thereby we can take the gradient always at the same position
        // get gradient of warped image (~gradient at warped position)
        //求导方法：利用四个方向像素的差除以2
        float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1] + w_ref_tr*ref_img_ptr[2] + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
                          -(w_ref_tl*ref_img_ptr[-1] + w_ref_tr*ref_img_ptr[0] + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
        float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride] + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
                          -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0] + w_ref_br*ref_img_ptr[1]));
        
//[***step 5***] 图像导数与对se(3)的导数乘积求Jacobian  
        //每个像素一行jacobian (1*2)×(2*6)，乘上对应的内参，为什么要缩小倍数???
        // cache the jacobian
        jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter) =
            (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length / (1<<level_));
      }
    }
  }
  have_ref_patch_cache_ = true;
}

/********************************
 * @ function: 逆向组合法对cur_image和ref_image进行计算res
 * 
 * @ param:   model                 待优化量
 *            linearize_system      是否为线性系统
 *            compute_weight_scale  是否计算权重，true则该函数是计算robust_weight
 *            
 *            返回平均的res×res, 用来判断误差是否增加
 * 
 * @ note: 这是个基类的虚函数
 *******************************/
double SparseImgAlign::computeResiduals(
    const SE3& T_cur_from_ref,
    bool linearize_system,
    bool compute_weight_scale)
{
//[***step 1***] 得到当前金字塔层的cur_image图像
  // Warp the (cur)rent image such that it aligns with the (ref)erence image
  const cv::Mat& cur_img = cur_frame_->img_pyr_.at(level_);

  //用来显示res图像
  if(linearize_system && display_)
    resimg_ = cv::Mat(cur_img.size(), CV_32F, cv::Scalar(0));
  //是否预计算了需要的导数jacobian_cache_等信息，否则计算
  if(have_ref_patch_cache_ == false)
    precomputeReferencePatches();

  // compute the weights on the first iteration
  std::vector<float> errors; //存放res的
  //如果计算compute_weight_scale则分配capacity
  if(compute_weight_scale)
    errors.reserve(visible_fts_.size());
  //定义并初始化一些信息
  const int stride = cur_img.cols; //当前图像列数
  const int border = patch_halfsize_+1; 
  const float scale = 1.0f/(1<<level_); 
  const Vector3d ref_pos(ref_frame_->pos()); //参考图像帧位置
  float chi2 = 0.0; //X^2检验用???
  size_t feature_counter = 0; // is used to compute the index of the cached jacobian
  std::vector<bool>::iterator visiblity_it = visible_fts_.begin();
  //对每个ref图像上的特征点进行循环
  for(auto it=ref_frame_->fts_.begin(); it!=ref_frame_->fts_.end();
      ++it, ++feature_counter, ++visiblity_it)
  {
    // check if feature is within image
    //不在图像中则忽略
    if(!*visiblity_it)
      continue;
//[***step 2***] 利用T_cur_from_ref将ref_image上的特征点对应的三维点投影到当前帧并转换到图像金字塔上
    // compute pixel location in cur img
    const double depth = ((*it)->point->pos_ - ref_pos).norm();
    const Vector3d xyz_ref((*it)->f*depth);
    const Vector3d xyz_cur(T_cur_from_ref * xyz_ref);
    const Vector2f uv_cur_pyr(cur_frame_->cam_->world2cam(xyz_cur).cast<float>() * scale);
    const float u_cur = uv_cur_pyr[0];
    const float v_cur = uv_cur_pyr[1];
    const int u_cur_i = floorf(u_cur);
    const int v_cur_i = floorf(v_cur);
    
    //判断转换后在不在当前层的cur_iamge的上
    // check if projection is within the image
    if(u_cur_i < 0 || v_cur_i < 0 || u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= cur_img.cols || v_cur_i+border >= cur_img.rows)
      continue;
    
    //插值系数
    // compute bilateral interpolation weights for the current image
    const float subpix_u_cur = u_cur-u_cur_i;
    const float subpix_v_cur = v_cur-v_cur_i;
    const float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
    const float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
    const float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
    const float w_cur_br = subpix_u_cur * subpix_v_cur;
    //ref_image上patch的mat型
    float* ref_patch_cache_ptr = reinterpret_cast<float*>(ref_patch_cache_.data) + patch_area_*feature_counter;
    size_t pixel_counter = 0; // is used to compute the index of the cached jacobian

//[***step 3***] 对每个cur_image上的feature周围的patch进行插值, 计算与ref_image上的patch的res    
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* cur_img_ptr = (uint8_t*) cur_img.data + (v_cur_i+y-patch_halfsize_)*stride + (u_cur_i-patch_halfsize_);

      for(int x=0; x<patch_size_; ++x, ++pixel_counter, ++cur_img_ptr, ++ref_patch_cache_ptr)
      {
        // compute residual
        const float intensity_cur = w_cur_tl*cur_img_ptr[0] + w_cur_tr*cur_img_ptr[1] + w_cur_bl*cur_img_ptr[stride] + w_cur_br*cur_img_ptr[stride+1];
        const float res = intensity_cur - (*ref_patch_cache_ptr);

        // used to compute scale for robust cost
        if(compute_weight_scale)
          errors.push_back(fabsf(res));

        // robustification
        //这是个啥???
        float weight = 1.0;
        if(use_weights_) {
          weight = weight_function_->value(res/scale_);
        }

        chi2 += res*res*weight; //x^2
        n_meas_++; //计算的所有patch中的像素数量
//[***step 4***] 利用jacobiansp_cache求H, J
        if(linearize_system)
        {
          // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
          const Vector6d J(jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter)); // 6*1
          H_.noalias() += J*J.transpose()*weight; //这里没有混淆, //6*6
          Jres_.noalias() -= J*res*weight;
          if(display_)
            resimg_.at<float>((int) v_cur+y-patch_halfsize_, (int) u_cur+x-patch_halfsize_) = res/255.0;
        }
      }
    }
  }

  // compute the weights on the first iteration
  if(compute_weight_scale && iter_ == 0)
    scale_ = scale_estimator_->compute(errors);
  //返回平均的res平方
  return chi2/n_meas_;
}

/********************************
 * @ function: 求解(A^T)A*x=(A^T)b的最小二乘问题
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
int SparseImgAlign::solve()
{

  //求解 H_*x_ = Jres_
  x_ = H_.ldlt().solve(Jres_);
  if((bool) std::isnan((double) x_[0]))
    return 0;
  return 1;
}
/********************************
 * @ function: 逆向组合的更新T
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void SparseImgAlign::update(
    const ModelType& T_curold_from_ref,
    ModelType& T_curnew_from_ref)
{
  T_curnew_from_ref =  T_curold_from_ref * SE3::exp(-x_);
}

void SparseImgAlign::startIteration()
{}

void SparseImgAlign::finishIteration()
{
  if(display_)
  {
    cv::namedWindow("residuals", CV_WINDOW_AUTOSIZE);
    cv::imshow("residuals", resimg_*10);
    cv::waitKey(0);
  }
}

} // namespace svo

