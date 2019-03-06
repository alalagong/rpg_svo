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

#include <cstdlib>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>
#include <vikit/patch_score.h>
#include <svo/matcher.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/config.h>
#include <svo/feature_alignment.h>

namespace svo {

namespace warp {

/********************************
 * @ function:  得到已知位姿的两个图像之间仿射变换
 *                            
 * @ param:     输入ref的相机参数,像素坐标,归一化坐标,深度,层数
 *              输入cur的相机参数,ref到cur的变换矩阵
 *              返回2*2的放射矩阵
 * 
 * @ note:      求法很有意思
 *              !这里的金字塔层数有什么意义??
 *******************************/
void getWarpMatrixAffine(
    const vk::AbstractCamera& cam_ref,
    const vk::AbstractCamera& cam_cur,
    const Vector2d& px_ref, // patch取得是中间点,所以加上half
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,
    Matrix2d& A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const int halfpatch_size = 5; // 考虑边界8+2的一半
  const Vector3d xyz_ref(f_ref*depth_ref); // 点在ref下的3D坐标
  //! 为什么层数越大加的数越大
  //* px_ref虽然是在某一金字塔层提取的, 但是也都会扩大到相应倍数到0层的坐标上 
  //* 因为特征是在level_ref上提取的, 所以该层的patch对应到0层上要扩大相应倍数
  // 图像上根据金字塔层数对应的patch大小,得到patch的右上角坐标
  Vector3d xyz_du_ref(cam_ref.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref))); 
  // 图像上根据金字塔层数对应的patch大小,得到patch的左下角坐标
  Vector3d xyz_dv_ref(cam_ref.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));
  // 根据该点的深度得到右上角,左下角的3D坐标(一种近似?)
  xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];
  // 将这三点变换到cur下的图像坐标
  const Vector2d px_cur(cam_cur.world2cam(T_cur_ref*(xyz_ref)));
  const Vector2d px_du(cam_cur.world2cam(T_cur_ref*(xyz_du_ref)));
  const Vector2d px_dv(cam_cur.world2cam(T_cur_ref*(xyz_dv_ref)));
  //* 把原来的当做轴, 变换得到对应的轴就是两列(相当于原来的是(1,0)和(0,1))
  //* 参见https://images2015.cnblogs.com/blog/120296/201602/120296-20160222070732869-1123994329.png 后几个图 
  //* 这个A_cur_ref是从金字塔层到cur第0层的变换 
  A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

//* 找到合适的搜索的匹配金字塔层(从cur上搜索) 
int getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  //* 直到A的行列式值小于3, 为什么这样好呢?? 尺度接近1比较好??
  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25; //每增加一层, 行列式值就除以4
  }
  return search_level;
}

/********************************
 * @ function:  将ref上图像块通过A_cur_ref变换到patch
 * 
 * @ param:   const Matrix2d& A_cur_ref   仿射变换矩阵
 *            const cv::Mat& img_ref      变换前的ref对应金字塔层数的图像
 *            const Vector2d& px_ref      ref上(0层)特征点的位置(patch中间)
 *            const int level_ref         img_ref所在的level !
 *            const int search_level      最佳的匹配的金字塔层
 *            const int halfpatch_size    patch一半大小 + 1
 *            uint8_t* patch              patch_with_border_头指针
 * 
 * @ note:    search_level 和 level_ref 怎么个意思???
 *******************************/
void warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int halfpatch_size,
    uint8_t* patch)
{
  const int patch_size = halfpatch_size*2 ;
  //* 求逆是相当于从cur第0层到ref的第level_ref层变换
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>(); // 求A的逆变换, 遍历out图像利用A逆去src取值!
  //? 为什么A的(0,0)位置是0就是没有变换呢?
  if(isnan(A_ref_cur(0,0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  // Perform the warp on a larger patch.
  uint8_t* patch_ptr = patch;
  // ! 0层是最大的, 越往上越小
  const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref); //变成float型, 变换到对应的层数上
  for (int y=0; y<patch_size; ++y) 
  {
    for (int x=0; x<patch_size; ++x, ++patch_ptr)
    {
      Vector2f px_patch(x-halfpatch_size, y-halfpatch_size); // 以中点建立的坐标系, 得到的变换(见getBestSearchLevel函数)      
      //* search_level是相当于cur的层, 变换到第0层进行仿射变换!!!
      px_patch *= (1<<search_level); // 在相应的搜索金字塔层,扩大patch大小(最小是10?)
      //* A_ref_cur*px_patch把cur的第0层变换到ref的第level_ref层!!!
      const Vector2f px(A_ref_cur*px_patch + px_ref_pyr); // 仿射变换到ref上
      if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
        *patch_ptr = 0; // 超出去就设置为0
      else
        *patch_ptr = (uint8_t) vk::interpolateMat_8u(img_ref, px[0], px[1]); // cur 逆变换到 ref 上面取值(通过双线性插值)
    }
  }
}

} // namespace warp

/********************************
 * @ function: 三角化
 * 
 * @ param:  R, t, X1, X2
 * 
 * @ note:  返回的是ref下的深度
 *          ! d2X2=Rd1X1+t ==>  [RX1, X2]*[-d1, d2]^T=t
 *******************************/
bool depthFromTriangulation(
    const SE3& T_search_ref,
    const Vector3d& f_ref,
    const Vector3d& f_cur,
    double& depth)
{
  //* A是公式中的[RX1, X2]
  Matrix<double,3,2> A; A << T_search_ref.rotation_matrix() * f_ref, f_cur;
  //* 正规方程A^TA*d=A^T*t
  const Matrix2d AtA = A.transpose()*A;
  //* 判断是否可逆
  if(AtA.determinant() < 0.000001)
    return false;
  //* 求解正规方程 
  const Vector2d depth2 = - AtA.inverse()*A.transpose()*T_search_ref.translation();
  //* 得到ref下的深度
  depth = fabs(depth2[0]);
  return true;
}

//* patch_with_border 生成 patch 
void Matcher::createPatchFromPatchWithBorder()
{
  uint8_t* ref_patch_ptr = patch_; 
  for(int y=1; y<patch_size_+1; ++y, ref_patch_ptr += patch_size_) // 行循环, 因为有border所以y从1取
  {
    uint8_t* ref_patch_border_ptr = patch_with_border_ + y*(patch_size_+2) + 1; // patch_with_border移动到第y+1行, 并且由于border多加一
    for(int x=0; x<patch_size_; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}

/********************************
 * @ function:  直接使用图像对齐来进行匹配
 * 
 * @ param:     const Point& pt         匹配特征对应的3D点
 *              const Frame& cur_frame  当前帧
 *              Vector2d& px_cur        当前匹配的特征位置(输出)
 * 
 * @ note:      深度这么求? 
 *******************************/
bool Matcher::findMatchDirect(
    const Point& pt,
    const Frame& cur_frame,
    Vector2d& px_cur)
{
//[ ***step 1*** ] 找到与点pt对应的, 离当前帧最近的帧上的的特征ref_ftr
  if(!pt.getCloseViewObs(cur_frame.pos(), ref_ftr_))
    return false;
//[ ***step 2*** ] 验证该特征的patch(+2), 是否超过该层图像的大小
  //* 特征点是在某一金字塔层上提取的
  if(!ref_ftr_->frame->cam_->isInFrame(
      ref_ftr_->px.cast<int>()/(1<<ref_ftr_->level), halfpatch_size_+2, ref_ftr_->level))
    return false;

  // warp affine
//[ ***step 3*** ] 根据ref_ftr_周围的8*8 patch求得ref到cur之间的1D仿射矩阵
  warp::getWarpMatrixAffine(
      *ref_ftr_->frame->cam_, *cur_frame.cam_, ref_ftr_->px, ref_ftr_->f,
      //! 深度为什么这么算??? 
      //* 这里深度本就是不准确的,为了求深度才进行的匹配
      (ref_ftr_->frame->pos() - pt.pos_).norm(), 
      cur_frame.T_f_w_ * ref_ftr_->frame->T_f_w_.inverse(), ref_ftr_->level, A_cur_ref_);
//[ ***step 4*** ] 找到cur_frame最合适的搜索的金字塔层
  search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);
//[ ***step 5*** ] 利用 A_cur_ref 将ref变换到patch_with_border上, 得到的是search_level层上的patch
  warp::warpAffine(A_cur_ref_, ref_ftr_->frame->img_pyr_[ref_ftr_->level], ref_ftr_->px,
                   ref_ftr_->level, search_level_, halfpatch_size_+1, patch_with_border_);        
//[ ***step 6*** ] 去掉patch_with_border的边界
  createPatchFromPatchWithBorder();

  // px_cur should be set
  // 缩小到cur相应的层数 
  Vector2d px_scaled(px_cur/(1<<search_level_));
//[ ***step 7*** ] 使用逆向组合图像对齐, 得到优化后的px_scaled
  bool success = false;
  if(ref_ftr_->type == Feature::EDGELET) 
  {
    Vector2d dir_cur(A_cur_ref_*ref_ftr_->grad);
    dir_cur.normalize();
    success = feature_alignment::align1D(
          cur_frame.img_pyr_[search_level_], dir_cur.cast<float>(),
          patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
  }
  else
  {
    success = feature_alignment::align2D(
      cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
      options_.align_max_iter, px_scaled);
  }
  px_cur = px_scaled * (1<<search_level_); // 扩大相应倍数, 到第0层
  return success;
}

/********************************
 * @ function:    已知ref和cur的位姿, 进行块匹配, 最终计算ref上精确的深度
 * 
 * @ param:       const Frame& ref_frame      参考帧
 *                const Frame& cur_frame      匹配的当前帧
 *                const Feature& ref_ftr      参考帧上的特征
 *                const double d_estimate     估计的深度值
 *                const double d_min          深度值范围最小值
 *                const double d_max          深度值范围最大值
 *                double& depth               最终求得的ref上特征点的深度
 * 
 * @ note:        极线搜索算法, ZMSSD得分
 *******************************/
bool Matcher::findEpipolarMatchDirect(
    const Frame& ref_frame,
    const Frame& cur_frame,
    const Feature& ref_ftr,
    const double d_estimate,
    const double d_min,
    const double d_max,
    double& depth)
{
  SE3 T_cur_ref = cur_frame.T_f_w_ * ref_frame.T_f_w_.inverse(); 
  int zmssd_best = PatchScore::threshold(); // 2000*patch_area
  Vector2d uv_best;
//[ ***step 1*** ] 得到单位平面上极线的范围
  // Compute start and end of epipolar line in old_kf for match search, on unit plane!
  Vector2d A = vk::project2d(T_cur_ref * (ref_ftr.f*d_min)); // 单位平面上最小深度对应的极线端点
  Vector2d B = vk::project2d(T_cur_ref * (ref_ftr.f*d_max)); // 单位平面上最大深度对应的极线端点
  epi_dir_ = A - B; // 单位平面上极线段向量

//[ ***step 2*** ] 计算一个粗略的仿射变化矩阵 A_cur_ref_
  // Compute affine warp matrix
  warp::getWarpMatrixAffine(
      *ref_frame.cam_, *cur_frame.cam_, ref_ftr.px, ref_ftr.f,
      d_estimate, T_cur_ref, ref_ftr.level, A_cur_ref_);

  // feature pre-selection
  reject_ = false;
  // 是边框特征，则检查是否满足条件，不满足则reject
  if(ref_ftr.type == Feature::EDGELET && options_.epi_search_edgelet_filtering)
  {
    const Vector2d grad_cur = (A_cur_ref_ * ref_ftr.grad).normalized();
    const double cosangle = fabs(grad_cur.dot(epi_dir_.normalized()));
    if(cosangle < options_.epi_search_edgelet_max_angle) {
      reject_ = true;
      return false;
    }
  }

//[ ***step 3*** ] 找到在cur上最佳的搜索金字塔层，以及极线搜索范围
  search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);

  // Find length of search range on epipolar line
  // 单位平面投影到像素坐标
  Vector2d px_A(cur_frame.cam_->world2cam(A));
  Vector2d px_B(cur_frame.cam_->world2cam(B));
  // 计算图像上极线长度，变换到cur相应的层数
  epi_length_ = (px_A-px_B).norm() / (1<<search_level_);

//[ ***step 4*** ] 将ref上的patch变换到cur上，并去掉border
  // Warp reference patch at ref_level
  warp::warpAffine(A_cur_ref_, ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
                   ref_ftr.level, search_level_, halfpatch_size_+1, patch_with_border_);
  createPatchFromPatchWithBorder();

/***
 * *    *     *
 * 
 * *    @     *
 * 
 * *    *     * 
***/
//[ ***step 5*** ] 若极线长度小于2, 则进行特征对齐得到更精确的特征点位置, 然后进行三角化计算深度  
  // 极线长度小于2则, 即其附近8个点,
  if(epi_length_ < 2.0)
  {
    px_cur_ = (px_A+px_B)/2.0; // 取平均值
    Vector2d px_scaled(px_cur_/(1<<search_level_)); //cur点变到相应的层
    bool res;
    if(options_.align_1d)
      res = feature_alignment::align1D(
          cur_frame.img_pyr_[search_level_], (px_A-px_B).cast<float>().normalized(),
          patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
    else
      res = feature_alignment::align2D(
          cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
          options_.align_max_iter, px_scaled);
    if(res)
    {
      px_cur_ = px_scaled*(1<<search_level_);
      if(depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
        return true;
    }
    return false;
  }


  size_t n_steps = epi_length_/0.7; //步数 // one step per pixel 
  Vector2d step = epi_dir_/n_steps; //步长

  if(n_steps > options_.max_epi_search_steps)
  {
    // %zu 输出 size_t
    printf("WARNING: skip epipolar search: %zu evaluations, px_lenght=%f, d_min=%f, d_max=%f.\n",
           n_steps, epi_length_, d_min, d_max);
    return false;
  }


  // for matching, precompute sum and sum2 of warped reference patch
  int pixel_sum = 0;
  int pixel_sum_square = 0;
  PatchScore patch_score(patch_); //? 使用cur上的先计算的patch_作为ref_patch???

  //[ ***step 6*** ] 在之前求出的单位平面的极线段上进行搜索ZMSSD得分最小的patch
  // now we sample along the epipolar line
  //* 前面是A-B, 所以B每次增加一个step, 进行搜索
  // 这里先在AB之外取一个uv, 用于下面的循环, 保证每个AB之间点都在循环内
  Vector2d uv = B-step;
  Vector2i last_checked_pxi(0,0);
  ++n_steps; // 先减了一个, 多一步
  for(size_t i=0; i<n_steps; ++i, uv+=step)
  {
    Vector2d px(cur_frame.cam_->world2cam(uv)); // 转化到图像坐标下
    Vector2i pxi(px[0]/(1<<search_level_)+0.5,
                 px[1]/(1<<search_level_)+0.5); // +0.5 to round to closest int 对应的层上, 加0.5四舍五入
    
    // 和上一个相同则下一个, 怎么会相同(因为变换到更高的层了)
    if(pxi == last_checked_pxi)
      continue;
    last_checked_pxi = pxi;

    // check if the patch is full within the new frame
    // 检查patch是否都在其内
    if(!cur_frame.cam_->isInFrame(pxi, patch_size_, search_level_))
      continue;

    // TODO interpolation would probably be a good idea
    //* 计算得到cur_patch的头指针 (pxi是中间位置的点)
    uint8_t* cur_patch_ptr = cur_frame.img_pyr_[search_level_].data
                             + (pxi[1]-halfpatch_size_)*cur_frame.img_pyr_[search_level_].cols
                             + (pxi[0]-halfpatch_size_);
    // 计算极线上的patch与ref仿射变换得到的patch_之间的ZMSSD得分
    int zmssd = patch_score.computeScore(cur_patch_ptr, cur_frame.img_pyr_[search_level_].cols);
    // 越小越好
    if(zmssd < zmssd_best) {
      zmssd_best = zmssd;
      uv_best = uv;

    }
  }
//[ ***step 7*** ] 若满足条件则进行精确地特征对齐, 并进行三角化, 得到ref上的深度
  // 得分小于阈值则进行下一步, 否则返回false
  if(zmssd_best < PatchScore::threshold())
  {
    // 设置中如果进行精确优化
    if(options_.subpix_refinement)
    {
      px_cur_ = cur_frame.cam_->world2cam(uv_best); // cur_frame单位平面上的点转到像素平面
      Vector2d px_scaled(px_cur_/(1<<search_level_));  //变换到相应层
      bool res;
      // 进行特征对齐计算精确地特征位置
      if(options_.align_1d)
        res = feature_alignment::align1D(
            cur_frame.img_pyr_[search_level_], (px_A-px_B).cast<float>().normalized(),
            patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
      else
        res = feature_alignment::align2D(
            cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
            options_.align_max_iter, px_scaled);
      if(res)
      {
        px_cur_ = px_scaled*(1<<search_level_);
        if(depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
          return true;
      }
      return false;
    }
    // 若设置不进行精确优化, 则之间使用ZMSSD得分最小点进行三角化求得深度
    px_cur_ = cur_frame.cam_->world2cam(uv_best);
    if(depthFromTriangulation(T_cur_ref, ref_ftr.f, vk::unproject2d(uv_best).normalized(), depth))
      return true;
  }
  return false;
}

} // namespace svo
