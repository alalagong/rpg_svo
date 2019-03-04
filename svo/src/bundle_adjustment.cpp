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

#include <vikit/math_utils.h>
#include <boost/thread.hpp>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>
#include <svo/bundle_adjustment.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/config.h>
#include <svo/map.h>

#define SCHUR_TRICK 1

namespace svo {
namespace ba {

/********************************
 * @ function:    初始化之后的 优化
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void twoViewBA(
    Frame* frame1,
    Frame* frame2,
    double reproj_thresh,
    Map* map)
{
  // scale reprojection threshold in pixels to unit plane
  reproj_thresh /= frame1->cam_->errorMultiplier2(); // 转化到单位平面

  // init g2o
  //[ ***step 1*** ] 配置图优化的参数
  g2o::SparseOptimizer optimizer;
  setupG2o(&optimizer);

  list<EdgeContainerSE3> edges;
  size_t v_id = 0;

  //[ ***step 2*** ] 增加两个相机位姿顶点
  // New Keyframe Vertex 1: This Keyframe is set to fixed!
  g2oFrameSE3* v_frame1 = createG2oFrameSE3(frame1, v_id++, true);
  optimizer.addVertex(v_frame1);

  // New Keyframe Vertex 2
  g2oFrameSE3* v_frame2 = createG2oFrameSE3(frame2, v_id++, false);
  optimizer.addVertex(v_frame2);

  // Create Point Vertices
  //[ ***step 3*** ] 增加地图点的顶点, 以及和相机所连成的边
  for(Features::iterator it_ftr=frame1->fts_.begin(); it_ftr!=frame1->fts_.end(); ++it_ftr)
  {
    Point* pt = (*it_ftr)->point;
    if(pt == NULL)
      continue;
    g2oPoint* v_pt = createG2oPoint(pt->pos_, v_id++, false);
    optimizer.addVertex(v_pt);
    pt->v_pt_ = v_pt; //放入做临时的G2O定点
    g2oEdgeSE3* e = createG2oEdgeSE3(v_frame1, v_pt, vk::project2d((*it_ftr)->f), true, reproj_thresh*Config::lobaRobustHuberWidth());
    optimizer.addEdge(e);
    edges.push_back(EdgeContainerSE3(e, frame1, *it_ftr)); // TODO feature now links to frame, so we can simplify edge container!

    // find at which index the second frame observes the point
    //* 找到和该点所对应的在 frame2 上的特征点, 构成边
    Feature* ftr_frame2 = pt->findFrameRef(frame2);
    e = createG2oEdgeSE3(v_frame2, v_pt, vk::project2d(ftr_frame2->f), true, reproj_thresh*Config::lobaRobustHuberWidth());
    optimizer.addEdge(e);
    edges.push_back(EdgeContainerSE3(e, frame2, ftr_frame2));
  }

  // Optimization
  //[ ***step 4*** ] 运行求解器
  double init_error, final_error;
  runSparseBAOptimizer(&optimizer, Config::lobaNumIter(), init_error, final_error);
  printf("2-View BA: Error before/after = %f / %f\n", init_error, final_error);

  //[ ***step 5*** ] 使用优化后的相机位姿和点位置进行更新
  // Update Keyframe Positions
  frame1->T_f_w_.rotation_matrix() = v_frame1->estimate().rotation().toRotationMatrix();
  frame1->T_f_w_.translation() = v_frame1->estimate().translation();
  frame2->T_f_w_.rotation_matrix() = v_frame2->estimate().rotation().toRotationMatrix();
  frame2->T_f_w_.translation() = v_frame2->estimate().translation();

  // Update Mappoint Positions
  for(Features::iterator it=frame1->fts_.begin(); it!=frame1->fts_.end(); ++it)
  {
    if((*it)->point == NULL)
     continue;
    (*it)->point->pos_ = (*it)->point->v_pt_->estimate();
    (*it)->point->v_pt_ = NULL;
  }

  //[ ***step 6*** ] 沿着重投影误差大的, 则从地图中删除, 从edge中删除
  // Find Mappoints with too large reprojection error
  const double reproj_thresh_squared = reproj_thresh*reproj_thresh;
  size_t n_incorrect_edges = 0;
  for(list<EdgeContainerSE3>::iterator it_e = edges.begin(); it_e != edges.end(); ++it_e)
    if(it_e->edge->chi2() > reproj_thresh_squared)
    {
      if(it_e->feature->point != NULL)
      {
        map->safeDeletePoint(it_e->feature->point);
        it_e->feature->point = NULL;
      }
      ++n_incorrect_edges;
    }

  printf("2-View BA: Wrong edges =  %zu\n", n_incorrect_edges);
}

/********************************
 * @ function:  
 * 
 * @ param:     Frame* center_kf                  不知道干啥用的???
 *              set<FramePtr>* core_kfs           待优化的核心关键帧
 *              Map* map                          地图     
 *              size_t& n_incorrect_edges_1       
 *              size_t& n_incorrect_edges_2       重投影误差过大的边数目
 *              double& init_error,
 *              double& final_error
 * 
 * @ note:
 *******************************/
void localBA(
    Frame* center_kf,
    set<FramePtr>* core_kfs,
    Map* map,
    size_t& n_incorrect_edges_1,
    size_t& n_incorrect_edges_2,
    double& init_error,
    double& final_error)
{

  // init g2o
  g2o::SparseOptimizer optimizer;
  setupG2o(&optimizer);

  list<EdgeContainerSE3> edges;
  set<Point*> mps;            // 要优化的地图点
  list<Frame*> neib_kfs;      // 和 core_kfs 具有共视关系的帧
  size_t v_id = 0;            // 顶点的ID号
  size_t n_mps = 0;           // 加入的地图点的个数
  size_t n_fix_kfs = 0;       // 固定的帧的个数
  size_t n_var_kfs = 1;       // 作为G2O变量顶点的关键帧数目
  size_t n_edges = 0;         // 边的个数
  n_incorrect_edges_1 = 0;
  n_incorrect_edges_2 = 0;    

  // Add all core keyframes
  //[ ***step 1*** ] 把 core_kfs 加入到优化器的顶点, 把其可观察到的地图点加入mps
  for(set<FramePtr>::iterator it_kf = core_kfs->begin(); it_kf != core_kfs->end(); ++it_kf)
  {
    g2oFrameSE3* v_kf = createG2oFrameSE3(it_kf->get(), v_id++, false);
    (*it_kf)->v_kf_ = v_kf;
    ++n_var_kfs;
    assert(optimizer.addVertex(v_kf));

    // all points that the core keyframes observe are also optimized:
    //* 循环每一关键帧的特征点
    for(Features::iterator it_pt=(*it_kf)->fts_.begin(); it_pt!=(*it_kf)->fts_.end(); ++it_pt)
      if((*it_pt)->point != NULL)
        mps.insert((*it_pt)->point);
  }

  // Now go throug all the points and add a measurement. Add a fixed neighbour
  // Keyframe if it is not in the set of core kfs
  //? 误差都是怎么定的???
  double reproj_thresh_2 = Config::lobaThresh() / center_kf->cam_->errorMultiplier2(); // 创建边时的误差
  //? 这个没有用啊???
  double reproj_thresh_1 = Config::poseOptimThresh() / center_kf->cam_->errorMultiplier2(); // 位姿误差阈值
  double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
  //[ ***step 2*** ] 把 mps 里面的点加入到优化器顶点
  for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
  {
    // Create point vertex
    g2oPoint* v_pt = createG2oPoint((*it_pt)->pos_, v_id++, false);
    (*it_pt)->v_pt_ = v_pt;
    assert(optimizer.addVertex(v_pt));
    ++n_mps;

    // Add edges
   //[ ***step 3*** ] 将3D点被观测的帧也加入到优化器, 属于固定的帧
    list<Feature*>::iterator it_obs=(*it_pt)->obs_.begin();
    while(it_obs!=(*it_pt)->obs_.end())
    {
      //? 这个 error 有啥用???
      Vector2d error = vk::project2d((*it_obs)->f) - vk::project2d((*it_obs)->frame->w2f((*it_pt)->pos_));
      //* 把未加入的加入
      if((*it_obs)->frame->v_kf_ == NULL)
      {
        // frame does not have a vertex yet -> it belongs to the neib kfs and
        // is fixed. create one:
        g2oFrameSE3* v_kf = createG2oFrameSE3((*it_obs)->frame, v_id++, true);
        (*it_obs)->frame->v_kf_ = v_kf;
        ++n_fix_kfs;
        assert(optimizer.addVertex(v_kf));
        neib_kfs.push_back((*it_obs)->frame);
      }
  //[ ***step 4*** ] 将所有的这些点构成边, 加入到优化中
      // create edge
      g2oEdgeSE3* e = createG2oEdgeSE3((*it_obs)->frame->v_kf_, v_pt,
                                       vk::project2d((*it_obs)->f),
                                       true,
                                       reproj_thresh_2*Config::lobaRobustHuberWidth(),
                                       1.0 / (1<<(*it_obs)->level)); // 金字塔层做权重
      assert(optimizer.addEdge(e));
      edges.push_back(EdgeContainerSE3(e, (*it_obs)->frame, *it_obs));
      ++n_edges;
      ++it_obs;
    }
  }
  //[ ***step 5*** ] 先单独对点进行优化, 再对点和位姿进行联合优化
  // structure only
  g2o::StructureOnlySolver<3> structure_only_ba; // 点的维度是3
  g2o::OptimizableGraph::VertexContainer points;
  for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it)
  {
    //* mVertexIDMap 是 map<ID, vertex>
    g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
    //* 是3D地图点的图优化顶点并且所连的边大于2, 则放入点容器 points
      if (v->dimension() == 3 && v->edges().size() >= 2)
        points.push_back(v);
  }
  //* 只优化这些地图点, 固定帧位姿. 用在联合优化之前或之后.
  structure_only_ba.calc(points, 10);

  // Optimization 
  if(Config::lobaNumIter() > 0)
    runSparseBAOptimizer(&optimizer, Config::lobaNumIter(), init_error, final_error);

  //[ ***step 6*** ] 对优化的帧和点进行更新, 共视帧不更新位姿
  // Update Keyframes
  for(set<FramePtr>::iterator it = core_kfs->begin(); it != core_kfs->end(); ++it)
  {
    (*it)->T_f_w_ = SE3( (*it)->v_kf_->estimate().rotation(),
                         (*it)->v_kf_->estimate().translation());
    (*it)->v_kf_ = NULL;
  }
  //* 固定的帧, 优化了但是我不更新, 这样可以么???
  //? 这样会不会不一致???
  for(list<Frame*>::iterator it = neib_kfs.begin(); it != neib_kfs.end(); ++it)
    (*it)->v_kf_ = NULL;

  // Update Mappoints
  for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it)
  {
    (*it)->pos_ = (*it)->v_pt_->estimate();
    (*it)->v_pt_ = NULL;
  }
  //[ ***step 7*** ] 如果重投影误差过大, 则将 point 和 feature 之间的的约束切断
  // Remove Measurements with too large reprojection error
  double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
  for(list<EdgeContainerSE3>::iterator it = edges.begin(); it != edges.end(); ++it)
  {
    if(it->edge->chi2() > reproj_thresh_2_squared) //*(1<<it->feature_->level))
    {
      map->removePtFrameRef(it->frame, it->feature);
      ++n_incorrect_edges_2;
    }
  }

  // TODO: delete points and edges!
  //* 乘以焦距 f 转化为像素误差
  init_error = sqrt(init_error)*center_kf->cam_->errorMultiplier2();
  final_error = sqrt(final_error)*center_kf->cam_->errorMultiplier2();
}


void globalBA(Map* map)
{
  // init g2o
  //[ ***step 1*** ] 设置求解器
  g2o::SparseOptimizer optimizer;
  setupG2o(&optimizer);

  list<EdgeContainerSE3> edges;
  list< pair<FramePtr,Feature*> > incorrect_edges;

  // Go through all Keyframes
  size_t v_id = 0;
  double reproj_thresh_2 = Config::lobaThresh() / map->lastKeyframe()->cam_->errorMultiplier2();
  double reproj_thresh_1_squared = Config::poseOptimThresh() / map->lastKeyframe()->cam_->errorMultiplier2();
  reproj_thresh_1_squared *= reproj_thresh_1_squared;
  
  for(list<FramePtr>::iterator it_kf = map->keyframes_.begin();
      it_kf != map->keyframes_.end(); ++it_kf)
  {
    //[ ***step 2*** ] 把关键帧作为优化顶点加入
    // New Keyframe Vertex
    g2oFrameSE3* v_kf = createG2oFrameSE3(it_kf->get(), v_id++, false);
    (*it_kf)->v_kf_ = v_kf;
    optimizer.addVertex(v_kf);
    //[ ***step 3*** ] 把关键帧上的地图点加入到优化顶点
    for(Features::iterator it_ftr=(*it_kf)->fts_.begin(); it_ftr!=(*it_kf)->fts_.end(); ++it_ftr)
    {
      // for each keyframe add edges to all observed mapoints
      Point* mp = (*it_ftr)->point;
      if(mp == NULL)
        continue;
      g2oPoint* v_mp = mp->v_pt_;
      if(v_mp == NULL)
      {
        // mappoint-vertex doesn't exist yet. create a new one:
        v_mp = createG2oPoint(mp->pos_, v_id++, false);
        mp->v_pt_ = v_mp;
        optimizer.addVertex(v_mp);
      }
    //[ ***step 4*** ] 防止重投影误差特别大的影响结果, 设置阈值进行滤除, 其余的用于构建边
      // Due to merging of mappoints it is possible that references in kfs suddenly
      // have a very large reprojection error which may result in distorted results.
      Vector2d error = vk::project2d((*it_ftr)->f) - vk::project2d((*it_kf)->w2f(mp->pos_));
      if(error.squaredNorm() > reproj_thresh_1_squared)
        incorrect_edges.push_back(pair<FramePtr,Feature*>(*it_kf, *it_ftr));
      else
      {
        g2oEdgeSE3* e = createG2oEdgeSE3(v_kf, v_mp, vk::project2d((*it_ftr)->f),
                                         true,
                                         reproj_thresh_2*Config::lobaRobustHuberWidth());

        edges.push_back(EdgeContainerSE3(e, it_kf->get(), *it_ftr));
        optimizer.addEdge(e);
      }
    }
  }
  //[ ***step 5*** ] 优化
  // Optimization
  double init_error=0.0, final_error=0.0;
  if(Config::lobaNumIter() > 0)
    runSparseBAOptimizer(&optimizer, Config::lobaNumIter(), init_error, final_error);
  //[ ***step 6*** ] 更新关键帧和地图点位姿
  // Update Keyframe and MapPoint Positions
  for(list<FramePtr>::iterator it_kf = map->keyframes_.begin();
        it_kf != map->keyframes_.end(); ++it_kf)
  {
    (*it_kf)->T_f_w_ = SE3( (*it_kf)->v_kf_->estimate().rotation(),
                            (*it_kf)->v_kf_->estimate().translation());
    (*it_kf)->v_kf_ = NULL;
    for(Features::iterator it_ftr=(*it_kf)->fts_.begin(); it_ftr!=(*it_kf)->fts_.end(); ++it_ftr)
    {
      Point* mp = (*it_ftr)->point;
      if(mp == NULL)
        continue;
      if(mp->v_pt_ == NULL)
        continue;       // mp was updated before
      mp->pos_ = mp->v_pt_->estimate();
      mp->v_pt_ = NULL;
    }
  }
  //[ ***step 7*** ] 移除之前重投影误差大的, 和优化之后重投影误差大的
  // Remove Measurements with too large reprojection error
  for(list< pair<FramePtr,Feature*> >::iterator it=incorrect_edges.begin();
      it!=incorrect_edges.end(); ++it)
    map->removePtFrameRef(it->first.get(), it->second);

  double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
  for(list<EdgeContainerSE3>::iterator it = edges.begin(); it != edges.end(); ++it)
  {
    if(it->edge->chi2() > reproj_thresh_2_squared)
    {
      map->removePtFrameRef(it->frame, it->feature);
    }
  }
}

void setupG2o(g2o::SparseOptimizer * optimizer)
{
  // TODO: What's happening with all this HEAP stuff? Memory Leak?
  optimizer->setVerbose(false);

#if SCHUR_TRICK
  // solver
  // g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
  //* 线性求解器是6自由度相机, 3自由度点, 用于3D SLAM
  //[ ***step 1*** ] 选择Cholmod求解器
  auto linearSolver =  g2o::make_unique< g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
  //linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
  //[ ***step 2*** ] 块求解的类型
  auto solver_ptr = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
  //[ ***step 3*** ] LM算法进行求解, 迭代策略
  g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr));
#else
  // g2o::BlockSolverX::LinearSolverType * linearSolver;
  auto linearSolver = g2o::make_unique< g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>> ();
  //linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>();
  auto solver_ptr = g2o::make_unique <g2o::BlockSolverX> (std::move(linearSolver));
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr));
#endif
  //* 失败后最多迭代5次
  solver->setMaxTrialsAfterFailure(5);
  //* 设置算法 
  optimizer->setAlgorithm(solver);

  // setup camera
  //* 相机焦距=1.0,  光心偏移,  基线
  g2o::CameraParameters * cam_params = new g2o::CameraParameters(1.0, Vector2d(0.,0.), 0.);
  cam_params->setId(0);
  if (!optimizer->addParameter(cam_params)) {
    assert(false);
  }
}

void
runSparseBAOptimizer(g2o::SparseOptimizer* optimizer,
                     unsigned int num_iter,
                     double& init_error, double& final_error)
{
  optimizer->initializeOptimization();
  optimizer->computeActiveErrors();
  init_error = optimizer->activeChi2();
  optimizer->optimize(num_iter);
  final_error = optimizer->activeChi2();
}

g2oFrameSE3*
createG2oFrameSE3(Frame* frame, size_t id, bool fixed)
{
  g2oFrameSE3* v = new g2oFrameSE3();
  v->setId(id);
  v->setFixed(fixed);
  v->setEstimate(g2o::SE3Quat(frame->T_f_w_.unit_quaternion(), frame->T_f_w_.translation())); // 初值
  return v;
}

g2oPoint*
createG2oPoint(Vector3d pos,
               size_t id,
               bool fixed)
{
  g2oPoint* v = new g2oPoint();
  v->setId(id);
#if SCHUR_TRICK
  v->setMarginalized(true);
#endif
  v->setFixed(fixed);  // 是否固定
  v->setEstimate(pos);
  return v;
}

/********************************
 * @ function:  构建G2O优化的边
 * 
 * @ param:     g2oFrameSE3* v_frame    优化的相机位姿
 *              g2oPoint* v_point       优化的点
 *              const Vector2d& f_up    相机观测的初值
 *              bool robust_kernel      核函数
 *              double huber_width      核函数宽度
 *              double weight           权重
 * 
 * @ note:
 *******************************/
g2oEdgeSE3*
createG2oEdgeSE3( g2oFrameSE3* v_frame,
                  g2oPoint* v_point,
                  const Vector2d& f_up,
                  bool robust_kernel,
                  double huber_width,
                  double weight)
{
  g2oEdgeSE3* e = new g2oEdgeSE3();
  e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
  e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
  e->setMeasurement(f_up);
  e->information() = weight * Eigen::Matrix2d::Identity(2,2); // 信息矩阵, 加权最小二乘
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();      // TODO: memory leak
  rk->setDelta(huber_width);
  e->setRobustKernel(rk);
  //? 这是干嘛的???
  e->setParameterId(0, 0); //old: e->setId(v_point->id());
  return e;
}

} // namespace ba
} // namespace svo
