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

#include <set>
#include <svo/map.h>
#include <svo/point.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <boost/bind.hpp>

namespace svo {

Map::Map() {}

Map::~Map()
{
  reset();
  SVO_INFO_STREAM("Map destructed");
}

 /********************************
 * @ function: 删除地图中关键帧
 *             候选点重置
 *             清空trash bin of kf & mp
 * @ param: 
 * 
 * @ note:
 *******************************/
void Map::reset()
{
  keyframes_.clear();
  point_candidates_.reset();
  emptyTrash();
}

/********************************
 * @ function: 删除关键帧
 * 
 * @ param: frame  要删除的frame共享指针
 * 
 * @ note:
 *******************************/
  //什么时候适合使用share_ptr??? 
bool Map::safeDeleteFrame(FramePtr frame)
{
  bool found = false;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
//[***step 1***] 从关键帧中找到frame
    if(*it == frame)
    {
//[***step 2***] 删除上面feature与point之间的联系
      std::for_each((*it)->fts_.begin(), (*it)->fts_.end(), [&](Feature* ftr){
        removePtFrameRef(it->get(), ftr);
      });
//[***step 3***] 删除这个关键帧      
      keyframes_.erase(it);
      found = true;
      break;
    }
  }
//[***step 4***] 从候选地图点
  point_candidates_.removeFrameCandidates(frame);

  if(found)
    return true;
//[***step 5***] 没找到输出error
  SVO_ERROR_STREAM("Tried to delete Keyframe in map which was not there.");
  return false;
}

/********************************
 * @ function: 删除point和feature之间的引用关系
 * 
 * @ param:   frame   要删除的关键帧
 *            ftr     frame与point之间的特征点
 *            
 * 
 * @ note:  point与frame是通过ftr连接的
 *******************************/
  // 删了之后是不是就剩下不包含3D point的feature，留着有用吗？
void Map::removePtFrameRef(Frame* frame, Feature* ftr)
{
  if(ftr->point == NULL)
    return; // mappoint may have been deleted in a previous ref. removal
//[***step 1***] 切断feature与point之间的联系
  Point* pt = ftr->point;
  ftr->point = NULL;
//[***step 2***] 该point观测少于2个则删除
  if(pt->obs_.size() <= 2)
  {
    // If the references list of mappoint has only size=2, delete mappoint
    safeDeletePoint(pt);
    return;
  }
//[***step 3***] 切断point与feature之间的联系，这里和step1不一样，互相指
  pt->deleteFrameRef(frame);  // Remove reference from map_point
//[***step 4***] 若ftr是keypoint则删除，keypoint要和3d point相连的
  frame->removeKeyPoint(ftr); // Check if mp was keyMp in keyframe
}

/********************************
 * @ function: 安全删除point，把point的obs_中每一个都删除
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void Map::safeDeletePoint(Point* pt)
{
  // Delete references to mappoints in all keyframes
  std::for_each(pt->obs_.begin(), pt->obs_.end(), [&](Feature* ftr){
    // 切断point与feature之间的联系
    ftr->point=NULL;
    // 是keypoint则删除
    ftr->frame->removeKeyPoint(ftr);
  });
  pt->obs_.clear();

  // Delete mappoint
  deletePoint(pt);
}

void Map::deletePoint(Point* pt)
{
  // point类型置为删除
  pt->type_ = Point::TYPE_DELETED;
  // 放入trash points
  trash_points_.push_back(pt);
}

// 增加关键帧
void Map::addKeyframe(FramePtr new_keyframe)
{
  keyframes_.push_back(new_keyframe);
}

/********************************
 * @ function: 找到接近frame的有共视关系的关键帧
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
void Map::getCloseKeyframes(
    const FramePtr& frame,
    std::list< std::pair<FramePtr,double> >& close_kfs) const
{
  for(auto kf : keyframes_)
  {
    // check if kf has overlaping field of view with frame, use therefore KeyPoints
//[***step 1***] 得到关键帧上的关键点
    for(auto keypoint : kf->key_pts_)
    {
      // 判断关键点有没有
      if(keypoint == nullptr)
        continue;
//[***step 2***] frame与各个关键帧是否有重叠
      if(frame->isVisible(keypoint->point->pos_))
      {
//[***step 3***]有则返回重叠的关键帧指针，和他们之间的距离close_kfs(pair list)
        close_kfs.push_back(
            std::make_pair(
              // 为什么这里的translation不求逆呢？？？
              // 因为这里的符号是一样的，即都是world原点在frame系下到frame的距离
              // 而getFurthestKeyframe函数里面的不一样
                kf, (frame->T_f_w_.translation()-kf->T_f_w_.translation()).norm()));
        break; // this keyframe has an overlapping field of view -> add to close_kfs
      }
    }
  }
}

/********************************
 * @ function: 从close_kfs中找出最近的
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
FramePtr Map::getClosestKeyframe(const FramePtr& frame) const
{
  // 获得close_kfs，为空则返回
  list< pair<FramePtr,double> > close_kfs;
  getCloseKeyframes(frame, close_kfs);
  if(close_kfs.empty())
  {
    return nullptr;
  }

  // 安照距离进行排序，sort还可以这样sort( a<b ）？？？对于自定义类型，牛X呀
  // Sort KFs with overlap according to their closeness
  close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                 boost::bind(&std::pair<FramePtr, double>::second, _2));

  // 判断下最接近的是不是它本身，不是才返回，是则弹出
  if(close_kfs.front().first != frame)
    return close_kfs.front().first;
  close_kfs.pop_front();
  return close_kfs.front().first;
}

// 找到离pos最远的关键帧
// 函数里面的pos是在world系下的到原点的距离
// 若还使用T_f_w.translation则是在frame系下与world原点距离
// 符号是相反的，必须求逆
FramePtr Map::getFurthestKeyframe(const Vector3d& pos) const
{
  FramePtr furthest_kf;
  double maxdist = 0.0;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    double dist = ((*it)->pos()-pos).norm();
    if(dist > maxdist) {
      maxdist = dist;
      furthest_kf = *it;
    }
  }
  return furthest_kf;
}

// 根据id找frame
bool Map::getKeyframeById(const int id, FramePtr& frame) const
{
  bool found = false;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
    if((*it)->id_ == id) {
      found = true;
      frame = *it;
      break;
    }
  return found;
}

/********************************
 * @ function: 把整个地图变换R, t, s
 * 
 * @ param:  旋转，平移，尺度
 * 
 * @ note: 还不懂这是干嘛的？？？ 好像是在更新or修正
 *******************************/
void Map::transform(const Matrix3d& R, const Vector3d& t, const double& s)
{
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    Vector3d pos = s*R*(*it)->pos() + t;
    // 这里为什么求个逆？？？
    Matrix3d rot = R*(*it)->T_f_w_.rotation_matrix().inverse();
    (*it)->T_f_w_ = SE3(rot, pos).inverse();
    for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
    {
      if((*ftr)->point == NULL)
        continue;
      // point对应于很多的ftr，可能重复变换，若是变换过赋值-1000
      if((*ftr)->point->last_published_ts_ == -1000)
        continue;
      (*ftr)->point->last_published_ts_ = -1000;
      (*ftr)->point->pos_ = s*R*(*ftr)->point->pos_ + t;
    }
  }
}


void Map::emptyTrash()
{
  // 删除point指针
  std::for_each(trash_points_.begin(), trash_points_.end(), [&](Point*& pt){
    delete pt; 
    pt=NULL;
  });
  // 清空
  trash_points_.clear();
  point_candidates_.emptyTrash();
}

// 装收敛的3D点，但还没有分配给关键帧
// 原来都是分配给关键帧啊
MapPointCandidates::MapPointCandidates()
{}

MapPointCandidates::~MapPointCandidates()
{
  reset();
}
// 创建候选点，来自收敛的种子点
// 直到下一个关键帧进来，这些点用来重投影和位姿优化
// 看完逆深度滤波，这里需要重新理解
void MapPointCandidates::newCandidatePoint(Point* point, double depth_sigma2)
{
  // 设置为候选类型
  point->type_ = Point::TYPE_CANDIDATE;
  // 上锁
  boost::unique_lock<boost::mutex> lock(mut_);
  // 把点和它最新的观测作为PointCandidate
  candidates_.push_back(PointCandidate(point, point->obs_.front()));
}

/********************************
 * @ function: 把候选特征点加到keyframe里
 * 
 * @ param: 
 * 
 * @ note: 原来不是投影上就加入的啊？？？
 *          这个feature哪里来的呢？？？不是图像提取的
 *          point对应的feature对应了frame，frame却没有feature？？？奇怪 19/1/6
 *******************************/
void MapPointCandidates::addCandidatePointToFrame(FramePtr frame)
{
  // 线程锁
  boost::unique_lock<boost::mutex> lock(mut_);
  PointCandidateList::iterator it=candidates_.begin();
  while(it != candidates_.end())
  {
    // point的最近观测若是与frame相同
    if(it->first->obs_.front()->frame == frame.get())
    {
      // 把这个点置为UNKNOWN
      // insert feature in the frame
      it->first->type_ = Point::TYPE_UNKNOWN;
      // 重投影失败次数0
      it->first->n_failed_reproj_ = 0;
      // 把特征加入到frame里
      it->second->frame->addFeature(it->second);
      // 从候选点删除
      it = candidates_.erase(it);
    }
    else
    // frame若不是最近的观测帧，略过
      ++it;
  }
}

// 候选点中删除point
bool MapPointCandidates::deleteCandidatePoint(Point* point)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  for(auto it=candidates_.begin(), ite=candidates_.end(); it!=ite; ++it)
  {
    if(it->first == point)
    {
      deleteCandidate(*it);
      candidates_.erase(it);
      return true;
    }
  }
  return false;
}

// 删除frame上feature对应的候选点
void MapPointCandidates::removeFrameCandidates(FramePtr frame)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  auto it=candidates_.begin();
  while(it!=candidates_.end())
  {
    if(it->second->frame == frame.get())
    {
      // 删除PointCandidate里的point和feature
      deleteCandidate(*it);
      // 从list中清楚指针
      it = candidates_.erase(it);
    }
    else
      ++it;
  }
}

// reset候选点队列，把所有的都给delete了？！！
void MapPointCandidates::reset()
{
  boost::unique_lock<boost::mutex> lock(mut_);
  std::for_each(candidates_.begin(), candidates_.end(), [&](PointCandidate& c){
    delete c.first;
    delete c.second;
  });
  candidates_.clear();
}

// 删除候选点
// trash作用是防止别的帧也有观测该点
void MapPointCandidates::deleteCandidate(PointCandidate& c)
{
  // camera-rig: another frame might still be pointing to the candidate point
  // therefore, we can't delete it right now.
  delete c.second; c.second=NULL; //从point里的obs_中删除
  // 把点置为delete状态，放入垃圾桶
  c.first->type_ = Point::TYPE_DELETED;
  trash_points_.push_back(c.first);
}

/********************************
 * @ function: 清空垃圾桶
 * 
 * @ param: 
 * 
 * @ note: 就这么直接删除point啊？？？直接相关的feature不管了嘛？
 *          一定是有什么清空的条件？？？
 *******************************/
void MapPointCandidates::emptyTrash()
{
  std::for_each(trash_points_.begin(), trash_points_.end(), [&](Point*& p){
    delete p; p=NULL;
  });
  trash_points_.clear();
}

// 收集debug和检查数据一致性
namespace map_debug {

// 检查地图
void mapValidation(Map* map, int id)
{
  // 检查地图中的keyframe
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    frameValidation(it->get(), id);
}

// 检查frame函数
// 之前各种删除，可能会有这种情况。。。point与frame本应该是双向的
// 之前说为什么feature留着，还真是留着了，他是中介，中介不能黄
void frameValidation(Frame* frame, int id)
{
  for(auto it = frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point==NULL)
      continue;
    // 该frame没有对应的点
    if((*it)->point->type_ == Point::TYPE_DELETED)
      printf("ERROR DataValidation %i: Referenced point was deleted.\n", id);
    // frame有对应的点，点没对应的frame
    if(!(*it)->point->findFrameRef(frame))
      printf("ERROR DataValidation %i: Frame has reference but point does not have a reference back.\n", id);
    // 验证点
    pointValidation((*it)->point, id);
  }
  // 验证keyframe的keypoint是否正确
  for(auto it=frame->key_pts_.begin(); it!=frame->key_pts_.end(); ++it)
    if(*it != NULL)
      if((*it)->point == NULL)
        printf("ERROR DataValidation %i: KeyPoints not correct!\n", id);
}

// emmmm，这个就有意思了
// point->feature->frame->feature->point是不是原来的feature，不一样就不一致了
void pointValidation(Point* point, int id)
{
  for(auto it=point->obs_.begin(); it!=point->obs_.end(); ++it)
  {
    bool found=false;
    for(auto it_ftr=(*it)->frame->fts_.begin(); it_ftr!=(*it)->frame->fts_.end(); ++it_ftr)
     if((*it_ftr)->point == point) {
       found=true; break;
     }
    if(!found)
      printf("ERROR DataValidation %i: Point %i has inconsistent reference in frame %i, is candidate = %i\n", id, point->id_, (*it)->frame->id_, (int) point->type_);
  }
}

// 地图统计
void mapStatistics(Map* map)
{
  // compute average number of features which each frame observes
  // 平均每个关键帧有多少个特征点
  size_t n_pt_obs(0);
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    n_pt_obs += (*it)->nObs();
  printf("\n\nMap Statistics: Frame avg. point obs = %f\n", (float) n_pt_obs/map->size());

  // compute average number of observations that each point has
  size_t n_frame_obs(0);
  size_t n_pts(0);
  // 这个Point没有重载过 < 啊，怎么排的序呢？？？
  std::set<Point*> points;
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
  {
    for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
    {
      if((*ftr)->point == NULL)
        continue;
        // set中不会有值一样的元素，会排序，难道是比的地址？？
        // set用来过滤掉指向相同对象的指针
      if(points.insert((*ftr)->point).second) {
        ++n_pts; //统计map有多少个点
        n_frame_obs += (*ftr)->point->nRefs(); //统计point有多少个特征点对应
      }
    }
  }
  printf("Map Statistics: Point avg. frame obs = %f\n\n", (float) n_frame_obs/n_pts);
}

} // namespace map_debug
} // namespace svo
