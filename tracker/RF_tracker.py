import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F


from .RF_filter import KalmanFilter
from .RF_matching import calc_distance, linear_assignment
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, xywhr, score, feat=None, feat_history=50):

        self._xywhr = np.asarray(xywhr, dtype=float)
        self._mean_sigma,covdeg  = self.xywhr_to_meancov(self._xywhr)
        self.covDeg = covdeg 
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.newTrack = self._xywhr

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
            mean_state[8] = 0
            mean_state[9] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
                    multi_mean[i][8] = 0
                    multi_mean[i][9] = 0
                
                if (multi_mean[i][2] * multi_mean[i][3]) < multi_mean[i][4]**2:
                     multi_mean[i][9] = 0

            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self._mean_sigma)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, 
                                                            new_track.meancov)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.newTrack = new_track._xywhr

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.meancov)

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def meancov(self):
        """
        Get current position in Gaussian format.
        """
        if self.mean is None:
            return self._mean_sigma.copy()
        
        ret = self.mean.copy()
        return ret
    
    @property
    def xywhr(self):
        if self.mean is None:
            ret = self._xywhr.copy()
            return ret
        
        ret = self.newTrack.copy()
        return ret

    @property
    def covMatrix(self):
        """
        The Gaussian distibution of the current track.
        """

        mean = np.array(self.meancov[:2])
        cov = self.meancov[2:]
        cov = np.array([cov[0],self.covDeg[0]*cov[2],
                        self.covDeg[1]*cov[2],cov[1]]).reshape([2,2])

        ret = [mean,cov]
        
        return ret


    @property
    def xywh(self):
        """
        Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.xywhr[:4].copy()
        return ret
    

    @staticmethod
    def xywhr_to_meancov(xywhr):
        """
        transform xywhr to Gaussian distribution.
        """
        x,y = xywhr[:2]
        wh = xywhr[2:4]
        r =  xywhr[-1]
        r = np.deg2rad(r)
    
        cos_r = np.cos(r)
        sin_r = np.sin(r)
        R = np.stack((cos_r, -sin_r, sin_r, cos_r)).reshape(2, 2)
        S = 0.5 * np.diag(wh)

        Rt = np.transpose(R)
        
        sigma = np.matmul(np.matmul(R,np.square(S)), Rt)

        covDeg = [1,1] if sigma[0,1] ==0 else [sigma[0,1],sigma[1,0]]/abs(sigma[0,1])

        return [x,y,sigma[0,0],sigma[1,1],abs(sigma[0,1])],covDeg

    @property
    def meanxywhr():
        """
        Convert Gaussian distribution to format (xywhr)
        """
        gaus = self.meancov
        mat = [[gaus[2],gaus[4]],[gaus[4],gaus[3]]]
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        stds = np.sqrt(eigenvalues)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        angle = np.degrees(angle)
        w,h = 2*stds[0],2*stds[1]
        return [gaus[0],gaus[1],w,h,angle]

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


'''Main Tracker code '''
class RFTrack(object):
    def __init__(self, args):
        self.reset(args)
        self.args = args
        self.track_thresh = args.track_thresh
        self.track_buffer = args.track_buffer
        self.match_thresh = args.match_thresh

        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int((args.fps / 30.0 )* self.track_buffer)
        self.max_time_lost = self.buffer_size     

        self.match1 = 0.65 
        self.match2 = 0.7

    def reset(self,args):
        BaseTrack._count = 0
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()

    def update(self, output_results):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        '''cx,cy,w,h,angle'''
        scores = output_results[:, 5]
        bboxes = output_results[:, :5]

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        '''High Score detections'''
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]

        '''Low Score detections'''
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]

        '''Detections to track'''
        if len(dets) > 0:

            detections = [STrack(box, s) for
                          (box, s) in zip(dets, scores_keep)]
        else:
            detections = []


        ''' Split currently tracked into last seen or not'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        dists = calc_distance(strack_pool, detections,match=self.args.match,
                                                            ratio=self.args.buff_ratio,
                                                            dynamic=self.args.buff_dynamic,
                                                            frame_size=self.args.frame_size)

        
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match1)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        
        ''' Step 3: Second association, with low score detection boxes'''
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(tlbr, s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = calc_distance(r_tracked_stracks, detections_second,match=self.args.match,
                                                            ratio=self.args.buff_ratio,
                                                            dynamic=self.args.buff_dynamic,
                                                            frame_size=self.args.frame_size)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=self.match1)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = calc_distance(unconfirmed, detections,match=self.args.match,
                                                            ratio=self.args.buff_ratio,
                                                            dynamic=self.args.buff_dynamic,
                                                            frame_size=self.args.frame_size)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=self.match2)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]

            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
    
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks,match=self.args.match)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb,match ):
    pdist = calc_distance(stracksa, stracksb,match=match,dynamic=False,ratio=1.0)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
