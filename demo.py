import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
import json
from loguru import logger
import imageio.v2 as imageio  
import sys

from f_utils.visualize import plot_tracking,plot_r_tracking
from f_utils.timer import Timer

from tracker.RF_tracker import RFTrack


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
ROT_TRACKERS = ['RFTrackB','RFTrackK',RFTrack]

def make_parser():
    parser = argparse.ArgumentParser("RF-TRACKER Demo!")
    
    parser.add_argument(
        "--path", default='example.gif', help="path to images or video"

    )

    parser.add_argument(
        "--dets", default='example_dets.json', help="path to json of detections"
    )

    parser.add_argument(
        "--save_result",
        action="store_true",
        default=True,
        help="whether to save the inference result of image/video",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
        
    parser.add_argument("--cut_off", type=int, default=0, help="early cutt off for experimentation")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    
    parser.add_argument("--match", default="gwd", help="Association type, e.g gwd, bd, kld")
    parser.add_argument("--buff_dynamic", type=bool, default=True, help="if buffer is dyanmic or static")
    parser.add_argument("--buff_ratio", type=float, default=1.3, help="maximum expansion of the buffer")

    return parser

def vid_to_imglist(vid_path):
    output_folder = 'demo_frames'  
    os.makedirs(output_folder, exist_ok=True)  
    
    print("Transforming video to folder of frames")
    # Read the GIF  
    decoded_vid = imageio.get_reader(vid_path)  
    
    # Iterate through each frame and save it as an image  
    for i, frame in enumerate(decoded_vid):  
        imageio.imsave(os.path.join(output_folder, f"demo_{i+1:03d}.png"), frame)

def get_image_list(path):
    image_paths = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_paths.append(apath)
    return image_paths

def get_frame_info(image_path):
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape 
    return frame, height, width

def get_img_info(image_path,label_map):
    
    img_num = image_path.split('.')[0]
    img_num = img_num.split(os.path.sep)[-1]

    if img_num in label_map:
        labels = label_map[img_num]
        labels = np.array(labels).astype(np.float64)
    else:
        labels = None

    frame, height, width = get_frame_info(image_path)
    info = {
        'raw_img': frame,
        "height": height,
        "width" :width,
    }

    return labels,info

def create_det_labels(path,with_rot=True):
    with open(path,"r") as f:
        labels = json.load(f)
    lbl_map = {}

    for lbl in labels:
        frame = lbl["img_name"]
        res = []
        
        for bbox,score in zip(lbl["bboxes"].copy(), lbl["scores"]):
            if with_rot:
                assert len(bbox) ==5 
            else:
                assert len(bbox) ==4

            res.append(bbox + [score]) 
        
        if res != []:
            if frame in lbl_map:
                lbl_map[frame ] += res
            else:
                lbl_map[frame ] = res
    return lbl_map 

def store_res(frame_id,track, with_rot=True,min_box_area=10):

    boxes,ids,scores = [],[],[]
    res = []


    box = track.xywhr if with_rot else track.tlwh
    tid = track.track_id
    score = track.score

    if box[2] * box[3] > min_box_area:
        boxes = [box]
        ids = [tid]
        scores = [score]
        
        # save results in a txt format
        if with_rot:
            save_format = '{frame},{id},{x1},{y1},{w},{h},{r},{s},-1,-1,-1\n'
            x1,y1,w,h,r = box
            res.append(
                save_format.format(frame=frame_id, id=tid, 
                                x1=round(x1, 1), y1=round(y1, 1), 
                                w=round(w, 1), h=round(h, 1),
                                r=round(r, 1), s=round(score,4))
            )
        else:

            save_format = '{frame},{id},{x},{y},{w},{h},0,{s},-1,-1,-1\n'
            t,l,w,h = box

            x = t + w/2
            y = l + h/2

            res.append(
                save_format.format(frame=frame_id, id=tid, 
                                x=round(x, 1), y=round(y, 1), 
                                w=round(w, 1), h=round(h, 1), s=round(score,4))
            )


    return boxes,ids,scores,res


def track(out_folder,
            img_path,
            det_path,
            args,
            out_name = None,
            frame_start = 0,
            with_rot=True,
            out_video =True ):
    
    files = get_image_list(img_path)

    files.sort()

    if out_name is None:
        current_time = time.localtime()
        out_name = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

    
    video_folder = osp.join(out_folder, 'video')
    os.makedirs(video_folder, exist_ok=True)
    det_folder = osp.join(out_folder, 'dets')
    os.makedirs(det_folder, exist_ok=True)
    
    _, height, width = get_frame_info(files[0])
    if out_video:
        save_path = osp.join(video_folder, f"{out_name}.mp4")
        
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (int(width), int(height))
        )

    args.frame_size = [int(height),int(width)]

    print(f"Frame size: width: {int(width)} , height:{int(height)}")
    print(f"Association method : {args.match}")
    
  
    tracker = RFTrack(args)
    f_frame = osp.basename(files[0]).split('_')[-1].split('.')[0]
    f_frame = (int(f_frame))

    

    timer = Timer()
    results = []

    print("Making labels")  

    # Create a hashmap of all the detection in each frame
    lbl_map = create_det_labels(det_path,with_rot=with_rot)

    # Break early if there is a cut off argument
    cut_off = True if args.cut_off != 0 else False

    print("Starting...")

    for frame_id, img_path in enumerate(files, frame_start):
        timer.tic()
        outputs, img_info = get_img_info(img_path,lbl_map)

        if cut_off:
            if frame_id == args.cut_off : break

        real_f_id = frame_id + f_frame
        if outputs is not None:
            # update the tracker with new bounding box frames
            online_targets = tracker.update(outputs)
            
            online_tlwhs = []
            online_ids = []
            online_scores = []

            #  store the new tracking results
            for t in online_targets:
                tboxes,tids,tscores,tres = store_res(real_f_id, t, with_rot,args.min_box_area)
                online_tlwhs += tboxes
                online_ids += tids
                online_scores += tscores
                results += tres
        timer.toc()
        if out_video:
            if outputs is not None:
                
                drawer = plot_r_tracking if with_rot else plot_tracking
                    
                online_im = drawer(
                            img_info['raw_img'], online_tlwhs, 
                            online_ids, frame_id=real_f_id, 
                            fps= 1. / timer.average_time)

            else:
                online_im = img_info['raw_img']

    
            vid_writer.write(online_im)

            
        if frame_id % 150 == 0:
            logger.info('Processing frame {} ({:.5f} fps)'.format(real_f_id, 1/max(1e-5, timer.average_time)))


    res_file = osp.join(det_folder, f"{out_name}.txt")
    with open(res_file, 'w') as f:
        f.writelines(results)
    if out_video:
        vid_writer.release()
    
    logger.info(f"Results saved at {osp.join(video_folder, out_name)}.mp4")
    

def single_run(args):
    vis_folder = osp.join(os.getcwd(), "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

    if os.path.isdir(args.path):
        img_path = args.path
    else:
        path_ext = os.path.splitext(args.path)[1]
        if path_ext not in ['.gif','.mp4']:
            raise ValueError("Error: If the path is a video, it has to be in gif or mp4.")
        vid_to_imglist(args.path) # comment this line if you do not want to recreate the image folder
        img_path = "demo_frames"
    
    assert os.path.splitext(args.dets)[1] == '.json'
    det_path = args.dets
    

    track( 
        vis_folder,
        img_path,
        det_path,
        args= args,
        with_rot = True)



if __name__ == "__main__":
    args = make_parser().parse_args()
    single_run(args)
