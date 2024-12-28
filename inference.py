from ultralytics import YOLO
import cv2
import numpy as np
import time



def adjust_hand_data(model_result):
    return model_result[0].boxes.data.cpu().numpy()


def match_hand(hand_result,pose_result):
    pose_listed = pose_result[0].keypoints.data.cpu().numpy()
    pose_result_extended = np.zeros((pose_listed.shape[0],19,3))
    if pose_listed.shape[1]==0:
        return pose_result_extended
    pose_result_extended[:,:17,:] = pose_listed.copy()
    list_wrist_poses = pose_listed[:,9:11,:2].reshape(pose_listed.shape[0]*2,2)
    temp_hand = []
    for hand in hand_result:
        hand[0:2] = (hand[0:2]+hand[2:4])/2
        dist = np.sqrt(np.sum((list_wrist_poses - hand[0:2])**2, axis=1))
        nearest_idx = np.argmin(dist)
        
        if nearest_idx//2 not in temp_hand:
            pose_result_extended[nearest_idx//2,17,:2] = hand[0:2]
            pose_result_extended[nearest_idx//2,17,2] = hand[4]
            temp_hand.append(nearest_idx//2)
        else:
            pose_result_extended[nearest_idx//2,18,:2] = hand[0:2]
            pose_result_extended[nearest_idx//2,18,2] = hand[4]
        list_wrist_poses[nearest_idx,:] = np.array([10000,10000])
    return pose_result_extended

def plot_poses(frame, pose_array,show_hand_conf):
    color = (50,50,255)
    for pose in pose_array:
        for i in range(19):
            x, y, conf = pose[i]
            x = int(x)  
            y = int(y)  
            frame = cv2.circle(frame, (x, y), radius=3, color=color[j%2], thickness=-1)
        if show_hand_conf:
            hand_indices = [17, 18]
            for idx in hand_indices:
                if idx < 19:  # safety check
                    x, y, conf = pose[idx]
                    x, y = int(x), int(y)
                    conf_text = f"{conf:.2f}"
                    frame = cv2.putText(
                        frame,
                        conf_text,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255,50,50),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )
    return frame


def main():
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        ret, frame = cap.read()
        if not ret:
            continue
        pose_result = pose_model.track(frame, verbose=False, conf=0.4,persist=True)
        hand_result = hand_model(frame,verbose=False,conf=0.3,iou=0.3)
        hand_result_listed = adjust_hand_data(model_result=hand_result)
        if len(pose_result) > 0:
            new_pose_with_hands = match_hand(hand_result=hand_result_listed,pose_result=pose_result)
            out_frame = plot_poses(frame=frame.copy(),pose_array=new_pose_with_hands,show_hand_conf=True)
        else:
            cap.stop()
            cv2.destroyAllWindows()
            cap = cv2.VideoCapture(stream)
            time.sleep(1)
        cv2.imshow("frame", out_frame)
        if key == ord('q'):
            break
    
    

if __name__ == "__main__":
    stream = 0 # webcam
    cap = cv2.VideoCapture(stream)
    hand_model = YOLO("hand.pt")
    pose_model = YOLO("yolo11n-pose.pt")
    main()
    cap.stop()
    cv2.destroyAllWindows()

