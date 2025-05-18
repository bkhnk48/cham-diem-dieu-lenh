import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import threading

def input_with_timeout(prompt, timeout=10, default='165'):
    result = [default]

    def ask():
        val = input(prompt)
        result[0] = val if val.strip() != '' else default

    thread = threading.Thread(target=ask)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print(f"\n⏰ Không nhập trong {timeout} giây. Dùng mặc định: {default}")
    return result[0]

def process_images_and_export_pose_info(image_dir, output_dir, excel_path, real_height_cm=165):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    mp_drawing = mp.solutions.drawing_utils

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_records = []

    bone_segments = [
        ("LEFT_SHOULDER", "LEFT_ELBOW"),#2
        ("LEFT_ELBOW", "LEFT_WRIST"),#4
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),#6
        ("RIGHT_ELBOW", "RIGHT_WRIST"),#8
        ("LEFT_HIP", "LEFT_KNEE"),#10
        ("LEFT_KNEE", "LEFT_ANKLE"),#12
        ("RIGHT_HIP", "RIGHT_KNEE"),#14
        ("RIGHT_KNEE", "RIGHT_ANKLE"),#16
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),#18
        ("LEFT_HIP", "RIGHT_HIP"),#20
        ("LEFT_SHOULDER", "LEFT_HIP"),#22
        ("RIGHT_SHOULDER", "RIGHT_HIP"),#24
    ]

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        row = {"filename": filename}

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for i, lm in enumerate(landmarks):
                row[f'P{i}_x'] = round(lm.x, 5)
                row[f'P{i}_y'] = round(1 - lm.y, 5)  # lật trục y

            # --- Tính chiều cao người trên ảnh ---
            top_y = min(landmarks[mp_pose.PoseLandmark.NOSE].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_EYE].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y)
            bottom_y = max(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,
                           landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y)
            person_height_px = bottom_y - top_y

            # --- Chiều dài thực đoạn vai phải đến cùi chỏ ---
            segment_real_cm = 0.184 * real_height_cm
            segment_pixel = (segment_real_cm / real_height_cm) * person_height_px

            # Toạ độ (x, y) đã đảo trục y
            p_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            p_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            x1, y1 = p_shoulder.x, 1 - p_shoulder.y
            x2, y2 = p_elbow.x, 1 - p_elbow.y

            # Chiều dài hình chiếu lên mặt phẳng Oxy
            proj_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Tính góc nghiêng so với mặt phẳng Oxy
            if segment_pixel > 0:
                cos_theta = np.clip(proj_length / segment_pixel, 0, 1)
                angle_rad = np.arccos(cos_theta)
                angle_deg = round(np.degrees(angle_rad), 2)
            else:
                angle_deg = 'NA'

            row["RIGHT_SHOULDER-RIGHT_ELBOW_angle_deg"] = angle_deg

        else:
            for i in range(33):
                row[f'P{i}_x'] = 'NA'
                row[f'P{i}_y'] = 'NA'
            row["RIGHT_SHOULDER-RIGHT_ELBOW_angle_deg"] = 'NA'

        for a, b in bone_segments:
            try:
                idx_a = mp_pose.PoseLandmark[a].value
                idx_b = mp_pose.PoseLandmark[b].value
                if results.pose_landmarks:
                    xa, ya = landmarks[idx_a].x, 1 - landmarks[idx_a].y
                    xb, yb = landmarks[idx_b].x, 1 - landmarks[idx_b].y
                    row[f'{a}-{b}_x'] = round((xa + xb) / 2, 5)
                    row[f'{a}-{b}_y'] = round((ya + yb) / 2, 5)
                else:
                    row[f'{a}-{b}_x'] = 'NA'
                    row[f'{a}-{b}_y'] = 'NA'
            except:
                row[f'{a}-{b}_x'] = 'NA'
                row[f'{a}-{b}_y'] = 'NA'

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Ghi chữ góc lên ảnh (nếu có giá trị)
            angle_text = row.get("RIGHT_SHOULDER-RIGHT_ELBOW_angle_deg", 'NA')
            if angle_text != 'NA':
                text = f"Angle RShoulder-RElbow: {angle_text} deg"
                cv2.putText(
                    image, text,
                    org=(30, 50),  # vị trí hiển thị
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255),  # màu đỏ (BGR)
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)

        data_records.append(row)

    df = pd.DataFrame(data_records)
    df.sort_values(by="filename", inplace=True)
    df.to_excel(excel_path, index=False)


def estimate_camera_parameters_auto(real_height_cm, image_path, focal_length_mm=4.15, sensor_height_mm=2.76):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return "Không nhận diện được tư thế người."

    landmarks = results.pose_landmarks.landmark

    head_y = landmarks[mp_pose.PoseLandmark.NOSE].y * img_height
    foot_y = max(
        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
    ) * img_height

    pixel_height = abs(foot_y - head_y)
    f_pixel = (focal_length_mm / sensor_height_mm) * img_height
    Z = (f_pixel * real_height_cm) / pixel_height

    y_center = img_height / 2
    pixel_offset = ((head_y + foot_y) / 2) - y_center
    vertical_fov_deg = 2 * np.arctan(sensor_height_mm / (2 * focal_length_mm)) * (180 / np.pi)
    degrees_per_pixel = vertical_fov_deg / img_height
    tilt_angle_deg = degrees_per_pixel * pixel_offset

    estimated_camera_height_cm = (y_center - head_y) / pixel_height * real_height_cm + (real_height_cm / 2)

    return {
        "estimated_distance_cm": round(Z, 2),
        "estimated_camera_height_cm": round(estimated_camera_height_cm, 2),
        "estimated_tilt_angle_deg": round(tilt_angle_deg, 2)
    }


# ---- Cấu hình chạy ----
if __name__ == "__main__":
    path = "normal.jpg"
    height_input = input_with_timeout("Nhập chiều cao người (cm) [mặc định 165]: ", timeout=10, default='165')
    try:
        height = float(height_input)
    except:
        height = 165

    result = estimate_camera_parameters_auto(height, path)
    print("Thông số camera:", result)

    image_dir = "images"
    output_dir = "poses_of_images"
    excel_path = "pose_data.xlsx"

    process_images_and_export_pose_info(image_dir, output_dir, excel_path, real_height_cm=height)
    print("✅ Xử lý hoàn tất. Dữ liệu lưu ở:", excel_path)
