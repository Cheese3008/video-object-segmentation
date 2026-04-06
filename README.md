# UAV Real-time Object Tracking System

Hệ thống theo dõi đối tượng thời gian thực cho UAV dựa trên **ROS2**, tập trung vào bài toán **phát hiện đối tượng, bám mục tiêu và ước lượng vị trí mục tiêu từ dữ liệu camera**.

---

## 1. Giới thiệu

Dự án này được xây dựng nhằm phục vụ các bài toán perception cho UAV, trong đó hệ thống sử dụng camera để phát hiện đối tượng, theo dõi mục tiêu theo thời gian thực và ước lượng vị trí mục tiêu để cung cấp cho các module điều khiển phía sau.

Pipeline chính của hệ thống như sau:

**Camera → YOLOv8 Detection → ByteTrack Tracking → Target Pose Estimation → ROS2 Topics → Các module UAV phía sau**

Hệ thống được thiết kế theo hướng dễ tích hợp với các thành phần khác như bộ lọc hợp nhất dữ liệu, bộ điều khiển gimbal, PX4 ROS2 Interface hoặc các node điều khiển bay.

---

## 2. Chức năng chính

- Phát hiện đối tượng thời gian thực bằng **YOLOv8**
- Theo dõi mục tiêu qua nhiều khung hình bằng **ByteTrack**
- Hỗ trợ chọn mục tiêu bằng **click chuột** hoặc **ROI**
- Ước lượng vị trí mục tiêu từ ảnh camera
- Hỗ trợ ước lượng khoảng cách dựa trên kích thước **bounding box**
- Publish dữ liệu mục tiêu qua **ROS2 topics**
- Publish ảnh debug để quan sát kết quả tracking
- Hỗ trợ tích hợp với các module UAV khác trong hệ thống
- Hỗ trợ hiển thị và giám sát qua giao diện web

---

## 3. Cấu trúc thư mục

```bash
UAV-Real-time-Object-Tracking-System/
├── README.md
└── follow/
    ├── object_lock_tracker_node.py
    ├── models/
    ├── src/
    │   ├── ai_follow/
    │   ├── gimbal_controller/
    │   ├── px4-ros2-interface-lib/
    │   ├── px4_msgs/
    │   ├── rtsp_camera/
    │   ├── target_pose_fusion/
    │   ├── utils/
    │   └── vision_opencv/
    ├── camear_calibration/
    ├── deploy/
    ├── services/
    ├── web/
    │   ├── backend/
    │   └── frontend/
    ├── web_env/
    ├── Makefile
    ├── install_opencv.sh
    ├── install_opencv_4_10_pi5.sh
    └── setup_web.sh
```

---

## 4. Các module chính

### 4.1. Object Lock Tracker Node

File chính của khối xử lý ảnh là:

```bash
object_lock_tracker_node.py
```

Đây là node trung tâm của hệ thống perception, đảm nhiệm việc nhận ảnh từ camera, phát hiện đối tượng, theo dõi mục tiêu và publish kết quả cho các node khác.

#### Chức năng

- Subscribe ảnh camera và thông tin camera
- Chạy YOLOv8 để phát hiện đối tượng
- Sử dụng ByteTrack để theo dõi mục tiêu
- Hỗ trợ chọn mục tiêu bằng click chuột hoặc vùng chọn ROI
- Publish pose mục tiêu
- Publish ảnh debug
- Gửi tín hiệu reset và trạng thái tracking cho các node khác

#### Topic đầu vào

- `/camera/image_raw`
- `/camera/camera_info`
- `/click_point`
- `/select_bbox`

#### Topic đầu ra

- `/detecd_pose`
- `/image_proc`
- `/reset`
- `/tag_state`

#### Một số tham số cấu hình chính

- `model_path`
- `conf_thres`
- `z_mode`
- `target_real_height_m`
- `target_class_id`
- `enable_gui`
- `publish_debug_image`

---

### 4.2. Target Pose Fusion

Package `target_pose_fusion` dùng để hợp nhất và làm mượt dữ liệu vị trí mục tiêu nhằm giảm nhiễu từ kết quả thị giác.

Thành phần này đặc biệt hữu ích khi dữ liệu từ camera cần được cung cấp cho các bộ điều khiển phía sau một cách ổn định hơn, chẳng hạn như gimbal controller hoặc bộ điều khiển bay.

---

### 4.3. AI Follow Package

Package `ai_follow` chứa cấu trúc ROS2 package cho hệ thống perception bám mục tiêu, bao gồm:

- cấu hình package
- metadata
- phần tích hợp trong workspace ROS2
- các thành phần hỗ trợ perception và tracking

---

### 4.4. Web Monitoring

Thư mục `web/` bao gồm:

- `backend/` với các file như `app.py`, `ros_bridge.py`
- `frontend/` với file giao diện `index.html`

Thành phần này phục vụ cho việc theo dõi hình ảnh đã xử lý hoặc hiển thị luồng camera thông qua giao diện web.

---

### 4.5. Deploy và tiện ích

Dự án cũng bao gồm:

- các script triển khai trong `deploy/`
- các file cấu hình service trong `services/`
- các script cài đặt OpenCV cho Linux và Raspberry Pi 5
- `Makefile` để build workspace ROS2
- các tiện ích phục vụ thiết lập môi trường web

---

## 5. Công nghệ sử dụng

Các công nghệ chính được sử dụng trong dự án:

- **Python**
- **ROS2**
- **YOLOv8**
- **ByteTrack**
- **OpenCV**
- **Ultralytics**
- **Supervision**
- **C++**
- **PX4 ROS2 Interface**
- **Kalman Filter**

---

## 6. Nguyên lý hoạt động

Hệ thống hoạt động theo các bước chính sau:

1. Camera publish dữ liệu ảnh lên ROS2.
2. Node `object_lock_tracker_node.py` nhận luồng ảnh từ camera.
3. YOLOv8 thực hiện phát hiện đối tượng trong ảnh.
4. ByteTrack duy trì ID và theo dõi mục tiêu qua nhiều frame.
5. Hệ thống xác định mục tiêu cần bám và tiến hành ước lượng vị trí mục tiêu.
6. Node publish các dữ liệu cần thiết, bao gồm:
   - pose mục tiêu
   - ảnh debug
   - trạng thái tracking
   - tín hiệu reset
7. Các module phía sau như fusion, gimbal hoặc flight controller sẽ subscribe các topic này để xử lý tiếp.

---

## 7. Giao tiếp ROS2

### 7.1. Topic subscribe

- `/camera/image_raw`
- `/camera/camera_info`
- `/click_point`
- `/select_bbox`

### 7.2. Topic publish

- `/detecd_pose`
- `/image_proc`
- `/reset`
- `/tag_state`

---

## 8. Mục đích tích hợp trong hệ thống UAV

Hệ thống tracking này có thể được sử dụng như một khối perception đầu vào cho nhiều bài toán UAV khác nhau, ví dụ:

- UAV bám theo người hoặc phương tiện
- UAV giám sát mục tiêu theo thời gian thực
- UAV phục vụ tìm kiếm cứu nạn
- UAV hỗ trợ điều khiển gimbal bám mục tiêu
- UAV cung cấp dữ liệu mục tiêu cho bộ điều khiển hạ cánh chính xác hoặc điều khiển tự động

---

## 9. Hướng phát triển

Một số hướng phát triển trong tương lai:

- Tăng độ ổn định khi track lại mục tiêu sau khi bị mất
- Cải thiện độ chính xác của ước lượng vị trí mục tiêu

---



