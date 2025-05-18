# 🎖️ Chương trình Chấm điểm Điều lệnh

Một công cụ tự động giúp đánh giá động tác của người quân nhân khi thực hiện điều lệnh thông qua phân tích ảnh.

---

## 📥 Đầu vào

- 🖼️ **Chuỗi frame ảnh** chụp người quân nhân đang tập điều lệnh  
- 📋 **Các tiêu chí chấm điểm** dựa trên quy chuẩn điều lệnh

---

## 📤 Đầu ra

- 🛑 **Các frame ảnh vi phạm** – tức là các khoảnh khắc mà quân nhân bị **trừ điểm** do không thực hiện đúng điều lệnh
- 
- 🛑 **Toạ độ các pose** – tức là toạ độ x, y của các điểm pose trong không gian 2D của bức ảnh
---

## 💡 Mục tiêu

Tự động hóa việc đánh giá điều lệnh, giảm gánh nặng chấm điểm thủ công, tăng tính khách quan và chính xác.

---

## 📌 Ghi chú

> Phiên bản đầu tiên này mới tập trung vào việc phát hiện lỗi trong tư thế tay và chân, sẽ tiếp tục được mở rộng cho các tiêu chí khác.

---

## 🔧 Hướng dẫn sử dụng

*(Bạn có thể bổ sung thêm phần này nếu muốn chia sẻ cách cài đặt, chạy chương trình, ví dụ như:)*

```bash
# Clone repo
git clone https://github.com/bkhnk48/cham-diem-dieu-lenh.git

# Chạy chương trình
python3 pose_batch_export.py
