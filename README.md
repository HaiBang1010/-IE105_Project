# Android APK Malware Detection System

Hệ thống phân tích và phát hiện phần mềm độc hại Android dựa trên Machine Learning, sử dụng phân tích tĩnh các file APK.

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Git
- Windows/Linux/macOS

## Cài đặt

1. Clone repository:
```bash
git clone <your-repo-url>
cd -IE105_Project
```

2. Tạo và kích hoạt môi trường ảo:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
-IE105_Project/
├── data/                  # Thư mục chứa dữ liệu (APK files)
├── models/               # Thư mục chứa model đã train
│   └── server_model.pt   # Model chính để phân tích
├── analyze_apk.py        # Script phân tích APK
└── requirements.txt      # File cài đặt dependencies
```

## Cách sử dụng

1. Đặt file APK cần phân tích vào thư mục `data/`

2. Chạy phân tích APK:
```bash
python analyze_apk.py path/to/your/app.apk
```

Ví dụ:
```bash
python analyze_apk.py data/example.apk
```

### Kết quả phân tích

Hệ thống sẽ hiển thị báo cáo chi tiết bao gồm:

1. Thông tin cơ bản:
   - Tên file
   - Package name
   - Tên ứng dụng
   - Kích thước file

2. Phân tích bảo mật:
   - Kết luận (Malware/Benign)
   - Độ tin cậy của model
   - Điểm rủi ro
   - Các chỉ báo rủi ro phát hiện được

3. Phân tích quyền:
   - Tổng số quyền
   - Các quyền nguy hiểm
   - Danh sách quyền được sử dụng

4. Phân tích thành phần:
   - Số lượng Activities
   - Số lượng Services
   - Số lượng Broadcast Receivers

## Các mã lỗi

- `[ERROR] Usage: python analyze_apk.py <apk_path>`: Thiếu đường dẫn file APK
- `[ERROR] APK file not found`: File APK không tồn tại
- `[ERROR] Failed to extract APK information`: Không thể trích xuất thông tin từ APK
- `[ERROR] Server model not found`: Không tìm thấy model phân tích

## Lưu ý

- Đảm bảo file APK hợp lệ và có thể đọc được
- Model sử dụng API Level 10 làm mặc định nếu không xác định được API Level của APK
- Các file APK và thư mục data/ không được commit lên Git

