import os
import pandas as pd
import json
from androguard.core.bytecodes.apk import APK

def extract_permissions(apk_path):
    """Trích xuất quyền từ file APK"""
    try:
        apk = APK(apk_path)
        permissions = apk.get_permissions()
        return list(permissions)
    except Exception as e:
        print(f"Error processing {apk_path}: {str(e)}")
        return []

def process_benign_folder(folder_path, year):
    """Xử lý tất cả APK benign trong thư mục và trích xuất quyền"""
    data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".apk"):
                apk_path = os.path.join(root, file)
                try:
                    permissions = extract_permissions(apk_path)
                    if permissions:
                        data.append({
                            "filename": file,
                            "permissions": permissions,
                            "year": year,
                            "label": "benign"
                        })
                        print(f"Processed benign: {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    return data

def create_benign_dataset():
    """Tạo dataset từ các quyền của APK benign"""
    # Tạo thư mục datacsv nếu chưa tồn tại
    if not os.path.exists('data/datacsv'):
        os.makedirs('data/datacsv')
    
    # Định nghĩa các năm của APK benign
    benign_years = {
        'Benign_2015': '2015',
        'Benign_2017': '2017'
    }
    
    # Tạo dictionary để lưu trữ tất cả các quyền
    all_permissions = set()
    all_data = []
    
    # Xử lý từng thư mục benign
    for folder, year in benign_years.items():
        folder_path = os.path.join('data', folder)
        if os.path.exists(folder_path):
            print(f"\nProcessing {year} benign APKs from {folder_path}...")
            folder_data = process_benign_folder(folder_path, year)
            if folder_data:
                all_data.extend(folder_data)
                # Cập nhật tập quyền
                for item in folder_data:
                    all_permissions.update(item['permissions'])
                print(f"Processed {len(folder_data)} {year} benign APKs")
            else:
                print(f"No APKs found in {folder_path}")
    
    # Tạo DataFrame và lưu vào CSV
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = 'data/datacsv/benign_permissions.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(all_data)} benign APKs to {output_file}")
        
        # Lưu danh sách tất cả các quyền
        with open('data/datacsv/benign_permissions.json', 'w') as f:
            json.dump(list(all_permissions), f)
        
        print(f"Total unique permissions found in benign apps: {len(all_permissions)}")
    else:
        print("No benign APKs were processed!")

def main():
    print("Starting benign APK processing and dataset creation...")
    create_benign_dataset()
    print("Benign processing completed!")

if __name__ == "__main__":
    main() 