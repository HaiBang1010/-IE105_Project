import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.analysis.analysis import Analysis

def extract_permissions(apk_path):
    """Trích xuất quyền từ file APK"""
    try:
        apk = APK(apk_path)
        permissions = apk.get_permissions()
        return list(permissions)
    except Exception as e:
        print(f"Error processing {apk_path}: {str(e)}")
        return []

def process_folder(folder_path, label):
    """Xử lý tất cả APK trong thư mục và trích xuất quyền"""
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
                            "label": label
                        })
                        print(f"Processed: {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    return data

def create_permission_dataset():
    """Tạo dataset từ các quyền của APK"""
    # Tạo thư mục datacsv nếu chưa tồn tại
    if not os.path.exists('data/datacsv'):
        os.makedirs('data/datacsv')
    
    # Định nghĩa mapping giữa thư mục và nhãn
    label_map = {
        'SMSmalware': 'SMSmalware',
        'Scareware': 'Scareware',
        'Ransomware': 'Ransomware',
        'Adware': 'Adware',
        'Benign_2015': 'Benign',
        'Benign_2017': 'Benign'
    }
    
    # Tạo dictionary để lưu trữ tất cả các quyền
    all_permissions = set()
    all_data = []
    
    # Xử lý từng thư mục
    for folder, label in label_map.items():
        folder_path = os.path.join('data', folder)
        if os.path.exists(folder_path):
            print(f"\nProcessing {folder} APKs from {folder_path}...")
            folder_data = process_folder(folder_path, label)
            if folder_data:
                all_data.extend(folder_data)
                # Cập nhật tập quyền
                for item in folder_data:
                    all_permissions.update(item['permissions'])
                print(f"Processed {len(folder_data)} APKs from {folder}")
            else:
                print(f"No APKs found in {folder_path}")
    
    # Tạo DataFrame và lưu vào CSV
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = 'data/datacsv/permissions.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(all_data)} APKs to {output_file}")
        
        # Lưu danh sách tất cả các quyền
        with open('data/datacsv/all_permissions.json', 'w') as f:
            json.dump(list(all_permissions), f)
        
        print(f"Total unique permissions found: {len(all_permissions)}")
    else:
        print("No APKs were processed!")

def main():
    print("Starting APK processing and dataset creation...")
    create_permission_dataset()
    print("Process completed!")

if __name__ == "__main__":
    main() 