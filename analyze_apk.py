import androguard.core.bytecodes.apk as apk
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
from datetime import datetime
import io

class MalwareModel(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(MalwareModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)    
        self.layer2 = nn.Linear(64, 32)            
        self.layer3 = nn.Linear(32, num_classes)   
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(64)      
        self.batch_norm2 = nn.BatchNorm1d(32)      

    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.softmax(x)
        return x

def sanitize_text(text):
    """Sanitize text to handle Unicode characters safely"""
    if not isinstance(text, str):
        text = str(text) if text is not None else "Unknown"
    # Replace non-ASCII characters with '?'
    return ''.join(c if ord(c) < 128 else '?' for c in text)

def extract_apk_info(apk_path):
    """Extract detailed information from APK file"""
    try:
        # Redirect stderr to capture Androguard warnings
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        
        a = apk.APK(apk_path)
        
        # Get any Androguard warnings
        androguard_warnings = sys.stderr.getvalue()
        # Restore stderr
        sys.stderr = old_stderr
        
        # Store warning for later use
        result = {
            'permissions': list(a.get_permissions()),
            'activities': list(a.get_activities()),
            'services': list(a.get_services()),
            'receivers': list(a.get_receivers()),
            'package': sanitize_text(a.get_package()),
            'app_name': sanitize_text(a.get_app_name()),
            'filename': os.path.basename(apk_path),
            'file_size': os.path.getsize(apk_path),
            'warnings': androguard_warnings
        }
        return result
    except Exception as e:
        print(f"Error extracting APK info: {str(e)}")
        return None

def format_file_size(size_in_bytes):
    """Convert file size to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} TB"

def get_risk_indicators(apk_info, matched_permissions):
    """Analyze risk indicators in the APK"""
    risks = []
    risk_score = 0

    # Dangerous permissions
    dangerous_permissions = {
        'android.permission.ACCESS_FINE_LOCATION': 'Precise location access',
        'android.permission.READ_PHONE_STATE': 'Phone information access',
        'android.permission.SYSTEM_ALERT_WINDOW': 'Overlay window (possible ad abuse)',
        'android.permission.GET_TASKS': 'Running apps monitoring',
        'android.permission.READ_SMS': 'SMS access',
        'android.permission.SEND_SMS': 'SMS sending capability',
        'android.permission.RECEIVE_BOOT_COMPLETED': 'Auto-start capability',
        'android.permission.READ_CONTACTS': 'Contacts access',
    }

    for perm in matched_permissions:
        if perm in dangerous_permissions:
            risks.append(f"Dangerous permission: {dangerous_permissions[perm]}")
            risk_score += 1

    # Component analysis
    if len(apk_info['services']) > 5:
        risks.append(f"High number of services: {len(apk_info['services'])}")
        risk_score += 1
    
    if len(apk_info['receivers']) > 5:
        risks.append(f"High number of broadcast receivers: {len(apk_info['receivers'])}")
        risk_score += 1

    # Package name analysis
    package = apk_info['package']
    if len(package.split('.')) < 3 or any(c.isdigit() for c in package):
        risks.append("Suspicious package name pattern")
        risk_score += 1

    return risks, risk_score

def print_analysis_report(apk_info, prediction, confidence, risks, risk_score, matched_permissions):
    """Print the analysis report in a well-formatted structure"""
    # Print header
    print("\n" + "="*50)
    print("            APK ANALYSIS REPORT")
    print("="*10)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*10)

    # Print any warnings from Androguard
    if 'warnings' in apk_info and "API Level could not be found" in apk_info['warnings']:
        print("\n[INFO] APK analysis note: Using default Android API Level 10 for compatibility")

    # Basic Information
    print("\n1. Basic Information:")
    print(f"   - File Name: {apk_info['filename']}")
    print(f"   - Package Name: {apk_info['package']}")
    print(f"   - App Name: {apk_info['app_name']}")
    print(f"   - File Size: {format_file_size(apk_info['file_size'])}")

    # Security Analysis
    print("\n2. Security Analysis:")
    high_risk = risk_score >= 3
    low_confidence = confidence < 0.7
    is_malware = prediction == 1 or (low_confidence and high_risk)

    if is_malware:
        print("   - Verdict: [MALWARE]")
        if prediction == 0:
            print("   - Note: Classified as benign but high risk indicators detected")
    else:
        print("   - Verdict: [BENIGN]")
        if high_risk:
            print("   - Note: Contains suspicious patterns but classified as benign")

    print(f"   - Model Confidence: {confidence*100:.2f}%")
    print(f"   - Risk Score: {risk_score} {'(HIGH)' if high_risk else '(LOW)'}")

    if risks:
        print("\n   Risk Indicators Found:")
        for risk in risks:
            print(f"   - [!] {risk}")

    # Permission Analysis
    print("\n3. Permission Analysis:")
    print(f"   - Total Permissions: {len(apk_info['permissions'])}")
    print(f"   - Matched Permissions: {len(matched_permissions)} (used in analysis)")

    if matched_permissions:
        print("\n   Significant Permissions Found:")
        for perm in matched_permissions[:10]:
            print(f"   - {perm}")
        if len(matched_permissions) > 10:
            print(f"   ... and {len(matched_permissions)-10} more")

    # Component Analysis
    print("\n4. Component Analysis:")
    print(f"   - Activities: {len(apk_info['activities'])}")
    print(f"   - Services: {len(apk_info['services'])}")
    print(f"   - Broadcast Receivers: {len(apk_info['receivers'])}")

    print("\n" + "="*10)
    print("            END OF ANALYSIS")
    print("="*10 + "\n")

def analyze_apk(apk_path):
    """Analyze APK file using server model and provide detailed analysis"""
    try:
        # Load server model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = 'models/server_model.pt'
        
        if not os.path.exists(model_path):
            return {
                'status': 'error',
                'message': 'Server model not found. Please combine models first.'
            }
        
        # Extract APK information first
        apk_info = extract_apk_info(apk_path)
        if not apk_info:
            return {
                'status': 'error',
                'message': 'Failed to extract APK information'
            }

        # Load and prepare model
        model_data = torch.load(model_path, map_location=device)
        input_size = model_data['input_size']
        model = MalwareModel(input_size=input_size)
        model.load_state_dict(model_data['model_state_dict'])
        model.to(device)
        model.eval()

        # Prepare feature vector
        saved_permissions = model_data['permissions']
        feature_vector = np.zeros(len(saved_permissions))
        matched_permissions = []
        
        for i, perm in enumerate(saved_permissions):
            if perm in apk_info['permissions']:
                feature_vector[i] = 1
                matched_permissions.append(perm)

        # Make prediction
        feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(feature_tensor)
            probabilities = outputs.cpu().numpy()[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])

        # Get risk analysis
        risks, risk_score = get_risk_indicators(apk_info, matched_permissions)
        
        # Print formatted report
        print_analysis_report(apk_info, prediction, confidence, risks, risk_score, matched_permissions)

        # Return structured result
        return {
            'status': 'success',
            'result': {
                'is_malware': prediction == 1 or (confidence < 0.7 and risk_score >= 3),
                'model_prediction': prediction == 1,
                'confidence': confidence,
                'risk_score': risk_score,
                'risks': risks,
                'details': {
                    'basic_info': {
                        'filename': apk_info['filename'],
                        'package': apk_info['package'],
                        'app_name': apk_info['app_name'],
                        'file_size': apk_info['file_size']
                    },
                    'permissions': {
                        'total': len(apk_info['permissions']),
                        'matched': len(matched_permissions),
                        'list': matched_permissions
                    },
                    'components': {
                        'activities': len(apk_info['activities']),
                        'services': len(apk_info['services']),
                        'receivers': len(apk_info['receivers'])
                    }
                }
            }
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == "__main__":
    # Redirect stderr to capture Androguard warnings
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    try:
        if len(sys.argv) != 2:
            print("\n[ERROR] Usage: python analyze_apk.py <apk_path>")
            sys.exit(1)
            
        apk_path = sys.argv[1]
        
        if not os.path.exists(apk_path):
            print(f"\n[ERROR] APK file not found at {apk_path}")
            sys.exit(1)
            
        result = analyze_apk(apk_path)
        if result['status'] == 'error':
            print(f"\n[ERROR] {result['message']}")
            sys.exit(1)
        sys.exit(0)
    finally:
        # Restore stderr
        sys.stderr = old_stderr 