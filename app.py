from flask import Flask, render_template, jsonify, request
import subprocess
import os
import signal
import time
import threading
import json
from werkzeug.utils import secure_filename
import queue
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to track processes
server_process = None
client_processes = {}
server_metrics = {}
client_metrics = {}

# Queues to store logs
server_log_queue = queue.Queue()
client_log_queues = {}

# Global variables to track client status
client_status = {
    'benign': False,
    'malware': False
}

# Global variable to track training progress
training_progress = {
    'progress': 0,
    'completed': False
}

def read_output(process, queue):
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            queue.put(output.strip())
    return

def start_server():
    global server_process
    if server_process is None:
        try:
            # Khởi động server trong một process riêng
            server_process = subprocess.Popen(
                ['python', 'server.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start thread to read server output
            log_thread = threading.Thread(
                target=read_output, 
                args=(server_process, server_log_queue), 
                daemon=True
            )
            log_thread.start()
            
            # Đợi một chút để đảm bảo server đã khởi động
            time.sleep(2)
            
            # Kiểm tra xem server có start thành công không
            if server_process.poll() is not None:
                # Server đã dừng - có lỗi xảy ra
                error_logs = get_logs_from_queue(server_log_queue)
                error_message = "Server failed to start. Error logs:\n" + "\n".join(error_logs)
                server_process = None
                return False, error_message
                
            return True, "Server started successfully"
        except Exception as e:
            server_process = None
            return False, f"Error starting server: {str(e)}"
    return False, "Server is already running"

def stop_server():
    global server_process
    if server_process is not None:
        # Send SIGTERM to the server process
        server_process.terminate()
        try:
            # Wait for the process to terminate
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # If process doesn't terminate, force kill it
            server_process.kill()
        server_process = None
        return True
    return False

def start_client(client_id):
    if client_id not in client_processes or client_processes[client_id] is None:
        client_processes[client_id] = subprocess.Popen(['python', 'client.py', str(client_id)],
                                                     stdout=subprocess.PIPE,
                                                     stderr=subprocess.STDOUT,
                                                     universal_newlines=True,
                                                     bufsize=1)
        # Create queue for this client if it doesn't exist
        if client_id not in client_log_queues:
            client_log_queues[client_id] = queue.Queue()
        # Start thread to read client output
        threading.Thread(target=read_output, args=(client_processes[client_id], client_log_queues[client_id]), daemon=True).start()
        return True
    return False

def get_server_status():
    if server_process is None:
        return False
    return server_process.poll() is None

def get_client_status(client_id):
    if client_id not in client_processes or client_processes[client_id] is None:
        return False
    return client_processes[client_id].poll() is None

def get_logs_from_queue(queue):
    logs = []
    while not queue.empty():
        try:
            logs.append(queue.get_nowait())
        except queue.Empty:
            break
    return logs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/server')
def server_page():
    return render_template('server.html')

@app.route('/client')
def client_page():
    client_type = request.args.get('type', 'benign')
    return render_template('client.html', client_type=client_type)

@app.route('/start_server')
def start_server_route():
    success, message = start_server()
    if success:
        return jsonify({
            'status': 'success',
            'message': message
        })
    return jsonify({
        'status': 'error',
        'message': message
    })

@app.route('/stop_server')
def stop_server_route():
    if stop_server():
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Server is not running'})

@app.route('/start_client/<int:client_id>')
def start_client_route(client_id):
    if start_client(client_id):
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': f'Client {client_id} already running'})

@app.route('/get_status')
def get_status():
    status = {
        'server': {
            'running': get_server_status()
        },
        'clients': {
            f'client_{i}': {
                'connected': get_client_status(i)
            } for i in range(1, 3)
        }
    }
    return jsonify(status)

@app.route('/get_logs')
def get_logs():
    client_id = request.args.get('client_id')
    if client_id:
        # Get client logs
        if int(client_id) in client_log_queues:
            logs = get_logs_from_queue(client_log_queues[int(client_id)])
            return jsonify({'status': 'success', 'logs': logs})
        return jsonify({'status': 'error', 'message': 'Client not found'})
    else:
        # Get server logs
        logs = get_logs_from_queue(server_log_queue)
        return jsonify({'status': 'success', 'logs': logs})

@app.route('/analyze_apk', methods=['POST'])
def analyze_apk():
    try:
        if 'apk_file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            })
        
        file = request.files['apk_file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            })
        
        if not file.filename.endswith('.apk'):
            return jsonify({
                'status': 'error',
                'message': 'File must be an APK'
            })
        
        # Kiểm tra kích thước file
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({
                'status': 'error',
                'message': f'File size exceeds limit of {app.config["MAX_CONTENT_LENGTH"] / (1024*1024)}MB'
            })
        
        # Lưu file tạm thời
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Kiểm tra model tổng hợp
            if not os.path.exists('models/server_model.pt'):
                return jsonify({
                    'status': 'error',
                    'message': 'Server model not found. Please combine models first.'
                })

            # Phân tích APK
            result = subprocess.run(
                ['python', 'analyze_apk.py', filepath],
                capture_output=True,
                text=True,
                check=True
            )
            
            try:
                # Parse kết quả JSON từ analyze_apk.py
                analysis_result = json.loads(result.stdout)
                
                if analysis_result['status'] == 'error':
                    return jsonify({
                        'status': 'error',
                        'message': analysis_result['message']
                    })
                
                return jsonify(analysis_result)  # Trả về kết quả trực tiếp từ analyze_apk.py
                
            except json.JSONDecodeError:
                # Nếu không parse được JSON, có thể là lỗi trong quá trình phân tích
                error_message = result.stderr if result.stderr else result.stdout
                return jsonify({
                    'status': 'error',
                    'message': f'Error analyzing APK: {error_message}'
                })
                
        except subprocess.CalledProcessError as e:
            return jsonify({
                'status': 'error',
                'message': f'Error analyzing APK: {e.stderr}'
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        })
        
    finally:
        # Dọn dẹp file tạm
        try:
            if 'filepath' in locals():
                os.remove(filepath)
        except Exception as e:
            print(f"Error cleaning up file: {str(e)}")

@app.route('/get_models', methods=['POST'])
def get_models():
    try:
        # Run combine_models.py
        subprocess.run(['python', 'combine_models.py'], check=True)
        return jsonify({'success': True})
    except subprocess.CalledProcessError as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/start_client', methods=['POST'])
def start_client_post():
    data = request.get_json()
    client_type = data.get('type')
    
    if client_type not in ['benign', 'malware']:
        return jsonify({'success': False, 'error': 'Invalid client type'})
    
    try:
        # Start client in a separate thread
        client_id = '1' if client_type == 'benign' else '2'
        thread = threading.Thread(target=run_client, args=(client_type, client_id))
        thread.daemon = True
        thread.start()
        
        client_status[client_type] = True
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def run_client(client_type, client_id):
    try:
        if client_type == 'benign':
            subprocess.run(['python', 'client_benign.py', client_id], check=True)
        else:
            subprocess.run(['python', 'client_malware.py', client_id], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {client_type} client: {e}")
    finally:
        client_status[client_type] = False
        training_progress['completed'] = True

@app.route('/client_status')
def get_client_status_route():
    return jsonify(client_status)

@app.route('/training_status')
def get_training_status_route():
    return jsonify(training_progress)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 