import json
import os

def save_and_update_history_json(new_history, filepath="training_history.json"):
    """
    Lưu hoặc cập nhật thêm dữ liệu huấn luyện vào file JSON.
    
    Args:
        new_history (dict): Dictionary chứa dữ liệu các epoch mới train.
        filepath (str): Đường dẫn file JSON để lưu.
    """
    # Nếu file đã tồn tại, đọc dữ liệu cũ ra để nối tiếp
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            existing_history = json.load(f)
        
        # Cập nhật nối tiếp danh sách các chỉ số
        for key in new_history:
            if key in existing_history:
                existing_history[key].extend(new_history[key])
            else:
                existing_history[key] = new_history[key]
        
        data_to_save = existing_history
        print(f"--> Đã CẬP NHẬT nối tiếp dữ liệu vào file: '{filepath}'")
    else:
        data_to_save = new_history
        print(f"--> Đã TẠO MỚI và lưu dữ liệu vào file: '{filepath}'")

    # Ghi dữ liệu ra file JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)
        
    return data_to_save