Sử dụng hàm transform_multi_frames_train(data,opt) trong spatial_transform.py để transform dữ liệu train. 
Input: tensor data => shape (numframes,C,H,W), opt load từ file config.py
Output: tensor data => shape (numframes,C,opt.sample_size,opt.sample_size)

Tương tự sử dụng hàm transform_multi_frames_val(data,opt) trong spatial_transform.py để transform dữ liệu train. 

