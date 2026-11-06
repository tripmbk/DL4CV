So sánh kiến trúc CNN và Transformer.
Sử dụng dataset CIFAR100
1.Datasets:
Tập CIFAR100 có tổng cộng 60.000 ảnh trong đó Training set 50.000 ảnh, Test set có 10.000 ảnh.
Số lớp: 100 
Chiến lượt chia dữ liệu:
Đối với tập Training set ban đầu: sẽ chia 90%(tương ứng 45.000 ảnh) cho tập Training set,
và 10%(5.000 ảnh) cho tập Validation.
Test set là tập set ban đầu.
Chiến lược: để giữ tỉ lệ đóng góp đồng đều, đặt seed cố định là 42
2. Kiến trúc:
Mô hình lựa chọn: ResNet-18 và Vision Transformer.
3. Chiến lược training:
Chia làm 3 thí nghiệm
A: From scatch
B: Pretrain
C: Fine-tuning
4. Thí nghiệm:
Lựa chọn các siêu tham số:
Optimier, learning rate, batch size, epoch, seed, nền tảng
5. Các metric:
Accuracy, F1, Training time,
