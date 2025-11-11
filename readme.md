# 🧠 Data Mining Workflow with scikit-learn
Dự án này thiết lập môi trường và quy trình cơ bản cho khai phá dữ liệu (Data Mining) bằng thư viện **scikit-learn**, phục vụ cho học tập và thực hành các thuật toán học máy.  
## Mục tiêu
**Mục tiêu bài toán:** Dự đoán xem khách hàng có happy (hài lòng) dựa trên trải nghiệm mua hàng của họ.
## ⚙️ Cài đặt môi trường
Tạo và kích hoạt môi trường ảo:
```bash
python -m venv sklearn-env # macos/linux
source sklearn-env/bin/activate  # activate
pip install -U scikit-learn
# Cài phần mềm trong môi trường sk-learn
# Các thư viện cần thiết sau khi kích hoạt vào sklearn-env

pip3 install -U scikit-learn pandas numpy matplotlib seaborn jupyter get_ipython
```
## Cấu trúc dự án
```bash
data-mining-workflow/
├── data/        # Dữ liệu
├── SourceImage # Ảnh sau khi train model
├── src/  # Nơi chứa mã nguồn Python
├──── cleaning.py         # Script làm sạch dữ liệu
├──── model_experiment.py # Model
├── models/      # Mô hình
├── output/      # Đầu ra sau khi làm sạch và chọn các cột
└── README.md # Mô tả dự án

```
## Convert notebook sang Python
<!-- Sources -->
```bash
cd src/
jupyter nbconvert --to python model_experiment.ipynb
python3 model_experiment.py
```
## Kiểm tra cài đặt
```bash
python -c "import sklearn; sklearn.show_versions()"
```

```bash
❯ python -c "import sklearn; sklearn.show_versions()"

System:
    python: 3.13.7 (main, Aug 14 2025, 11:12:11) [GCC 11.4.0]
executable: /home/thohnb/projects/251_data_mining_assigment/sklearn-env/bin/python3
   machine: Linux-6.17.4-orbstack-00308-g195e9689a04f-aarch64-with-glibc2.39

Python dependencies:
      sklearn: 1.7.2
          pip: 25.2
   setuptools: None
        numpy: 2.3.4
        scipy: 1.16.3
       Cython: None
       pandas: 2.3.3
   matplotlib: 3.10.7
       joblib: 1.5.2
threadpoolctl: 3.6.0

Built with OpenMP: True

threadpoolctl info:
       user_api: blas
   internal_api: openblas
    num_threads: 8
         prefix: libscipy_openblas
       filepath: /home/thohnb/projects/251_data_mining_assigment/sklearn-env/lib/python3.13/site-packages/numpy.libs/libscipy_openblas64_-71e1b124.so
        version: 0.3.30
threading_layer: pthreads
   architecture: neoversen1

       user_api: blas
   internal_api: openblas
    num_threads: 8
         prefix: libscipy_openblas
       filepath: /home/thohnb/projects/251_data_mining_assigment/sklearn-env/lib/python3.13/site-packages/scipy.libs/libscipy_openblas-d651f195.so
        version: 0.3.29.dev
threading_layer: pthreads
   architecture: neoversen1

       user_api: openmp
   internal_api: openmp
    num_threads: 8
         prefix: libgomp
       filepath: /home/thohnb/projects/251_data_mining_assigment/sklearn-env/lib/python3.13/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0
        version: None
```
