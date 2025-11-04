# ================================================
# Mục tiêu: Khảo sát độ hài lòng của khách hàng dựa trên
# các đặc trưng của đơn hàng giao hàng trực tuyến.
# ===============================================
##################################################
# 1. Import thư viện
##################################################
import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
import numpy as np

##################################################
# 2. Tiền xử lý & Ghép dữ liệu
##################################################
# Đặt thư mục làm việc hiện tại
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Đọc dữ liệu
dirty_data = pd.read_csv("data/dirty_data.csv")
missing_data = pd.read_csv("data/missing_data.csv")

# Chuẩn hóa cột ngày tháng (date)
def parse_dates(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

dirty_data = parse_dates(dirty_data)
missing_data = parse_dates(missing_data)

# Gộp dữ liệu
merged_data = pd.concat([dirty_data, missing_data], ignore_index=True)

# NA ban đầu
print("NA ban đầu:")
print(merged_data.isna().sum())

##################################################
# 3. Làm sạch dữ liệu
##################################################
# 3.1 Xử lý order_price & order_total
merged_data["order_total"] = np.where(
    merged_data["order_total"].isna(),
    merged_data["order_price"] * (100 - merged_data["coupon_discount"]) / 100 + merged_data["delivery_charges"],
    merged_data["order_total"]
)

merged_data["order_price"] = np.where(
    merged_data["order_price"].isna(),
    (merged_data["order_total"] - merged_data["delivery_charges"]) * 100 / (100 - merged_data["coupon_discount"]),
    merged_data["order_price"]
)

# 3.2 Làm sạch season + mã hóa OneHotEncoder
merged_data["season"] = merged_data["season"].astype(str).str.lower()

month_value = merged_data["date"].dt.month
merged_data.loc[merged_data["season"].isin(["nan", "none", "na"]), "season"] = np.nan  # chuẩn hóa NA

merged_data["season"] = np.where(merged_data["season"].notna(), merged_data["season"],
                                 np.select(
                                     [
                                         month_value.isin([12, 1, 2]),
                                         month_value.isin([3, 4, 5]),
                                         month_value.isin([6, 7, 8])
                                     ],
                                     ["winter", "spring", "summer"],
                                     default="spring"
                                 ))

# OneHotEncoder cho cột season
encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(merged_data[["season"]])

# Tạo dataframe cho các cột mã hóa
season_df = pd.DataFrame(season_encoded, columns=encoder.get_feature_names_out(["season"]))
merged_data = pd.concat([merged_data.drop(columns=["season"]), season_df], axis=1)

# 3.3 Điền NA cho is_happy_customer
# Điền NA cho is_happy_customer
median_happy = int(round(merged_data["is_happy_customer"].median(skipna=True)))
merged_data["is_happy_customer"] = merged_data["is_happy_customer"].fillna(median_happy)
merged_data["is_happy_customer"] = merged_data["is_happy_customer"].astype(int)
# Kiểm tra NA sau xử lý
print("\nNA sau khi xử lý:")
print(merged_data.isna().sum())


##################################################
# 5. Xử lý ngoại lai (IQR)
##################################################
def iqr_adjust(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.clip(series, lower, upper)

merged_data["order_total"] = iqr_adjust(merged_data["order_total"])
merged_data["order_price"] = iqr_adjust(merged_data["order_price"])

# bool sang dạng số
merged_data["is_expedited_delivery"] = merged_data["is_expedited_delivery"].astype(int)

print("\nHoàn tất xử lý dữ liệu!")


# Lưu file kết quả và xuất ra các columns yêu cầu
merged_data.to_excel("data/ds_final.xlsx", index=False,columns=['delivery_charges','coupon_discount','is_expedited_delivery','distance_to_nearest_warehouse','order_price','order_total','season_spring','season_autumn','season_summer','season_winter','is_happy_customer'])