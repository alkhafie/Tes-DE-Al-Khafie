import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv('C:\Users\Al Khafie\Tes Data Engineer\ecommerce-session-bigquery.csv')

# Cetak info data untuk memahami struktur dan tipe data
print(df.info())

# Konversi kolom 'date' menjadi datetime jika belum
df['date'] = pd.to_datetime(df['date'])

# Identifikasi Produk Teratas Berdasarkan Pendapatan Transaksi Harian
grouped_product_data = df.groupby(['productSKU', 'date'])['transactionRevenue'].sum().reset_index()
sorted_product_data = grouped_product_data.sort_values(by=['date', 'transactionRevenue'], ascending=[True, False])
top_products_per_day = sorted_product_data.groupby('date').head(3)

# Mendeteksi Anomali dalam Jumlah Transaksi Produk menggunakan Isolation Forest pada 'itemQuantity'
anomaly_detector = IsolationForest(contamination=0.10)
df['is_anomaly'] = anomaly_detector.fit_predict(df[['itemQuantity']])

# Identifikasi Kota atau Provinsi Paling Menguntungkan
grouped_location_data = df.groupby(['city', 'country'])['transactionRevenue'].sum().reset_index()
most_profitable_location = grouped_location_data.sort_values(by='transactionRevenue', ascending=False).iloc[0]

# Tampilkan Hasil
print("Produk Teratas Berdasarkan Pendapatan Transaksi Harian:")
print(top_products_per_day[['productSKU', 'date', 'transactionRevenue']])
print("\nProduk dengan Anomali dalam Jumlah Transaksi:")
print(df[df['is_anomaly'] == -1][['productSKU', 'itemQuantity']])
print("\nKota atau Provinsi Paling Menguntungkan:")
print(most_profitable_location)

