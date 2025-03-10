# Brazilian E-Commerce RFM and Geospatial Analysis

Proyek ini merupakan analisis data e-commerce di Brazil yang mencakup analisis RFM (Recency, Frequency, Monetary) dan geospatial. Dengan memanfaatkan dataset Olist Brazilian E-Commerce, proyek ini mengidentifikasi pola pembelian pelanggan dan distribusi geografis transaksi untuk memberikan insight bisnis yang bernilai.

## Requirements
- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- streamlit
- plotly
- geopandas
- babel
- unidecode
- matplotlib-ticker

## Setup Environment - Anaconda
```
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```
## Setup Environment - Shell/Terminal
```
mkdir analisis_data
cd analisis_data
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Run Streamlit App
```
streamlit run dashboard/dashboard.py
```


