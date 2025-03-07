import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.ticker as ticker 
from plotly.subplots import make_subplots
import geopandas as gpd
from matplotlib import colormaps as cmaps
import unidecode

st.set_page_config(page_title="Brazilian E-Commerce Dashboard", layout="wide")

# Import Dataset
df_highest = pd.read_csv('./data/processed/df_highest.csv')
geojson_file = './data/brazil-states.geojson'
gdf = gpd.read_file(geojson_file)

@st.cache_data
def load_data():
    # Loading all necessary datasets
    customers_df = pd.read_csv('data/customers_dataset.csv')
    orders_df = pd.read_csv('data/orders_dataset.csv')
    order_items_df = pd.read_csv('data/order_items_dataset.csv')
    products_df = pd.read_csv('data/products_dataset.csv')
    product_category_df = pd.read_csv('data/product_category_name_translation.csv')
    geolocation_df = pd.read_csv('data/geolocation_dataset.csv')
    
    # Clean and transform data as in the notebook
    # Fixing column names
    products_df.rename(columns={
        'product_name_lenght': 'product_name_length',
        'product_description_lenght': 'product_description_length'
    }, inplace=True)
    
    # Convert datetime columns
    datetime_columns = ['order_purchase_timestamp', 'order_approved_at',
                     'order_delivered_carrier_date', 'order_delivered_customer_date', 
                     'order_estimated_delivery_date']
    for column in datetime_columns:
        orders_df[column] = pd.to_datetime(orders_df[column])
    
    # Clean geolocation data - remove diacritics
    def remove_diacritic(column):
        column_space = ' '.join(column.split())
        return unidecode.unidecode(column_space.lower())
    
    geolocation_df['geolocation_city'] = geolocation_df['geolocation_city'].apply(remove_diacritic)
    
    # Process geolocation data
    max_state = geolocation_df.groupby(['geolocation_zip_code_prefix','geolocation_state']).size().reset_index(name='count').drop_duplicates(subset='geolocation_zip_code_prefix').drop('count',axis=1)
    geolocation_grouping = geolocation_df.groupby(['geolocation_zip_code_prefix','geolocation_city','geolocation_state'])[['geolocation_lat','geolocation_lng']].median().reset_index()
    geolocation_grouping = geolocation_grouping.merge(max_state, on=['geolocation_zip_code_prefix','geolocation_state'], how='inner')
    
    # Merge datasets
    # 1. Merge geolocation with customers
    customers_geolocation = customers_df.merge(geolocation_grouping, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='inner')
    
    # 2. Merge products with categories
    products_with_category_df = pd.merge(
        left=products_df,
        right=product_category_df,
        how='outer',
        left_on='product_category_name',
        right_on='product_category_name'
    )
    
    # 3. Merge order items with products
    order_items_with_category_df = pd.merge(
        left=order_items_df,
        right=products_with_category_df,
        how='outer',
        left_on='product_id',
        right_on='product_id'
    )
    
    # 4. Merge orders with customers
    orders_customers_df = pd.merge(
        left=orders_df,
        right=customers_geolocation,
        how='outer',
        left_on='customer_id',
        right_on='customer_id'
    )
    
    # 5. Final merge - all data
    all_data = pd.merge(
        left=order_items_with_category_df,
        right=orders_customers_df,
        how='outer',
        left_on='order_id',
        right_on='order_id'
    )
    
    return all_data

# Load data
try:
    all_data = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title and description
st.title("📊 Brazilian E-commerce Analysis Dashboard")
st.write("This dashboard provides insights from Brazilian e-commerce data analysis.")

# Sidebar Settings
with st.sidebar:
    st.title("Olist E-Commerce Dashboard")
    st.image("https://as2.ftcdn.net/v2/jpg/00/90/67/29/1000_F_90672947_9o36fMzvYpFoS2cvgxACFUR0wleV5Yq5.jpg")
    st.subheader("Filter Data by Date 📅")

    # Date Selection
    min_date = all_data["order_purchase_timestamp"].min().date()
    max_date = all_data["order_purchase_timestamp"].max().date()

    # Set Default Start Date to prevent large data load
    default_start_date = max(min_date, pd.to_datetime("2018-01-01").date())

    st.write("Select the date range below:")

    start_date = st.date_input(
        "Start Date", value=default_start_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", value=max_date,
                             min_value=min_date, max_value=max_date)
    st.write(f"Data available from {min_date} to {max_date}")

# Filter Data by Date
filtered_data = all_data[
    (all_data["order_purchase_timestamp"].dt.date >= start_date) &
    (all_data["order_purchase_timestamp"].dt.date <= end_date)
]

# Check if filtered data exists
if filtered_data.empty:
    st.warning("No data available for the selected date range. Please select different dates.")
    st.stop()

# Main dashboard sections
tab1, tab2, tab3, tab4 = st.tabs(["Product Categories", "Geographic Distribution", "RFM Analysis", "Order Analysis"])

# TAB 1: Product Categories Analysis
with tab1:
    st.header("Product Category Analysis")
    
    # Best and worst performing categories
    st.subheader("🏆 Best & Worst Performing Product Categories")
    
    order_category_df = all_data.groupby('product_category_name_english')['order_id'].nunique().reset_index(name='order_count')

    # Sort untuk kategori tertinggi dan terendah
    best_categories = order_category_df.sort_values(by='order_count', ascending=False).head(5)
    worst_categories = order_category_df.sort_values(by='order_count', ascending=True).head(5)
    
    # Urutkan best_categories dari terbanyak ke terendah
    best_categories = best_categories.sort_values('order_count', ascending=True)

    # Urutkan worst_categories dari terendah ke tertinggi
    worst_categories = worst_categories.sort_values('order_count', ascending=False)

    # Define categories to highlight
    highlight_categories = ['bed_bath_table', 'security_and_services']

    # Create color lists dynamically for both charts
    best_colors = ['#72BCD4' if cat.lower().replace(' ', '_') in highlight_categories 
                else '#D3D3D3' for cat in best_categories['product_category_name_english']]

    worst_colors = ['#72BCD4' if cat.lower().replace(' ', '_') in highlight_categories 
                    else '#D3D3D3' for cat in worst_categories['product_category_name_english']]

    # Create subplots with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Highest Demand Categories", "Lowest Demand Categories"),
                        specs=[[{"type": "bar"}, {"type": "bar"}]])

    # Add best categories bar chart with dynamic colors
    fig.add_trace(
        go.Bar(
            x=best_categories['order_count'],
            y=best_categories['product_category_name_english'],
            marker_color=best_colors,
            orientation='h',
            name='Best Categories'
        ),
        row=1, col=1
    )

    # Add worst categories bar chart with dynamic colors
    fig.add_trace(
        go.Bar(
            x=worst_categories['order_count'],
            y=worst_categories['product_category_name_english'],
            marker_color=worst_colors,
            orientation='h',
            name='Worst Categories'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Highest and Lowest Performing Categories by Order Count",
        height=400,
        width=1300,
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False,
        template="plotly_white",
        font=dict(family="Arial", size=12)
    )

    # Update axes
    fig.update_xaxes(title_text="Order Count", row=1, col=1)
    fig.update_xaxes(title_text="Order Count", row=1, col=2)
    fig.update_yaxes(title_text=None, row=1, col=1)
    fig.update_yaxes(title_text=None, row=1, col=2)

    st.plotly_chart(fig)

    # Line Chart
    # Category Selection for detailed analysis
    st.subheader("📊 Category Sales Analysis")
    all_categories = sorted(filtered_data["product_category_name_english"].dropna().unique())
    
    if all_categories:
        selected_category = st.selectbox("Select Product Category:", all_categories)
        
        category_data = filtered_data[filtered_data["product_category_name_english"] == selected_category]
        
        if not category_data.empty:
            # Group data by month for the selected category
            category_data['month'] = category_data['order_purchase_timestamp'].dt.strftime('%Y-%m')
            monthly_sales = category_data.groupby('month')['price'].sum().reset_index()
            
            # Create sales trend chart
            fig = px.line(monthly_sales, x='month', y='price', 
                        title=f"Monthly Sales Trend for {selected_category}",
                        labels={"price": "Sales Amount", "month": "Month"})
            st.plotly_chart(fig)
        else:
            st.info(f"No data available for {selected_category} in the selected date range.")
    else:
        st.info("No category data available for the selected date range.")

# TAB 2: Geographic Distribution
with tab2:
    st.header("Customer Geographic Distribution")
    
    # Count customers by state
    customer_by_state = filtered_data.groupby('customer_state')['customer_unique_id'].nunique().reset_index(name='customer_count')
    
    # Merge GeoDataFrame dengan df_highest berdasarkan 'sigla' dan 'customer_state'
    gdf = gdf.merge(df_highest, left_on='sigla', right_on='customer_state', how='left')

    # Membuat color map untuk kategori produk
    unique_categories = df_highest['product_category_name_english'].unique()
    colors = cmaps['tab20']
    category_color_map = {category: colors(i % 20) for i, category in enumerate(unique_categories)}

    # Menetapkan warna ke state berdasarkan kategori produk
    gdf['color'] = gdf['product_category_name_english'].map(category_color_map)

    # Membuat plot
    fig, ax = plt.subplots(figsize=(21, 10))
    gdf.plot(ax=ax, color=gdf['color'], edgecolor='black')

    # Menghitung centroid dari polygon
    gdf['centroid'] = gdf.geometry.centroid

    # Menambahkan label pada centroid yang menunjukkan kategori produk
    for x, y, label, state in zip(gdf.centroid.x, gdf.centroid.y, gdf['product_category_name_english'], gdf['sigla']):
        if pd.notna(label): 
            ax.text(x, y, f"{state}\n{label}", fontsize=8, ha='center', color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

    # Add title and labels
    plt.title('Most Popular Product Categories by State', fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    st.pyplot(fig)

    # Add a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=category_color_map[category], markersize=10) 
            for category in unique_categories]
    ax.legend(handles, unique_categories, title="Product Category", loc='upper left', bbox_to_anchor=(1, 1))
    
    # visualization - bar chart of customers by state
    fig = px.bar(customer_by_state.sort_values('customer_count', ascending=False), 
               x='customer_state', y='customer_count',
               title='Number of Customers by State',
               labels={'customer_count': 'Number of Customers', 'customer_state': 'State'})
    st.plotly_chart(fig)
    
    # Most popular category by state
    st.subheader("🔝 Most Popular Category by State")
    
    # Get the most popular category in each state
    state_category = filtered_data.groupby(['customer_state', 'product_category_name_english']).size().reset_index(name='count')
    df_highest = state_category.loc[state_category.groupby('customer_state')['count'].idxmax()]
    
    fig = px.bar(df_highest.sort_values('count', ascending=False),
               x='customer_state', y='count', color='product_category_name_english',
               title='Most Popular Category in Each State',
               labels={'count': 'Number of Orders', 'customer_state': 'State'})
    st.plotly_chart(fig)

# TAB 3: RFM Analysis
with tab3:
    st.header("RFM Analysis")
    st.write("RFM analysis segments customers based on their Recency, Frequency, and Monetary value.")
    
    # Create RFM dataframe
    rfm_df = filtered_data.groupby(by="customer_unique_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",  
        "price": "sum"  
    })

    # Rename columns for clarity
    rfm_df.columns = ["customer_unique_id", "max_order_timestamp", "frequency", "monetary"]

    # Calculate recency (in days)
    recent_date = filtered_data["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].dt.date.apply(lambda x: (recent_date - x).days)
    
    # Drop unnecessary column
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    
    # Display RFM dataframe sample
    st.subheader("RFM Data Sample")
    st.dataframe(rfm_df.head())
    
    # Visualize top customers by RFM parameters
    st.subheader("Best Customers by RFM Parameters")
    
    # Create columns for the charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Recency chart
        recency_df = rfm_df.sort_values(by="recency", ascending=True).head(5).reset_index(drop=True)
        recency_df["customer"] = [f"Customer {i+1}" for i in range(len(recency_df))]
        
        fig = px.bar(recency_df, x="customer", y="recency", 
                   title="Top Customers by Recency (days)",
                   color_discrete_sequence=["#72BCD4"],
                   labels={"recency": "Days since last purchase", "customer": ""})
        st.plotly_chart(fig)
    
    with col2:
        # Frequency chart
        frequency_df = rfm_df.sort_values(by="frequency", ascending=False).head(5).reset_index(drop=True)
        frequency_df["customer"] = [f"Customer {i+1}" for i in range(len(frequency_df))]
        
        fig = px.bar(frequency_df, x="customer", y="frequency", 
                   title="Top Customers by Frequency",
                   color_discrete_sequence=["#72BCD4"],
                   labels={"frequency": "Number of orders", "customer": ""})
        st.plotly_chart(fig)
    
    with col3:
        # Monetary chart
        monetary_df = rfm_df.sort_values(by="monetary", ascending=False).head(5).reset_index(drop=True)
        monetary_df["customer"] = [f"Customer {i+1}" for i in range(len(monetary_df))]
        
        fig = px.bar(monetary_df, x="customer", y="monetary", 
                   title="Top Customers by Monetary Value",
                   color_discrete_sequence=["#72BCD4"],
                   labels={"monetary": "Total spent ($)", "customer": ""})
        st.plotly_chart(fig)

# TAB 4: Order Analysis
with tab4:
    st.header("Order Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_orders = len(filtered_data["order_id"].unique())
        st.metric("Total Orders", f"{total_orders:,}")
        
    with col2:
        total_customers = len(filtered_data["customer_unique_id"].unique())
        st.metric("Total Customers", f"{total_customers:,}")
        
    with col3:
        avg_order_value = filtered_data.groupby("order_id")["price"].sum().mean()
        st.metric("Average Order Value", f"${avg_order_value:.2f}")
        
    with col4:
        top_category = order_category_df.sort_values(by="order_count", ascending=False).iloc[0]["product_category_name_english"]
        st.metric("Most Popular Category", top_category)
    

    # Order trends over time
    st.subheader("Order Trends Over Time")
    
    filtered_data['order_month'] = filtered_data['order_purchase_timestamp'].dt.strftime('%Y-%m')
    monthly_orders = filtered_data.groupby('order_month')['order_id'].nunique().reset_index(name='order_count')
    
    fig = px.line(monthly_orders, x='order_month', y='order_count',
                title="Monthly Order Trends",
                labels={'order_month': 'Month', 'order_count': 'Number of Orders'})
    st.plotly_chart(fig)

    # Top 20 City by Payment Value
    st.subheader("Top 20 City by Total Payment Value")

    # Calculate total payment value per city
    city_payment_sum = all_data.groupby('customer_city')['price'].sum().reset_index()

    # Sort cities by payment value in descending order
    city_payment_sum_sorted = city_payment_sum.sort_values('price', ascending=False)

   
    col5, col6 = st.columns([3,2])

    with col5:
        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(8, 7))

        barplot = sns.barplot(
            x='price',  
            y='customer_city', 
            data=city_payment_sum_sorted.head(20),
            ax=ax  # Pass the axis object
        )

        # Add a title and labels
        ax.set_title('Top 20 Cities by Total Payment Value', fontsize=16)
        ax.set_xlabel('Total Payment Value', fontsize=12)
        ax.set_ylabel('City', fontsize=14)

        # Set the x-axis to display currency format
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        st.pyplot(fig)

    # Penjelasan tambahan
    with col6:
        st.write("### Explanation:")
        st.write("Sao Paulo memimpin jumlah pembayaran dengan nilai hampir dua kali lipat dari Rio de Janeiro yang berada di posisi kedua. Rio de Janeiro sendiri mencatat pembayaran dua hingga tiga kali lipat lebih besar dari Belo Horizonte di posisi ketiga. Perbedaan nilai pembayaran antara kota-kota di peringkat selanjutnya tidak terlalu signifikan dibandingkan kesenjangan antara tiga kota teratas.")
        

# Footer
st.caption('Brazilian E-Commerce Dashboard - Created by Royanrosyad')