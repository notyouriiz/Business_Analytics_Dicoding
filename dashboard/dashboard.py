import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load datasets
orders = pd.read_csv('../E-Commerce Public Dataset/orders_dataset.csv')
products = pd.read_csv('../E-Commerce Public Dataset/products_dataset.csv')
sellers = pd.read_csv('../E-Commerce Public Dataset/sellers_dataset.csv')
category_translation = pd.read_csv('../E-Commerce Public Dataset/product_category_name_translation.csv')
order_items = pd.read_csv('../E-Commerce Public Dataset/order_items_dataset.csv')
customer_reviews = pd.read_csv('../E-Commerce Public Dataset/order_reviews_dataset.csv') 

# Sidebar filter
st.sidebar.header('Filter Data')
categories = products['product_category_name'].unique()  # Use English category names
selected_category = st.sidebar.selectbox('Select Product Category', categories)

# Filter data based on selected category
filtered_products = products[products['product_category_name'] == selected_category]
filtered_orders = orders[orders['order_id'].isin(order_items[order_items['product_id'].isin(filtered_products['product_id'])]['order_id'])]

# Metrics
st.title('E-Commerce Dashboard')

total_sales = filtered_orders['order_id'].count()
total_products = filtered_products['product_id'].nunique()

col1, col2 = st.columns(2)

with col1:
    st.metric('Total Sales', total_sales)

with col2:
    st.metric('Total Products in Selected Category', total_products)

# Average rating for selected category
average_rating = customer_reviews[customer_reviews['order_id'].isin(filtered_orders['order_id'])]['review_score'].mean()

st.header('Average Rating for Selected Category')
st.write(f'The average rating for {selected_category} is {average_rating:.2f}')

# Calculate total sales per product category for visualization
sales_data = order_items.merge(products[['product_id', 'product_category_name']], on='product_id')
sales_data = sales_data.groupby('product_category_name')['order_id'].count().reset_index(name='total_sales')

# Visualization for Top Products
top_products = sales_data.nlargest(10, 'total_sales')

def plot_top_products(top_products):
    max_value = top_products['total_sales'].max()
    colors = ['blue' if x != max_value else 'green' for x in top_products['total_sales']]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_products, x='product_category_name', y='total_sales', palette=colors)
    plt.title('Top 10 Product Categories by Sales')
    plt.xticks(rotation=45)
    plt.xlabel('Product Category')
    plt.ylabel('Total Sales')
    plt.tight_layout()
    plt.savefig('top_products.png')
    plt.close()

plot_top_products(top_products)
st.header('Top 10 Product Categories by Sales')
image = Image.open('top_products.png')
st.image(image, caption='Top 10 Product Categories by Sales', use_column_width=True)

# Identify undersold products
undersold_products = sales_data[sales_data['total_sales'] < 10]  # 10 for threshold

def plot_undersold_products(undersold_products):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=undersold_products, x='product_category_name', y='total_sales', palette='mako')
    plt.title('Undersold Products (Total Sales < 10)')
    plt.xticks(rotation=45)
    plt.xlabel('Product Category')
    plt.ylabel('Total Sales')
    plt.tight_layout()
    plt.savefig('undersold_products.png')
    plt.close()

plot_undersold_products(undersold_products)
st.header('Undersold Products')
image_undersold = Image.open('undersold_products.png')
st.image(image_undersold, caption='Undersold Products (Total Sales < 10)', use_column_width=True)

# Merge customer_reviews with order_items to get 'product_id'
merged_reviews = pd.merge(customer_reviews, order_items, on='order_id')

# Merge with products to get 'product_category_name' in reviews data
merged_reviews = pd.merge(merged_reviews, products[['product_id', 'product_category_name']], on='product_id', how='left')

# Group by 'product_category_name' and calculate the mean review score
review_data = merged_reviews.groupby('product_category_name')['review_score'].mean().reset_index()

# Merge review data with undersold products
undersold_with_reviews = pd.merge(undersold_products, review_data, on='product_category_name', how='left')

def plot_undersold_with_reviews(undersold_with_reviews):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=undersold_with_reviews, x='product_category_name', y='review_score', palette='mako')
    plt.title('Undersold Products with Average Review Scores')
    plt.xticks(rotation=45)
    plt.xlabel('Product Category')
    plt.ylabel('Average Review Score')
    plt.tight_layout()
    plt.savefig('undersold_with_reviews.png')
    plt.close()

plot_undersold_with_reviews(undersold_with_reviews)
st.header('Undersold Products with Average Review Scores')
image_reviews = Image.open('undersold_with_reviews.png')
st.image(image_reviews, caption='Undersold Products with Average Review Scores', use_column_width=True)

# RFM Analysis
# Merge orders and order_items to get the details of each order
merged_data = orders.merge(order_items, on='order_id')

# Convert the order creation date to datetime format
merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])

# Calculate Recency
snapshot_date = merged_data['order_purchase_timestamp'].max() + pd.DateOffset(days=1)  # Date after last purchase
merged_data['Recency'] = (snapshot_date - merged_data['order_purchase_timestamp']).dt.days

# Calculate Frequency and Monetary
rfm_df = merged_data.groupby('customer_id').agg({
    'Recency': 'min',  # Last purchase recency
    'order_id': 'nunique',  # Unique order count
    'price': 'sum'  # Total expenditure
}).rename(columns={'order_id': 'Frequency', 'price': 'Monetary'})

# Segment the RFM data
def rfm_segment(row):
    if row['Recency'] < 100 and row['Frequency'] > 5 and row['Monetary'] > 200:
        return 'High-Value Customers'
    elif row['Recency'] < 100 and row['Frequency'] <= 5:
        return 'Potential Loyal Customers'
    elif row['Recency'] > 100 and row['Frequency'] == 1:
        return 'New Customers'
    else:
        return 'At-Risk Customers'

# Apply segmentation
rfm_df['Segment'] = rfm_df.apply(rfm_segment, axis=1)

# Check if RFM DataFrame is empty
if rfm_df.empty:
    st.warning("The RFM DataFrame is empty. Please check the data processing steps.")
else:
    # Apply segmentation
    rfm_df['Segment'] = rfm_df.apply(rfm_segment, axis=1)

    # Count the occurrences of each segment
    segment_counts = rfm_df['Segment'].value_counts()

    # Display segment counts for debugging
    st.subheader('Segment Counts')
    st.write(segment_counts)



# Conclusion and Insights
st.header('Conclusion and Insights')

st.markdown("""<ol>
    <li style= "color:orange">What product is needed to be prioritized for advertisement and improvements?</li>
    <li style= "color:orange">What is the most undersold product, and how to boost the sales?</li>
    <li style= "color:orange">If we want to launch a new product, what product criteria it needs to have so it can be a best seller in the market or at least it can support the undersold products to have better sales?</li>
</ol>

<b>INSIGHT:</b>
<ol>
    <li><b style= "color:lightblue">Prioritize improvements and advertising for household products.</b> Besides having a general market where all segments are potential customers, household products also represent a bestselling product category.</li>
    <li><b style= "color:lightblue">Focus on market research and advertise according to the market.</b> It can be used as feedback to enhance product features according to customer preferences.</li>
    <li><b style= "color:lightblue">Analyze trends and sales to identify potential opportunities.</b> Focusing on features that have an existing demand can result in potential sales growth.</li>
</ol>
""", unsafe_allow_html=True)

# Footer
st.sidebar.text('© 2024 E-Commerce Analysis Team')
