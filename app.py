import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Load pickle files
with open('dbscan_model.pkl', 'rb') as file:
    dbscan_model = pickle.load(file)

with open('scaler_model.pkl', 'rb') as file:
    scaler_model = pickle.load(file)

with open('pca_model.pkl', 'rb') as file:
    pca = pickle.load(file)

with open('df_csv_file.pkl', 'rb') as file:
    df = pickle.load(file)

# Ensure that df has 'CUST_ID' column
has_cust_id = 'CUST_ID' in df.columns

# Define min and max values based on df
min_values = df.min()
max_values = df.max()

# Streamlit UI
st.title("💳 Credit Cart Customer Clustering")

st.image("https://static.vecteezy.com/system/resources/thumbnails/009/384/393/small_2x/credit-card-clipart-design-illustration-free-png.png", caption='Cluster Analysis',width=600)

st.sidebar.header('User Input Features')

def get_user_input():
    data = {
        'BALANCE': st.sidebar.slider('BALANCE',
                                     min_value=float(min_values['BALANCE']),
                                     max_value=float(max_values['BALANCE']),
                                     value=float((min_values['BALANCE'] + max_values['BALANCE']) / 2)),  # Default value is set to midpoint
        'BALANCE_FREQUENCY': st.sidebar.slider('BALANCE_FREQUENCY',
                                               min_value=float(min_values['BALANCE_FREQUENCY']),
                                               max_value=float(max_values['BALANCE_FREQUENCY']),
                                               value=float((min_values['BALANCE_FREQUENCY'] + max_values['BALANCE_FREQUENCY']) / 2)),
        'PURCHASES': st.sidebar.slider('PURCHASES',
                                       min_value=float(min_values['PURCHASES']),
                                       max_value=float(max_values['PURCHASES']),
                                       value=float((min_values['PURCHASES'] + max_values['PURCHASES']) / 2)),
        'ONEOFF_PURCHASES': st.sidebar.slider('ONEOFF_PURCHASES',
                                              min_value=float(min_values['ONEOFF_PURCHASES']),
                                              max_value=float(max_values['ONEOFF_PURCHASES']),
                                              value=float((min_values['ONEOFF_PURCHASES'] + max_values['ONEOFF_PURCHASES']) / 2)),
        'INSTALLMENTS_PURCHASES': st.sidebar.slider('INSTALLMENTS_PURCHASES',
                                                    min_value=float(min_values['INSTALLMENTS_PURCHASES']),
                                                    max_value=float(max_values['INSTALLMENTS_PURCHASES']),
                                                    value=float((min_values['INSTALLMENTS_PURCHASES'] + max_values['INSTALLMENTS_PURCHASES']) / 2)),
        'CASH_ADVANCE': st.sidebar.slider('CASH_ADVANCE',
                                          min_value=float(min_values['CASH_ADVANCE']),
                                          max_value=float(max_values['CASH_ADVANCE']),
                                          value=float((min_values['CASH_ADVANCE'] + max_values['CASH_ADVANCE']) / 2)),
        'PURCHASES_FREQUENCY': st.sidebar.slider('PURCHASES_FREQUENCY',
                                                 min_value=float(min_values['PURCHASES_FREQUENCY']),
                                                 max_value=float(max_values['PURCHASES_FREQUENCY']),
                                                 value=float((min_values['PURCHASES_FREQUENCY'] + max_values['PURCHASES_FREQUENCY']) / 2)),
        'ONEOFF_PURCHASES_FREQUENCY': st.sidebar.slider('ONEOFF_PURCHASES_FREQUENCY',
                                                        min_value=float(min_values['ONEOFF_PURCHASES_FREQUENCY']),
                                                        max_value=float(max_values['ONEOFF_PURCHASES_FREQUENCY']),
                                                        value=float((min_values['ONEOFF_PURCHASES_FREQUENCY'] + max_values['ONEOFF_PURCHASES_FREQUENCY']) / 2)),
        'PURCHASES_INSTALLMENTS_FREQUENCY': st.sidebar.slider('PURCHASES_INSTALLMENTS_FREQUENCY',
                                                              min_value=float(min_values['PURCHASES_INSTALLMENTS_FREQUENCY']),
                                                              max_value=float(max_values['PURCHASES_INSTALLMENTS_FREQUENCY']),
                                                              value=float((min_values['PURCHASES_INSTALLMENTS_FREQUENCY'] + max_values['PURCHASES_INSTALLMENTS_FREQUENCY']) / 2)),
        'CASH_ADVANCE_FREQUENCY': st.sidebar.slider('CASH_ADVANCE_FREQUENCY',
                                                    min_value=float(min_values['CASH_ADVANCE_FREQUENCY']),
                                                    max_value=float(max_values['CASH_ADVANCE_FREQUENCY']),
                                                    value=float((min_values['CASH_ADVANCE_FREQUENCY'] + max_values['CASH_ADVANCE_FREQUENCY']) / 2)),
        'CASH_ADVANCE_TRX': st.sidebar.slider('CASH_ADVANCE_TRX',
                                              min_value=int(min_values['CASH_ADVANCE_TRX']),
                                              max_value=int(max_values['CASH_ADVANCE_TRX']),
                                              value=int((min_values['CASH_ADVANCE_TRX'] + max_values['CASH_ADVANCE_TRX']) / 2)),
        'PURCHASES_TRX': st.sidebar.slider('PURCHASES_TRX',
                                           min_value=int(min_values['PURCHASES_TRX']),
                                           max_value=int(max_values['PURCHASES_TRX']),
                                           value=int((min_values['PURCHASES_TRX'] + max_values['PURCHASES_TRX']) / 2)),
        'CREDIT_LIMIT': st.sidebar.slider('CREDIT_LIMIT',
                                          min_value=float(min_values['CREDIT_LIMIT']),
                                          max_value=float(max_values['CREDIT_LIMIT']),
                                          value=float((min_values['CREDIT_LIMIT'] + max_values['CREDIT_LIMIT']) / 2)),
        'PAYMENTS': st.sidebar.slider('PAYMENTS',
                                      min_value=float(min_values['PAYMENTS']),
                                      max_value=float(max_values['PAYMENTS']),
                                      value=float((min_values['PAYMENTS'] + max_values['PAYMENTS']) / 2)),
        'MINIMUM_PAYMENTS': st.sidebar.slider('MINIMUM_PAYMENTS',
                                              min_value=float(min_values['MINIMUM_PAYMENTS']),
                                              max_value=float(max_values['MINIMUM_PAYMENTS']),
                                              value=float((min_values['MINIMUM_PAYMENTS'] + max_values['MINIMUM_PAYMENTS']) / 2)),
        'PRC_FULL_PAYMENT': st.sidebar.slider('PRC_FULL_PAYMENT',
                                              min_value=float(min_values['PRC_FULL_PAYMENT']),
                                              max_value=float(max_values['PRC_FULL_PAYMENT']),
                                              value=float((min_values['PRC_FULL_PAYMENT'] + max_values['PRC_FULL_PAYMENT']) / 2)),
        'TENURE': st.sidebar.slider('TENURE',
                                    min_value=int(min_values['TENURE']),
                                    max_value=int(max_values['TENURE']),
                                    value=int((min_values['TENURE'] + max_values['TENURE']) / 2))
    }
    return pd.DataFrame(data, index=[0])

user_input = get_user_input()
if st.button("Predict Cluster"):

    # Ensure that columns are in the same order as in the training DataFrame
    # input_columns = df.columns
    # user_input = user_input[input_columns]


    # del user_input['cluster']
    # Apply the scaler
    input_scaled = scaler_model.fit_transform(user_input)

    # Apply PCA transformation
    input_pca = pca.transform(input_scaled)


    # Ensure that columns are in the same order as in the training DataFrame
    input_scaled = scaler_model.transform(user_input)

    # Apply PCA transformation
    input_pca = pca.transform(input_scaled)

    # Predict using DBSCAN
    cluster = dbscan_model.fit_predict(input_pca)

    st.write(f"Cluster Prediction: {cluster[0]}")
    if cluster==-1:
        st.success('Customers with average balance and low credit_limit that frequently purchase')
    else:
        st.success('Cutomers with low balance and low credit limit that they are not frequently purchase in type of oneoff or installment.')
