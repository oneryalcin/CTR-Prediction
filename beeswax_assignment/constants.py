"""Constants for Data/ML pipeline"""

# Source data path
SRC_DATA_PATH = 'data/ctr_data.pickle'

# Columns to drop from the original dataset
cols_to_drop = ['id', 'site_id', 'site_domain', 'app_id', 'app_domain', 'device_id', 'device_ip', 'device_model', 'device_type','C14', 'C17', 'C19', 'C21']

# Columns to be used in one hot encoding
one_hot_encoded_cols = ['C1', 'banner_pos', 'site_category', 'app_category', 'device_conn_type', 'C15', 'C16', 'C18', 'C20', 'day', 'hour']

# Fbeta value, higher value (>1) means high recall, lower means (<1) high precision 
f_BETA = 2