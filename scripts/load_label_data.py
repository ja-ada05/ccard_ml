import pandas as pd

expected_columns = ['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

def load_data(path:str):
    df = pd.read_csv(path)    
    return df

def validate_and_clean_data(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Validate the input data against specific expectations and return a cleaned copy.

    Checks:
    1. Matches the expected list of columns
    2. Minimum Payments is less than Payments for each row
    3. No rows have empty fields (rows with empty fields are dropped)

    Drops and returns a copy of the data without the CUST_ID column.

    Parameters:
    df: pandas DataFrame to validate and clean
    expected_columns: list of expected column names in exact order

    Returns:
    pandas DataFrame: cleaned copy without CUST_ID column,
                      or the original copy if validation fails for column mismatch
    """
    # Work on a copy to avoid mutating the original
    df_clean = df.copy()

    # 1. Check the dataframe matches the list of columns
    if list(df_clean.columns) != expected_columns:
        print("Column mismatch detected!")
        print(f"Expected columns: {expected_columns}")
        print(f"Actual columns:   {list(df_clean.columns)}")
        return df_clean

    # 2. Check Minimum Payments is less than Payments for its corresponding row
    if 'MINIMUM_PAYMENTS' in df_clean.columns and 'PAYMENTS' in df_clean.columns:
        invalid_mask = df_clean['MINIMUM_PAYMENTS'] >= df_clean['PAYMENTS']
        if invalid_mask.any():
            print(f"Dropped {invalid_mask.sum()} rows where MINIMUM_PAYMENTS >= PAYMENTS")
            df_clean = df_clean[~invalid_mask]

    # 3. Drop rows with empty fields
    initial_count = len(df_clean)
    df_clean = df_clean.dropna()
    dropped_count = initial_count - len(df_clean)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with empty fields")

    # Drop the CUST_ID column
    if 'CUST_ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['CUST_ID'])

    return df_clean


def preprocess_data(df:pd.DataFrame):
    expected_columns = ['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

    # Validate and clean the data
    return validate_and_clean_data(df, expected_columns)
