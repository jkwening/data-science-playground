import os

import pandas as pd

from kaggle_projects.kaggle_utils import KAGGLE_PROJECT_ROOT_DIR

PROJECT_NAME = "home_credit_default_risk"
PROJECT_DIR = os.path.join(KAGGLE_PROJECT_ROOT_DIR, PROJECT_NAME)
RAW_DATASET_DIR = os.path.join(PROJECT_DIR, "data")


def get_app_bureau_dataset(test: bool = False) -> pd.DataFrame:
    """Return merged dataset of all application related data sources."""
    # start with application dataset
    print("Loading application dataset - test: ", test)
    application_csv = "application_test.csv" if test else "application_train.csv"
    df = pd.read_csv(os.path.join(RAW_DATASET_DIR, application_csv))

    # merge Credit Bureau datasets
    print("Merging Credit Bureau data...")
    bureau_df = pd.read_csv(os.path.join(RAW_DATASET_DIR, "bureau.csv"))
    df = pd.merge(left=df, right=bureau_df, on="SK_ID_CURR", how="left")
    bureau_bal_df = pd.read_csv(os.path.join(RAW_DATASET_DIR, "bureau_balance.csv"))
    df = pd.merge(left=df, right=bureau_bal_df, on="SK_ID_BUREAU", how="left")
    return df


def get_prev_app_dataset() -> pd.DataFrame:
    """Return previous application dataset including additional data."""
    # merge previous application datasets
    print("Merging previous application data...")
    df = pd.read_csv(os.path.join(RAW_DATASET_DIR, "previous_application.csv"))
    install_payments_df = pd.read_csv(
        os.path.join(RAW_DATASET_DIR, "installments_payments.csv")
    )
    df = pd.merge(left=df, right=install_payments_df, on="SK_ID_PREV", how="left")
    cc_bal_df = pd.read_csv(os.path.join(RAW_DATASET_DIR, "credit_card_balance.csv"))
    df = pd.merge(left=df, right=cc_bal_df, on="SK_ID_PREV", how="left")
    pos_cash_bal_df = pd.read_csv(os.path.join(RAW_DATASET_DIR, "POS_CASH_balance.csv"))
    df = pd.merge(left=df, right=pos_cash_bal_df, on="SK_ID_PREV", how="left")
    return df


# def get_full_dataset(test: bool = False) -> pd.DataFrame:
#     """Return merged dataset of all application related data sources."""
#     # start with application dataset
#     print("Loading application dataset - test: ", test)
#     application_csv = "application_test.csv" if test else "application_train.csv"
#     df = pd.read_csv(os.path.join(RAW_DATASET_DIR, application_csv))

#     # merge Credit Bureau datasets
#     print("Merging Credit Bureau data...")
#     bureau_df = pd.read_csv(os.path.join(RAW_DATASET_DIR, "bureau.csv"))
#     df = pd.merge(left=df, right=bureau_df, on="SK_ID_CURR", how="left")
#     bureau_bal_df = pd.read_csv(os.path.join(RAW_DATASET_DIR, "bureau_balance.csv"))
#     df = pd.merge(left=df, right=bureau_bal_df, on="SK_ID_BUREAU", how="left")

#     # merge previous application datasets
#     print("Merging previous application data...")
#     prev_app_df = pd.read_csv(os.path.join(RAW_DATASET_DIR, "previous_application.csv"))
#     df = pd.merge(left=df, right=prev_app_df, on="SK_ID_CURR", how="left")
#     install_payments_df = pd.read_csv(
#         os.path.join(RAW_DATASET_DIR, "installments_payments.csv")
#     )
#     df = pd.merge(left=df, right=install_payments_df, on="SK_ID_PREV", how="left")
#     cc_bal_df = pd.read_csv(os.path.join(RAW_DATASET_DIR, "credit_card_balance.csv"))
#     df = pd.merge(left=df, right=cc_bal_df, on="SK_ID_PREV", how="left")
#     pos_cash_bal_df = pd.read_csv(os.path.join(RAW_DATASET_DIR, "POS_CASH_balance.csv"))
#     df = pd.merge(left=df, right=pos_cash_bal_df, on="SK_ID_PREV", how="left")

#     return df
