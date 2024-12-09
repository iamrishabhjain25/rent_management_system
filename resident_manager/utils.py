import collections.abc
import datetime as dt
import functools
import os
import pathlib
import shutil
import sqlite3
import time
from typing import Any, Literal, Optional, Sequence

import configs
import numpy as np
import pandas as pd
import inspect


def log_and_backup(
    date_format="%Y-%m-%d_%H-%M-%S",
    bed_id_col: str = "BedID",
    uid_col: str = "EnrollmentID",
    room_id_col: str = "RoomNo",
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            db_handler = self if isinstance(self, DatabaseHandler) else self.db_handler
            data_manager = DataManager(confs=configs, db_handler=db_handler)

            row = kwargs.get("row", None)
            input_df = kwargs.get("input_df", None)
            data = row if (row is not None) else input_df
            room = kwargs.get("room", None)

            if func.__name__ == "copy_and_refresh_db":
                kwargs["copy_db"] = False

            comments = kwargs.get("log_comments", "")
            copy_db = kwargs.pop("copy_db", True)

            # Checking target func signature and only passing valid kwargs
            sig = inspect.signature(func)
            valid_params = sig.parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

            bed_ids, rooms, uids, errors = "", "", "", ""
            if data is not None:
                if isinstance(data, pd.Series):
                    data = data.to_frame().T
                if bed_id_col in data.columns:
                    data[room_id_col] = data[bed_id_col].str.replace(r"\D", "", regex=True)
                    bed_available = data[bed_id_col].tolist()
                    bed_ids = bed_available[0] if (len(bed_available) == 1) else np.nan

                if room_id_col in data.columns:
                    rooms_available = data[room_id_col].tolist()
                    rooms_available = rooms_available[0] if (len(rooms_available) == 1) else np.nan
                    rooms = room if room else rooms_available

                if uid_col in data.columns:
                    uids = "_".join(data[uid_col].tolist())

            # Initialise logs
            logs = {
                "Date": np.nan,
                f"{bed_id_col}": bed_ids,
                f"{room_id_col}": rooms,
                f"{uid_col}": uids,
                "Type": func.__name__,
                "DB_Before": np.nan,
                "DB_After": np.nan,
                "Error": np.nan,
                "Comments": comments,
            }

            log_time = dt.datetime.now()
            log_old_db = db_handler.get_latest_db_path(full_path=False)
            log_new_db = np.nan

            if copy_db:
                log_time, log_old_db, log_new_db = db_handler.copy_and_refresh_db(date_format=date_format)

            logs["Date"] = log_time
            logs["DB_Before"] = log_old_db
            logs["DB_After"] = log_new_db

            try:
                result = func(self, *args, **filtered_kwargs)
                data_manager.insert_log(input_df=pd.DataFrame([logs]))

                return result

            except Exception as e:
                errors += f"| Error in target function. {str(e)}"
                logs["Error"] = errors
                data_manager.insert_log(input_df=pd.DataFrame([logs]))
                raise

        return wrapper

    return decorator


class DatabaseHandler:
    """Handler for db. The purpose of this class to interact with db and handle
    all insertion, updation and deletion of data.
    """

    def __init__(self, confs):
        self.confs = confs
        self.db_path = pathlib.Path(confs.db_path)
        self.db_filename = confs.db_filename
        self.db_extension = confs.db_extension
        self.connection = self.connect()
        self._current_db_in_use = self.get_latest_db_path()

    def get_latest_db_path(self, fl_name_dt_frmt: str = "%Y-%m-%d_%H-%M-%S", full_path: bool = True) -> str:
        """Gets the latest db name. The files are always store in a certain format. This function
        use this format to always fetch the latest databse that we need to load.
        """
        rand_dt = dt.datetime.now().strftime(fl_name_dt_frmt)
        files_list = os.listdir(str(self.db_path))
        db_list = [file.strip(self.db_extension) for file in files_list if (file.endswith(self.db_extension) and file.startswith(self.db_filename))]

        if not db_list:
            raise FileNotFoundError("No database files found in the directory")

        dt_filename_dict = {file[-len(rand_dt):]: f"{file}{self.db_extension}" for file in db_list}

        if not dt_filename_dict:
            raise ValueError("No database files matching the expected pattern found")

        last_db_date = max([dt.datetime.strptime(date, fl_name_dt_frmt) for date in dt_filename_dict.keys()])
        last_db_date_str = last_db_date.strftime(fl_name_dt_frmt)
        return str(self.db_path / dt_filename_dict[last_db_date_str]) if full_path else dt_filename_dict[last_db_date_str]

    def list_all_db(self, fl_name_dt_frmt: str = "%Y-%m-%d_%H-%M-%S"):
        files_list = os.listdir(str(self.db_path))
        db_list = [file.strip(self.db_extension) for file in files_list if (file.endswith(self.db_extension) and file.startswith(self.db_filename))]

        if not db_list:
            raise FileNotFoundError("No database files found in the directory")

        return sorted(db_list)

    def connect(self):
        """connects to the required db"""
        return sqlite3.connect(self.get_latest_db_path())

    def current_db_in_use(self):
        conn = self.connection
        cursor = conn.cursor()
        cursor.execute("PRAGMA database_list;")
        db_info = cursor.fetchall()
        return db_info

    def table_exists(self, table_name):
        """Check if the table exists in the databse"""
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        result = self.connection.execute(query).fetchone()
        return result is not None

    def load_table(self, table_name, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
        """loads the required table from db"""
        df = pd.read_sql(f"SELECT * FROM {table_name}", self.connection, parse_dates=parse_dates)
        object_cols = df.select_dtypes(include="object").columns
        df[object_cols] = df[object_cols].astype("string")
        return df

    def insert_records(
        self,
        table_name: str,
        df: pd.DataFrame,
        if_exists: Literal["fail", "replace", "append"] = "replace",
        index: bool = False,
    ):
        """Insert the data into table"""

        if not table_name:
            raise ValueError(f"table name cannot be none or blank, table name passed is : {table_name}")

        table_exists = self.table_exists(table_name=table_name)
        if not table_exists and if_exists != "replace":
            raise ValueError(f"Table: {table_name} does not exist and 'if_exists' is {if_exists}")

        df.to_sql(table_name, self.connection, if_exists=if_exists, index=index)

        return

    @log_and_backup()
    def copy_and_refresh_db(self, date_format="%Y-%m-%d_%H-%M-%S"):
        time.sleep(1.2)
        copy_time = dt.datetime.now()
        new_time = copy_time.strftime(date_format)
        old_path = self._current_db_in_use
        old_filename = self.get_latest_db_path(full_path=False)
        new_filename = self.db_filename + "_" + new_time + self.db_extension
        new_path = str(self.db_path / new_filename)

        try:
            shutil.copy(old_path, new_path)
        except Exception as e:
            raise IOError(f"Failed to save new database: {e}")

        self.close()
        self._restart_connection()
        return copy_time, old_filename, new_filename

    def _restart_connection(self):
        self.connection = self.connect()
        self._current_db_in_use = self.get_latest_db_path()

    def revert_to_last_backup(self, renamed_filename="Trashed_at", fl_name_dt_frmt: str = "%Y-%m-%d_%H-%M-%S"):
        curr_datetime = dt.datetime.now().strftime(fl_name_dt_frmt)
        files_list = os.listdir(str(self.db_path))
        db_list = [file.strip(self.db_extension) for file in files_list if (file.endswith(self.db_extension) and file.startswith(self.db_filename))]

        if not db_list:
            raise FileNotFoundError("No database files found in the directory")

        if len(db_list) <= 2:
            raise ValueError("Not enough backups founds to be reverted.")

        old_db_path = self.get_latest_db_path()
        old_dir, old_filename = os.path.split(old_db_path)
        new_filename = f"{renamed_filename}_{curr_datetime}_{old_filename}"
        new_db_path = os.path.join(old_dir, new_filename)

        try:
            if self.connection:
                self.close()
            shutil.copy(old_db_path, new_db_path)
            os.remove(old_db_path)
            self._restart_connection()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to close database connection: {e}")
        except OSError as e:
            raise IOError(f"Failed to rename database file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to revert to last backup: {e}")

        return

    def close(self):
        """Close the connection"""
        return self.connection.close()


class DataManager:
    """ "
    Only task for this class is to handle data edits, new data insertion and
    data updates
    """

    def __init__(self, confs, db_handler):
        self.confs = confs
        self.db_handler = db_handler
        self.uid = confs.uid_col
        self.bed_id = confs.bedId_col
        self.room_id = confs.room_col

    def generate_new_uid(self) -> str:
        this_year = str(dt.datetime.now().year)
        residents_db = self.load_residents_info_table()
        residents_db["numeric_uid"] = residents_db[self.uid].apply(lambda x: int(x))
        latest_uid = residents_db["numeric_uid"].max()
        new_uid = str(latest_uid + 1)

        new_uid_yr = new_uid[:4]
        if new_uid_yr == this_year:
            return new_uid

        if this_year > new_uid_yr:
            if int(f"{this_year}001") not in residents_db["numeric_uid"].values:
                return f"{this_year}001"

        return new_uid

    def load_residents_info_table(
        self,
        filter_ids: Optional[str | list[str]] = None,
        filter_cols: Optional[str | list[str]] = None,
    ) -> pd.DataFrame:
        """Loads the residents information given the IDs and columns filter"""

        if isinstance(filter_ids, str):
            filter_ids = [filter_ids]

        if isinstance(filter_cols, str):
            filter_ids = [filter_cols]

        df = self.db_handler.load_table(self.confs.residents_tbl, parse_dates=self.confs.date_cols_residents_tbl)
        df_out = df
        if filter_ids:
            if not all([uid in df[self.uid].values for uid in filter_ids]):
                raise ValueError(f"uids in filter uids not found in residents database. {filter_ids}")

            df_out = df_out[df_out[self.uid].isin(filter_ids)]

        if filter_cols:
            if not all([col in df.columns for col in filter_cols]):
                raise ValueError("fiter columns passed does not exist in databse")
            df_out = df_out[filter_cols]
        return df_out

    def load_current_status(
        self,
        filter_bedId: Optional[list[str]] = None,
        filter_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Loads the current satus of the beds. Always has a fixed length of total number of beds.
        If the bed is empty, that is no one is staying in the room, then uid is nan
        """

        df = self.db_handler.load_table(self.confs.current_status_tbl, parse_dates=self.confs.date_cols_status_tbl)

        if isinstance(filter_bedId, str):
            filter_bedId = [filter_bedId]

        if isinstance(filter_cols, str):
            filter_cols = [filter_cols]

        if filter_bedId:
            df = df[df[self.bed_id].isin(filter_bedId)]

        if filter_cols:
            df = df[filter_cols]

        return df

    def load_electricity_table(self) -> pd.DataFrame:
        df = self.db_handler.load_table(self.confs.electricity_tbl, parse_dates=self.confs.date_cols_electricity_tbl)
        df.columns = df.columns.str.replace("Room_", "")
        return df.sort_values(self.confs.date_cols_electricity_tbl)

    def load_transactions_table(self) -> pd.DataFrame:
        return self.db_handler.load_table(
            self.confs.transactions_tbl,
            parse_dates=self.confs.date_cols_transactions_tbl,
        )

    def load_final_settlement_table(self) -> pd.DataFrame:
        return self.db_handler.load_table(
            self.confs.final_settlement_tbl,
            parse_dates=self.confs.date_cols_final_settlement_tbl,
        )

    def load_rent_history(self) -> pd.DataFrame:
        """Load the the transactions table"""
        return self.db_handler.load_table(
            self.confs.rent_history_tbl,
            parse_dates=self.confs.date_cols_rent_history_tbl,
        )

    def get_occupied_beds(self):
        status = self.load_current_status().reset_index()
        return status.loc[status[self.uid].notna(), self.bed_id].tolist()

    def get_empty_beds(self):
        status = self.load_current_status().reset_index()
        return status.loc[status[self.uid].isna(), self.bed_id].tolist()

    def load_logs(self):
        return self.db_handler.load_table(self.confs.logs_tbl, parse_dates=self.confs.date_cols_logs_tbl)

    def insert_log(self, input_df: pd.DataFrame):
        self.db_handler.insert_records(self.confs.logs_tbl, input_df, if_exists="append")

    @log_and_backup()
    def insert_resident_record(self, input_df: pd.DataFrame, log_comments: Optional[str] = None):
        valid_input = self.prepare_and_validate_resident_input(input_df, check_if_exists_in_old=[self.uid])
        self.db_handler.insert_records(self.confs.residents_tbl, valid_input, if_exists="append")
        return

    @log_and_backup()
    def insert_electricity_record(self, input_df: pd.DataFrame, log_comments: Optional[str] = None):
        valid_input = self.prepare_and_validate_elect_input(input_df, check_if_exists_in_old=self.confs.date_cols_electricity_tbl)
        self.db_handler.insert_records(self.confs.electricity_tbl, valid_input, if_exists="append")
        return

    def insert_transaction(self, input_df: pd.DataFrame):
        # valid_input = self.prepare_and_validate_trans_input(input_df)
        self.db_handler.insert_records(self.confs.transactions_tbl, input_df, if_exists="append")

    def insert_final_settlement_record(self, input_df: pd.DataFrame):
        self.db_handler.insert_records(table_name=self.confs.final_settlement_tbl, df=input_df, if_exists="append")

    def insert_rent_history(self, input_df: pd.DataFrame):
        self.db_handler.insert_records(table_name=self.confs.rent_history_tbl, df=input_df, if_exists="append")

    def update_current_status(self, new_status: pd.DataFrame):
        valid_status = self.prepare_and_validate_status(new_status)
        return self.db_handler.insert_records(self.confs.current_status_tbl, valid_status, if_exists="replace")

    @log_and_backup()
    def edit_electricity_record(self, input_df, log_comments: Optional[str] = None):
        all_records = self.load_electricity_table()
        new_record = self.prepare_and_validate_elect_input(input_df, check_if_exists_in_old=None)

        for col in all_records.columns:
            if col not in ["Date"]:
                all_records.loc[all_records["Date"].isin(new_record["Date"]), col] = new_record[col].iloc[0].squeeze()

        self.db_handler.insert_records(self.confs.electricity_tbl, all_records, if_exists="replace")
        return True

    @log_and_backup()
    def edit_resident_record(self, new_resident_record: pd.DataFrame, log_comments: Optional[str] = None):
        all_records = self.load_residents_info_table()
        new_record = self.prepare_and_validate_resident_input(new_resident_record, check_if_exists_in_old=None)

        for col in all_records.columns:
            if col not in [self.uid]:
                all_records.loc[all_records[self.uid].isin(new_record[self.uid]), col] = new_record[col].iloc[0]

        self.db_handler.insert_records(self.confs.residents_tbl, all_records, if_exists="replace")
        return True

    def revert_to_last_backup(self):
        return self.db_handler.revert_to_last_backup()

    def _coerce_to_list(self, input: Any | Sequence) -> list:
        """Coerce any object to list. Returns empty list if input is None"""
        if input:
            if isinstance(input, collections.abc.Sequence) and not isinstance(input, (str, bytes)):
                return list(input)

        return []

    def check_valid_bedID(self, bed_id: str | Sequence[str], valid_ids: Sequence[str]) -> bool:

        bed_ids = [bed_id] if isinstance(bed_id, str) else list(bed_id)
        if not bed_ids:
            raise ValueError("bed IDs cannot be empty or None")

        return all(a_id in valid_ids for a_id in bed_ids)

    def validate_columns_or_index(
            self,
            data: pd.DataFrame | pd.Series,
            mandatory_cols: Sequence[str],
            if_missing: Literal["raise", "coerce"] = "raise",
            fill_value: dict | Any = np.nan
    ) -> pd.DataFrame | pd.Series:

        """
        Validates whether mandatory columns or index exist in the given data.

        Parameters:
        - data: pd.DataFrame or pd.Series
            The input data to validate.
        - mandatory_cols: Sequence[str]
            List of mandatory columns to check.
        - if_missing: Literal["raise", "coerce"]
            Action to take if columns are missing: "raise" or "coerce".
        - fill_value: dict | Any, default np.nan
            Value(s) to fill missing columns when if_missing="coerce".
            Can be a scalar or a dictionary mapping column names to values.

        Returns:
        - pd.DataFrame or pd.Series
            The updated data with missing columns handled.
        """
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError("data must be an instance of pd.Series or pd.DataFrame")

        is_series = isinstance(data, pd.Series)

        if is_series:
            data = data.to_frame().T    # type: ignore

        cols_present = set(data.columns.tolist())   # type: ignore
        missing_cols = [col for col in mandatory_cols if col not in cols_present]

        if missing_cols:
            if if_missing == "raise":
                raise ValueError(f"Mandatory columns/index missing: {missing_cols}")

            # Handle missing columns
            for col in missing_cols:
                data[col] = fill_value[col] if isinstance(fill_value, dict) and col in fill_value else fill_value

        return data.squeeze() if is_series else data    # type: ignore

    def convert_data_types(
        self,
        data: pd.DataFrame | pd.Series,
        date_cols: Optional[str | list[str]] = None,
        float_cols: Optional[str | list[str]] = None,
        int_cols: Optional[str | list[str]] = None,
        str_cols: Optional[str | list[str]] = None,
    ) -> pd.DataFrame | pd.Series:

        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError("data can only be an instance of pd.Series or pd.DataFrame")

        isSeries = False
        if isinstance(data, pd.Series):
            isSeries = True
            data = data.to_frame().T

        date_cols = self._coerce_to_list(date_cols)
        float_cols = self._coerce_to_list(float_cols)
        int_cols = self._coerce_to_list(int_cols)
        str_cols = self._coerce_to_list(str_cols)

        for a_col in int_cols:
            if a_col in data.columns:
                data[a_col] = pd.to_numeric(data[a_col], errors="raise", downcast="integer")
        for a_col in date_cols:
            if a_col in data.columns:
                data[a_col] = pd.to_datetime(data[a_col])
        for a_col in float_cols:
            if a_col in data.columns:
                data[a_col] = pd.to_numeric(data[a_col], errors="raise").astype("float")
        for a_col in str_cols:
            if a_col in data.columns:
                data[a_col] = data[a_col].astype("string")

        return data.squeeze() if isSeries else data

    def string_colum_checks(
        self,
        data: pd.DataFrame,
        str_cols: Sequence[str],
        allow_duplicates: bool = False,
        allow_nan_cols: Optional[list[str]] = None,
    ) -> None:
        """
        If allow_duplicates is False it checks each column in cols_name individually to check if they have duplicates
        """

        if isinstance(str_cols, str):
            str_cols = [str_cols]

        if allow_nan_cols:
            if isinstance(allow_nan_cols, str):
                allow_nan_cols = [allow_nan_cols]

        if allow_nan_cols is None:
            allow_nan_cols = []

        missing_cols = [col for col in str_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Not all col_names ({missing_cols}) in data columns ({data.columns})")

        not_str = [col for col in str_cols if not (pd.api.types.is_string_dtype(data[col]))]
        if not_str:
            raise ValueError(f"Data type of the columns: ({not_str}) is not string")

        if any((data[str_cols] == "").any(axis=0)):
            raise ValueError(f"Empty string '' found in id_col: {str_cols} in input_df")

        check_nan_cols = set(str_cols) - set(allow_nan_cols)
        if check_nan_cols:
            if any(data[check_nan_cols].isna().any(axis=0)):
                raise ValueError(f"Nan in the input_df id cols:{check_nan_cols}")

        if not allow_duplicates:
            duplicates = [col for col in str_cols if any(data.duplicated(subset=[col]).to_numpy())]
            if duplicates:
                raise ValueError(f"Duplicated values in id_cols {duplicates} in the new data to be inserted")
        return

    def float_colum_checks(
        self,
        data: pd.DataFrame,
        float_cols: Sequence[str],
        allow_nan_cols: Optional[list[str]] = None,
        allow_zero: bool = False,
    ) -> None:

        if isinstance(float_cols, str):
            float_cols = [float_cols]

        if allow_nan_cols:
            if isinstance(allow_nan_cols, str):
                allow_nan_cols = [allow_nan_cols]

        if allow_nan_cols is None:
            allow_nan_cols = []

        missing_cols = [col for col in float_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Not all col_names ({missing_cols}) in data columns ({data.columns})")

        not_float = [col for col in float_cols if not (pd.api.types.is_float_dtype(data[col]))]
        if not_float:
            raise ValueError(f"Data type of the columns: ({not_float}) is not float")

        check_nan_cols = set(float_cols) - set(allow_nan_cols)
        if check_nan_cols:
            if any(data[check_nan_cols].isna().any(axis=0)):
                raise ValueError(f"Nan in the input_df for float cols:{check_nan_cols}")

        if not allow_zero:
            is_zero = [col for col in float_cols if any((data[col] == 0))]
            if is_zero:
                raise ValueError(f"zeros found in float columns {float_cols}")

        return

    def date_colum_checks(
        self,
        data: pd.DataFrame,
        date_cols: Sequence[str],
        allow_nan: bool = True,
        allow_duplicates: bool = False,
    ) -> None:

        if isinstance(date_cols, str):
            date_cols = [date_cols]

        missing_cols = [col for col in date_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Not all col_names ({date_cols}) in data columns ({data.columns})")

        not_date = [col for col in date_cols if not (pd.api.types.is_datetime64_any_dtype(data[col]))]
        if not_date:
            raise ValueError(f"Data type of the columns: ({not_date}) is not float")

        invalid_dt = [col for col in date_cols if (data[col] > dt.datetime.now()).any()]
        if invalid_dt:
            raise ValueError(f"Invalid Dates: Cannot take future date time. Found future date time in col ({invalid_dt})")

        if not allow_nan:
            if any(data[date_cols].isna().any(axis=0)):
                raise ValueError(f"Nan in the input_df for float cols:{date_cols}")

        if not allow_duplicates:
            duplicates = [col for col in date_cols if any(data.duplicated(subset=[col]).to_numpy())]
            if duplicates:
                raise ValueError(f"Duplicated values in date cols {duplicates} in the new data to be inserted")

        return

    def validate_df_new_with_old(
        self,
        old_df: pd.DataFrame,
        new_df: pd.DataFrame,
        check_if_exists_in_old: Optional[str | Sequence[str]] = None,
    ):
        if not isinstance(new_df, pd.DataFrame):
            raise ValueError("input_df is not an instance of pd.DataFrame")

        if old_df.empty:
            raise ValueError("Existing database table (old_df) is empty.")

        if new_df.empty:
            raise ValueError("Data to be inserted is empty.")

        missing_cols = [col for col in new_df if col not in old_df.columns]
        if missing_cols:
            raise ValueError(f"Columns from new df missing in old df: Missing ({missing_cols})")

        # aligning columns
        new_df = new_df[old_df.columns]
        if any((old_df.dtypes != new_df.dtypes).to_numpy()):
            raise ValueError("Data types of old and new data do not match for comparison")

        if check_if_exists_in_old:
            check_if_exists_in_old = [check_if_exists_in_old] if isinstance(check_if_exists_in_old, str) else list(check_if_exists_in_old)
            duplicates = [col for col in check_if_exists_in_old if any(new_df[col].isin(old_df[col]).to_numpy())]
            if duplicates:
                raise ValueError(f"Values of these cols ({duplicates}) in new data already exists in old database")

        return new_df

    def prepare_and_validate_resident_input(
        self,
        data: pd.DataFrame,
        check_if_exists_in_old: Optional[str | Sequence[str]] = None,
    ) -> pd.DataFrame:

        date_cols = self.confs.date_cols_residents_tbl
        float_cols = self.confs.float_cols_residents_tbl
        str_cols = [col for col in data.columns if col not in date_cols + float_cols]
        data = self.convert_data_types(data=data, date_cols=date_cols, float_cols=float_cols, str_cols=str_cols)    # type: ignore

        self.string_colum_checks(data=data, str_cols=[self.uid], allow_nan_cols=None, allow_duplicates=False)
        self.float_colum_checks(data=data, float_cols=float_cols, allow_nan_cols=None, allow_zero=False)
        self.check_valid_bedID(data[self.bed_id].values.tolist(), valid_ids=self.confs.valid_bedIDs)

        invalid_uid = (data[self.uid].str.isdigit()) & (data[self.uid].str.len() == self.confs.uid_length)
        invalid_uid = invalid_uid[invalid_uid == 0].values.tolist()
        if invalid_uid:
            raise ValueError(
                f"Invalid {self.uid}: can only have digits in its length should" f" be {self.confs.uid_length}: Invalid items -> ({invalid_uid})"
            )

        if self.room_id not in data.columns:
            data[self.room_id] = data[self.bed_id].str.replace(r"\D", "", regex=True)

        old_data = self.load_residents_info_table()
        return self.validate_df_new_with_old(old_df=old_data, new_df=data, check_if_exists_in_old=check_if_exists_in_old)

    def prepare_and_validate_elect_input(
        self,
        data: pd.DataFrame,
        check_if_exists_in_old: Optional[str | Sequence[str]] = None,
    ) -> pd.DataFrame:

        data.columns = data.columns.str.replace("Room_", "")
        date_cols = self.confs.date_cols_electricity_tbl
        float_cols = self.confs.float_cols_electricity_tbl
        str_cols = [col for col in data.columns if col not in date_cols + float_cols]

        data = self.convert_data_types(data=data, date_cols=date_cols, float_cols=float_cols, str_cols=str_cols)    # type: ignore
        self.float_colum_checks(data=data, float_cols=float_cols, allow_zero=True)
        self.date_colum_checks(
            data=data,
            date_cols=self.confs.date_cols_electricity_tbl,
            allow_nan=False,
            allow_duplicates=False,
        )

        old_data = self.load_electricity_table()
        valid_data = self.validate_df_new_with_old(old_df=old_data, new_df=data, check_if_exists_in_old=check_if_exists_in_old)

        valid_data.columns = valid_data.columns.map(lambda col: f"Room_{col}" if col.isdigit() else col)

        return valid_data

    def validate_entry_transaction(self, row: pd.Series):
        mandatory_cols = [self.uid, self.bed_id, "TransDate", "TransType", "RoomElectricityReading", "Comments"]
        self.validate_columns_or_index(data=row, mandatory_cols=mandatory_cols)
        return self._validate_entry_exit(row)

    def validate_exit_transaction(self, row: pd.Series):
        mandatory_cols = [self.uid, self.bed_id, "TransDate", "TransType", "RoomElectricityReading",
                          "PrevDueAmount", "AdditionalCharges", "RentThruDate", "Comments"]
        self.validate_columns_or_index(data=row, mandatory_cols=mandatory_cols)
        return self._validate_entry_exit(row)

    def _validate_entry_exit(self, row: pd.Series):
        if self.room_id not in row.index:
            row[f"{self.room_id}"] = pd.Series(row[self.bed_id]).str.replace(r"\D", "", regex=True).iloc[0]

        date_cols = self.confs.date_cols_transactions_tbl
        float_cols = self.confs.float_cols_transactions_tbl
        str_cols = [col for col in row.index if col not in date_cols + float_cols]
        row = self.convert_data_types(data=row, date_cols=date_cols, float_cols=float_cols, str_cols=str_cols)  # type: ignore
        self.check_valid_bedID(row[self.bed_id], valid_ids=self.confs.valid_bedIDs)

        if (row[self.uid] == "") or pd.isna(row[self.uid]):
            raise ValueError(f"{self.uid} cannot be null or empty string")

        if pd.isna(row["RoomElectricityReading"]):
            raise ValueError("RoomElectricityReading cannot be nan")

        prev_reading = self.load_current_status(filter_bedId=row[self.bed_id], filter_cols=["RoomElectricityReading"]).squeeze()
        if row["RoomElectricityReading"] < prev_reading:
            raise ValueError(f"RoomElectricityReading ({row['RoomElectricityReading']}) cannot be less than previous reading ({prev_reading})")

        return row

    def validate_payment_transaction(self, row: pd.Series):
        curr_status = self.load_current_status()
        mandatory_cols = [self.uid, self.bed_id, "TransDate", "TransType", "PaymentAmount", "Comments"]
        self.validate_columns_or_index(data=row, mandatory_cols=mandatory_cols, if_missing="raise")

        if self.room_id not in row.index:
            row[f"{self.room_id}"] = pd.Series(row[self.bed_id]).str.replace(r"\D", "", regex=True).iloc[0]

        row["TransDate"] = pd.to_datetime(row["TransDate"])
        row["PaymentAmount"] = pd.to_numeric(row["PaymentAmount"], errors="coerce")

        if pd.isna(row["PaymentAmount"]):
            ValueError("Cannot have nan in Payment Amount")

        if row[f"{self.uid}"] not in curr_status[self.uid].values:
            raise ValueError("cannot accept a payment for a resident who is not staying")
        return row

    def validate_transaction(self, row: pd.Series):

        trans_type = row["TransType"]

        if trans_type not in ["entry", "exit", "payment"]:
            raise ValueError("Found Invalid Transaction. Cannot Validate")

        if trans_type == "entry":
            valid_row = self.validate_entry_transaction(row=row)

        if trans_type == "exit":
            valid_row = self.validate_exit_transaction(row=row)

        if trans_type == "payment":
            valid_row = self.validate_payment_transaction(row=row)

        return valid_row

    def prepare_and_validate_status(self, data: pd.DataFrame) -> pd.DataFrame:

        self.check_valid_bedID(data[self.bed_id].values.tolist(), valid_ids=self.confs.valid_bedIDs)

        if len(data) != len(self.confs.valid_bedIDs):
            raise ValueError(
                f"Invalid Status Update ! Total number of records ({len(data)}) "
                f" do not match number of valid bedID ({len(self.confs.valid_bedIDs)})"
            )

        if data[self.bed_id].isna().any():
            raise ValueError("Bed Id in current status cannot be nan")

        if data[self.bed_id].duplicated().any():
            raise ValueError("Duplicate bed ID in status")

        occupied_beds = data[data[self.uid].notna()]
        if occupied_beds[self.uid].duplicated().any():
            raise ValueError("Duplicated UIDs exist at occupying beds in Status")

        return data.sort_values([self.room_id], key=lambda x: x.astype(int))

    def prepare_validate_room_transfer_input(self, row_input: pd.Series) -> pd.Series:

        current_status = self.load_current_status()
        residents_info = self.load_residents_info_table()

        date_cols = ["TransDate"]
        float_cols = [
            "SourceElectReading",
            "SourceResidentOldRent",
            "SourceResidentNewRent",
            "SourceResidentOldDeposit",
            "SourceResidentNewDeposit",
            "DestinationElectReading",
            "DestinationResidentOldRent",
            "DestinationResidentNewRent",
            "DestinationResidentOldDeposit",
            "DestinationResidentNewDeposit",
        ]
        str_cols = ["TransType", "SourceBedId", "SourceEnrollmentID", "DestinationBedId", "DestinationEnrollmentID", "Comments"]

        row_input = self.convert_data_types(data=row_input, date_cols=date_cols, float_cols=float_cols, str_cols=str_cols)    # type: ignore

        row_input["SourceRoomNo"] = pd.Series(row_input["SourceBedId"]).str.replace(r"\D", "", regex=True).iloc[0]
        row_input["DestinationRoomNo"] = pd.Series(row_input["DestinationBedId"]).str.replace(r"\D", "", regex=True).iloc[0]
        row_input["IsSwapping"] = True if row_input.get("TransType", None) == "Swapping" else False

        source_bed = row_input["SourceBedId"]
        destination_bed = row_input["DestinationBedId"]
        transfer_type = row_input["TransType"]
        swap = row_input["IsSwapping"]

        self.check_valid_bedID(bed_id=[source_bed, destination_bed], valid_ids=self.confs.valid_bedIDs)

        if source_bed == destination_bed:
            raise ValueError("Source and Destination Bed IDs cannot be the same.")

        if row_input["TransDate"] > dt.datetime.now():
            raise ValueError("Transfer date cannot be in the future.")

        prev_src_reading = current_status.loc[current_status[self.bed_id] == source_bed, "RoomElectricityReading"].squeeze()
        if (row_input["SourceElectReading"] < prev_src_reading) or (pd.isna(row_input["SourceElectReading"])):
            raise ValueError("Source electricity reading cannot be less than the previous reading or cannot be nan.")

        prev_dest_reading = current_status.loc[current_status[self.bed_id] == destination_bed, "RoomElectricityReading"].squeeze()
        if (row_input["DestinationElectReading"] < prev_dest_reading) or (pd.isna(row_input["DestinationElectReading"])):
            raise ValueError("Destination electricity reading cannot be less than the previous reading.")

        if transfer_type not in ["Moving to Empty Bed", "Swapping"]:
            raise ValueError(f"Invalid TransferType: {transfer_type}")

        if not swap and not pd.isna(row_input["DestinationEnrollmentID"]):
            raise ValueError(
                "Destination Enrollment ID should not be provided when moving to an empty bed. " f"Found: {row_input['DestinationEnrollmentID']}"
            )

        if swap and not row_input["DestinationEnrollmentID"]:
            raise ValueError("Destination Enrollment ID is required when swapping.")

        src_rent_deposit = residents_info.loc[residents_info[self.uid] == row_input["SourceEnrollmentID"], ["Rent", "Deposit"]].to_numpy()
        src_input_rent_deposit = row_input[["SourceResidentOldRent", "SourceResidentOldDeposit"]].to_numpy()
        if not (src_input_rent_deposit == src_rent_deposit).all():
            raise ValueError(
                f"Source Rent/Deposit does not match. value is database '{src_rent_deposit}'. "
                f"Values in input '{src_input_rent_deposit}'"
            )    # type: ignore

        if row_input["TransType"] == "Swapping":
            des_rent_deposit = residents_info.loc[residents_info[self.uid] == row_input["DestinationEnrollmentID"], ["Rent", "Deposit"]].to_numpy()
            des_input_rent_deposit = row_input[["DestinationResidentOldRent", "DestinationResidentOldDeposit"]].to_numpy()
            if not (des_input_rent_deposit == des_rent_deposit).all():
                raise ValueError(
                    f"Destination Rent/Deposit does not match. value is database '{des_rent_deposit}'. "
                    f"Values in input '{des_input_rent_deposit}'"
                )    # type: ignore

        return row_input

    def prepare_validate_final_setllement(self):
        pass

    def prepare_validate_rent_history(self):
        pass

    def valiate_src_exit(self):
        return
