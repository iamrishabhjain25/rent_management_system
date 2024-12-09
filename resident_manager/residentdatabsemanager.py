import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
from utils import log_and_backup


class ResidentManager:
    """Main class to handle the computation of all rent logic"""

    def __init__(self, confs, db_handler, data_manager):
        self.confs = confs
        self.db_handler = db_handler
        self.data_manager = data_manager
        self.uid = confs.uid_col
        self.bed_id = confs.bedId_col
        self.room_id = confs.room_col

    def process_transaction(self, transaction: pd.Series, log_comments: Optional[str] = None):
        valid_transaction = self.data_manager.validate_transaction(row=transaction)
        trans_type = valid_transaction["TransType"]

        if trans_type == "exit":
            res = self._process_resident_exit(row=valid_transaction, log_comments=log_comments)
            self.data_manager.insert_final_settlement_record(res)

        if trans_type == "entry":
            res = self._process_resident_entry(row=valid_transaction, log_comments=log_comments)

        if trans_type in ["payment", "charges"]:
            res = self._process_resident_payment(row=valid_transaction, log_comments=log_comments)

        self.data_manager.insert_transaction(input_df=valid_transaction.to_frame().T)
        return res

    @log_and_backup()
    def _process_resident_payment(self, row: pd.Series, log_comments: Optional[str] = None):
        curr_status = self.data_manager.load_current_status()
        prev_due = curr_status.loc[curr_status[self.uid] == row[self.uid], "PrevDueAmount"].squeeze()
        prev_charges = curr_status.loc[curr_status[self.uid] == row[self.uid], "AdditionalCharges"].squeeze()

        if row["TransType"] == "payment":
            curr_status.loc[curr_status[self.uid] == row[self.uid], "PrevDueAmount"] = prev_due - row["TransactionAmount"]

        if row["TransType"] == "charges":
            curr_status.loc[curr_status[self.uid] == row[self.uid], "AdditionalCharges"] = prev_charges + row["TransactionAmount"]

        self.data_manager.update_current_status(new_status=curr_status)
        return

    @log_and_backup()
    def _process_resident_exit(
        self,
        row: pd.Series,
        monthly_factor: float = 30,
        log_comments: Optional[str] = None,
    ):
        """Process the workflow when a resident exits"""

        curr_status = self.data_manager.load_current_status()
        if not curr_status[self.uid].isin([row[self.uid]]).any():
            raise ValueError(f"Inlavid Exit. {self.uid}:{row[self.uid]} does not exist in status")

        curr_bed = curr_status.loc[curr_status[self.uid] == row[self.uid], self.bed_id].squeeze()
        if curr_bed != row[self.bed_id]:
            raise ValueError(f"Invalid Exit. The bedID of exiting {self.uid} does not match this uid in status")

        prev_dues = self.data_manager.load_current_status(filter_bedId=row[self.bed_id], filter_cols="PrevDueAmount").squeeze()
        prev_charges = self.data_manager.load_current_status(filter_bedId=row[self.bed_id], filter_cols="AdditionalCharges").squeeze()

        adj_room_status = self._pre_transaction_room_elect_adjustment(
            curr_status=curr_status,
            trans_room=row[self.room_id],
            trans_date=row["TransDate"],
            room_elect_reading=row["RoomElectricityReading"],
        )
        curr_status = curr_status[curr_status[self.room_id] != row[self.room_id]]
        curr_status = pd.concat([curr_status, adj_room_status[curr_status.columns]])
        cols_to_chng = [
            col
            for col in curr_status.columns if col not in [
                self.room_id, self.bed_id, "RoomElectricityReading", "LastElectricityCalcDate", "TransDate"
                ]
                ]
        curr_status.loc[curr_status[self.uid] == row[self.uid], cols_to_chng] = np.nan
        self.data_manager.update_current_status(new_status=curr_status)

        row_with_calc = adj_room_status[adj_room_status[self.uid] == row[self.uid]].copy()
        row_with_calc = row_with_calc.rename(
            columns={
                "TransDate": "ExitDate",
                "RoomElectricityReading": "ExitElectricityReading",
            }
        )
        resident_info = self.data_manager.load_residents_info_table(
            filter_ids=row_with_calc[self.uid].values,
            filter_cols=[self.uid, "Name", "Rent", "Deposit"],
        )
        row_with_calc = row_with_calc.merge(resident_info, on=self.uid, how="left")

        row_with_calc["RentThruDate"] = row["RentThruDate"]
        row_with_calc["PrevDueAmount"] = prev_dues
        row_with_calc["AdditionalCharges"] = prev_charges
        row_with_calc["Comments"] = row["Comments"]

        row_with_calc["RentDays"] = (row_with_calc["RentThruDate"] - row_with_calc["LastRentCalcDate"]).dt.days
        row_with_calc["RentDue"] = row_with_calc["RentDays"] * row_with_calc["Rent"] / monthly_factor
        row_with_calc["ElectricityCharges"] = row_with_calc["CumulativeElectConsumption"] * self.confs.elect_rate

        row_with_calc["TotalAmountDue"] = (
            row_with_calc["RentDue"]
            + row_with_calc["ElectricityCharges"]
            + row_with_calc["PrevDueAmount"]
            + row_with_calc["AdditionalCharges"]
        )
        row_with_calc["NetAmountDue"] = row_with_calc["TotalAmountDue"] - row_with_calc["Deposit"]
        return row_with_calc

    @log_and_backup()
    def _process_resident_entry(self, row: pd.Series, log_comments: Optional[str] = None):
        """Process the workflow when a resident enters"""

        curr_status = self.data_manager.load_current_status()
        residents_info = self.data_manager.load_residents_info_table()

        uit_at_entering_bed = curr_status.loc[curr_status[self.bed_id] == row[self.bed_id], self.uid].squeeze()
        if not pd.isna(uit_at_entering_bed):
            raise ValueError(f"Invalid Entry. The BedID {row[self.bed_id]} is not empty.")

        if not row[self.uid] in residents_info[self.uid].values:
            raise ValueError(f"{self.uid}: {row[self.uid]} does not exist in databse. " "Please create an admission or enter valid Enrollment ID")

        if row[self.uid] in curr_status[self.uid].values:
            curr_room = curr_status.loc[curr_status[self.uid] == row[self.uid], self.bed_id].squeeze()
            raise ValueError(f"Invalid Entry. The Enrollment id is already staying in Bed {curr_room}")

        adj_room_status = self._pre_transaction_room_elect_adjustment(
            curr_status=curr_status,
            trans_room=row[self.room_id],
            trans_date=row["TransDate"],
            room_elect_reading=row["RoomElectricityReading"],
        )
        curr_status = curr_status[curr_status[self.room_id] != row[self.room_id]]
        curr_status = pd.concat([curr_status, adj_room_status[curr_status.columns]])

        row_df = row.to_frame().T
        row_df["CumulativeElectConsumption"] = 0
        admission_date = self.data_manager.load_residents_info_table(filter_ids=[row[self.uid]], filter_cols=["RentStartDate"]).squeeze()
        row_df["LastRentCalcDate"] = row["LastRentCalcDate"] if "LastRentCalcDate" in row.index else (admission_date - dt.timedelta(days=1))
        row_df["LastElectricityCalcDate"] = row["TransDate"]

        for col in ["PrevDueAmount", "AdditionalCharges"]:
            if col not in row_df.columns:
                row[col] = 0.0
        # Updating new entry alredy staying
        curr_status = pd.concat([curr_status, row_df[curr_status.columns]]).drop_duplicates(subset=[self.bed_id], keep="last")
        self.data_manager.update_current_status(new_status=curr_status)

        return

    def _pre_transaction_room_elect_adjustment(self, curr_status, trans_room, trans_date, room_elect_reading) -> pd.DataFrame:
        """
        This function handles the calculation before any entry or exit in the room.
        It gets the uids ofresidents staying in the room. And calculate the total
        electricity consumption just before the transactionand divides the electricity
        among all residents staying equally before transaction happens
        """

        room_status = curr_status[curr_status[self.room_id] == trans_room].copy()
        empty_bed = room_status[room_status[self.uid].isna()][self.bed_id].to_numpy()
        occupied_bed = room_status[room_status[self.uid].notna()][self.bed_id].to_numpy()

        if len(empty_bed) > 0:
            if room_status[room_status[self.uid].isna()]["CumulativeElectConsumption"].notna().any():
                raise ValueError(f"Empty bed cannot have non nan CumulativeElectConsumption. {empty_bed}")

        room_status = room_status[room_status[self.uid].notna()]
        room_status["TransDate"] = pd.to_datetime(trans_date)
        room_status["NewRoomElectricityReading"] = room_elect_reading
        room_status["RoomElectricityConsumption"] = room_status["NewRoomElectricityReading"] - room_status["RoomElectricityReading"]
        room_status["CumulativeElectConsumption"] += room_status["RoomElectricityConsumption"] / len(occupied_bed)

        room_status = room_status.rename(
            columns={
                "RoomElectricityReading": "PrevRoomElectricityReading",
                "NewRoomElectricityReading": "RoomElectricityReading",
            }
        )
        room_status["PrevElectricityCalcDate"] = room_status["LastElectricityCalcDate"]
        room_status["LastElectricityCalcDate"] = room_status["TransDate"]

        return room_status

    @log_and_backup()
    def process_room_transfers(self, row: pd.Series):

        valid_row = self.data_manager.prepare_validate_room_transfer_input(row_input=row)

        src_exit, transaction_src = self._source_room_exit(input_row=valid_row)
        if valid_row["IsSwapping"]:
            desti_exit, transaction_des = self._destination_room_exit(input_row=valid_row)

        self._entry_in_destination(input_row=valid_row, src_exit_details=src_exit)
        if valid_row["IsSwapping"]:
            self._entry_in_source(input_row=valid_row, desti_exit_details=desti_exit)

        self.data_manager.insert_transaction(transaction_src)

        if valid_row["IsSwapping"]:
            self.data_manager.insert_transaction(transaction_des)
        return

    def _source_room_exit(self, input_row: pd.Series):
        src_resident_info = self.data_manager.load_residents_info_table(filter_ids=input_row["SourceEnrollmentID"])
        data = {
            "TransDate": input_row["TransDate"],
            "RentThruDate": input_row["TransDate"],
            f"{self.bed_id}": input_row["SourceBedId"],
            f"{self.uid}": input_row["SourceEnrollmentID"],
            f"{self.room_id}": input_row["SourceRoomNo"],
            "RoomElectricityReading": input_row["SourceElectReading"],
            "TransType": "exit",
            "PrevDueAmount": self.data_manager.load_current_status(filter_bedId=input_row["SourceBedId"], filter_cols="PrevDueAmount").squeeze(),
            "AdditionalCharges": self.data_manager.load_current_status(
                filter_bedId=input_row["SourceBedId"], filter_cols="AdditionalCharges"
            ).squeeze(),
            "Comments": input_row["Comments"],
            "TransactionAmount": np.nan,
            "Rent": input_row["SourceResidentOldRent"],
            "Deposit": input_row["SourceResidentOldDeposit"],
            "NewBedID": input_row["DestinationBedId"],
            "NewBedIDElectReading": input_row["DestinationElectReading"],
            "NewRent": input_row["SourceResidentNewRent"],
            "NewDeposit": input_row["SourceResidentNewDeposit"],
        }
        exit_details = self._process_resident_exit(row=pd.Series(data), copy_db=False)
        exit_details["TotalAmountDue"] = exit_details["TotalAmountDue"] + data["NewDeposit"] - data["Deposit"]
        exit_details["NetAmountDue"] = np.nan
        data["TransType"] = "Room Transfer"

        src_resident_info["Rent"] = input_row["SourceResidentNewRent"]
        src_resident_info["Deposit"] = input_row["SourceResidentNewDeposit"]

        self.data_manager.edit_resident_record(
            new_resident_record=src_resident_info, log_comments="Updating Rent and Deposit of Source UID", copy_db=False
        )
        self.data_manager.insert_final_settlement_record(exit_details)

        return exit_details, pd.DataFrame([data])

    def _destination_room_exit(self, input_row: pd.Series):
        des_resident_info = self.data_manager.load_residents_info_table(filter_ids=input_row["DestinationEnrollmentID"])
        data = {
            "TransDate": input_row["TransDate"],
            "RentThruDate": input_row["TransDate"],
            f"{self.bed_id}": input_row["DestinationBedId"],
            f"{self.uid}": input_row["DestinationEnrollmentID"],
            f"{self.room_id}": input_row["DestinationRoomNo"],
            "RoomElectricityReading": input_row["DestinationElectReading"],
            "TransType": input_row["TransType"],
            "PrevDueAmount": self.data_manager.load_current_status(filter_bedId=input_row["DestinationBedId"], filter_cols="PrevDueAmount").squeeze(),
            "AdditionalCharges": self.data_manager.load_current_status(
                filter_bedId=input_row["DestinationBedId"], filter_cols="AdditionalCharges"
            ).squeeze(),
            "Comments": input_row["Comments"],
            "TransactionAmount": np.nan,
            "Rent": input_row["DestinationResidentOldRent"],
            "Deposit": input_row["DestinationResidentOldDeposit"],
            "NewBedID": input_row["SourceBedId"],
            "NewBedIDElectReading": input_row["SourceElectReading"],
            "NewRent": input_row["DestinationResidentNewRent"],
            "NewDeposit": input_row["DestinationResidentNewDeposit"],
        }
        exit_details = self._process_resident_exit(row=pd.Series(data), copy_db=False)
        exit_details["TotalAmountDue"] += data["NewDeposit"] - data["Deposit"]
        exit_details["NetAmountDue"] = np.nan
        data["TransType"] = "Room Transfer"

        des_resident_info["Rent"] = input_row["SourceResidentNewRent"]
        des_resident_info["Deposit"] = input_row["SourceResidentNewDeposit"]

        self.data_manager.edit_resident_record(
            new_resident_record=des_resident_info, log_comments="Updating Rent and Deposit of Source UID", copy_db=False
        )
        self.data_manager.insert_final_settlement_record(exit_details)

        return exit_details, pd.DataFrame([data])

    def _entry_in_destination(self, input_row: pd.Series, src_exit_details: pd.DataFrame):
        data = {
            "TransDate": input_row["TransDate"],
            "RentThruDate": np.nan,
            f"{self.bed_id}": input_row["DestinationBedId"],
            f"{self.uid}": input_row["SourceEnrollmentID"],
            f"{self.room_id}": input_row["DestinationRoomNo"],
            "RoomElectricityReading": input_row["DestinationElectReading"],
            "TransType": "entry",
            "PrevDueAmount": (src_exit_details["TotalAmountDue"]).squeeze(),
            "AdditionalCharges": 0,
            "Comments": input_row["Comments"],
            "LastRentCalcDate": input_row["TransDate"],
        }
        self._process_resident_entry(row=pd.Series(data), copy_db=False)
        return

    def _entry_in_source(self, input_row: pd.Series, desti_exit_details: pd.DataFrame):
        data = {
            "TransDate": input_row["TransDate"],
            "RentThruDate": np.nan,
            f"{self.bed_id}": input_row["SourceBedId"],
            f"{self.uid}": input_row["DestinationEnrollmentID"],
            f"{self.room_id}": input_row["SourceRoomNo"],
            "RoomElectricityReading": input_row["SourceElectReading"],
            "TransType": "entry",
            "PrevDueAmount": (desti_exit_details["TotalAmountDue"] - desti_exit_details["AdditionalCharges"]).squeeze(),
            "AdditionalCharges": desti_exit_details["AdditionalCharges"].squeeze(),
            "Comments": input_row["Comments"],
            "LastRentCalcDate": input_row["TransDate"],
        }

        return self._process_resident_entry(row=pd.Series(data), copy_db=False)

    @log_and_backup()
    def room_meter_change(
        self,
        room: str,
        trans_dt_time: dt.datetime,
        old_meter_reading: float,
        new_meter_reading: float,
        log_comments: Optional[str] = None,
    ):
        curr_status = self.data_manager.load_current_status()
        room_adjusted = self._pre_transaction_room_elect_adjustment(
            curr_status=curr_status,
            trans_room=room,
            trans_date=trans_dt_time,
            room_elect_reading=old_meter_reading,
        )
        curr_status = curr_status[curr_status[self.room_id] != room]
        curr_status = pd.concat([curr_status, room_adjusted[curr_status.columns]])

        curr_status.loc[curr_status[self.room_id] == room, "RoomElectricityReading"] = new_meter_reading
        self.data_manager.update_current_status(new_status=curr_status)

        return

    def calculate_rent(
        self,
        eom_rent_calc_date,
        update_and_save=False,
        elec_read_days_tolerance: int = 3,
        monthly_factor: float = 30,
    ):

        eom_rent_calc_date = pd.to_datetime(eom_rent_calc_date)
        prev_eom_rent_calc_date = eom_rent_calc_date.replace(day=1) - dt.timedelta(days=1)

        elect_records = self.data_manager.load_electricity_table()
        latest_elect_dt = max(elect_records["Date"])
        residents_info = self.data_manager.load_residents_info_table()
        rent_history = self.data_manager.load_rent_history()

        if abs((eom_rent_calc_date - latest_elect_dt).days) > elec_read_days_tolerance:
            raise ValueError(
                "There seems to be some issue. The difference between rent"
                f" calculation date and last elect record is > {elec_read_days_tolerance} days"
            )

        curr_status = self.data_manager.load_current_status()
        latest_elect = elect_records.tail(1).squeeze().rename("EOMElectReading")

        empty_beds = curr_status[curr_status[self.uid].isna()]
        occupied_beds = curr_status[curr_status[self.uid].notna()]

        if (occupied_beds["LastRentCalcDate"] < prev_eom_rent_calc_date).any():
            raise ValueError(f"Previous month's rent is not yet settled. " f"Still have LastRentCalcDate before {prev_eom_rent_calc_date}")

        if (occupied_beds["LastRentCalcDate"] >= eom_rent_calc_date).any():
            raise ValueError(
                f"Invalid Calculation: Found LastRentCalcDate >= {eom_rent_calc_date}."
                " Cannot have an entry of resident in new month before setlling prev month rent."
            )

        # Adding required fields form other tables
        occupied_beds = occupied_beds.merge(latest_elect, left_on=self.room_id, right_index=True, how="left")
        occupied_beds = occupied_beds.merge(
            residents_info[[self.uid, "Name", "Rent", "Deposit"]],
            on=self.uid,
            how="left",
        )

        occupied_beds["RoomElectricityConsumption"] = occupied_beds["EOMElectReading"] - occupied_beds["RoomElectricityReading"]
        occupied_beds["CumulativeElectConsumption"] += occupied_beds["RoomElectricityConsumption"] / occupied_beds.groupby(self.room_id)[
            "RoomElectricityConsumption"
        ].transform("count")
        occupied_beds["PrevElectricityReading"] = occupied_beds["RoomElectricityReading"]
        occupied_beds["RoomElectricityReading"] = occupied_beds["EOMElectReading"]
        occupied_beds["UnitsConsumed"] = occupied_beds["CumulativeElectConsumption"]
        occupied_beds["CumulativeElectConsumption"] = 0
        occupied_beds["ElectricityCharges"] = occupied_beds["UnitsConsumed"] * self.confs.elect_rate
        occupied_beds["PrevElectricityCalcDate"] = occupied_beds["LastElectricityCalcDate"]
        occupied_beds["LastElectricityCalcDate"] = eom_rent_calc_date

        occupied_beds["TransDate"] = eom_rent_calc_date

        occupied_beds["RentDays"] = np.where(
            occupied_beds["LastRentCalcDate"] == prev_eom_rent_calc_date,
            monthly_factor,
            (eom_rent_calc_date - occupied_beds["LastRentCalcDate"]).dt.days,
        )

        occupied_beds["RentDue"] = occupied_beds["RentDays"] * occupied_beds["Rent"] / monthly_factor
        occupied_beds["TotalAmountDue"] = (
            occupied_beds["RentDue"] + occupied_beds["ElectricityCharges"] + occupied_beds["PrevDueAmount"] + occupied_beds["AdditionalCharges"]
        )
        occupied_beds["PrevRentCalcDate"] = occupied_beds["LastRentCalcDate"]
        occupied_beds["LastRentCalcDate"] = eom_rent_calc_date

        # Chaning qppropiate columns in the empty beds
        empty_beds["LastElectricityCalcDate"] = eom_rent_calc_date
        empty_beds = empty_beds.merge(latest_elect, left_on=self.room_id, right_index=True, how="left")
        empty_beds["RoomElectricityReading"] = empty_beds["EOMElectReading"]

        # Creating Full report
        report = pd.concat([occupied_beds, empty_beds])
        report = report[rent_history.columns]
        report[self.confs.date_cols_rent_history_tbl] = report[self.confs.date_cols_rent_history_tbl].apply(lambda x: x.dt.strftime("%d-%b-%Y"))

        # Updating PrevDue and additional charges for new status
        occupied_beds["PrevDueAmount"] = occupied_beds["TotalAmountDue"]
        occupied_beds["AdditionalCharges"] = 0

        # new status
        new_status = pd.concat([empty_beds, occupied_beds])
        new_status = new_status[curr_status.columns]

        if update_and_save:
            self.update_and_save_rent_calculation(new_status=new_status, rent_history=report)

        return report

    @log_and_backup()
    def update_and_save_rent_calculation(self, new_status: pd.DataFrame, rent_history: pd.DataFrame):
        self.data_manager.update_current_status(new_status=new_status)
        self.data_manager.insert_rent_history(input_df=rent_history)
