import datetime as dt
import os
import sqlite3
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import shutil
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



    def process_transaction(self, input_df: pd.DataFrame, log_comments: Optional[str] = None):
        """Process a transaction which could entry or exit
        input_df = (TransDate, BedID, EnrollmentID, RoomElectricityreading, TransType)
        """

        valid_input = self.data_manager.prepare_and_validate_trans_input(input_df).sort_values("TransDate")
        for _, row in valid_input.iterrows():
            if row["TransType"] == "exit":
                res = self._process_resident_exit(row=row, log_comments=log_comments)
                self.data_manager.insert_transaction(input_df=valid_input)
                return res

            if row["TransType"] == "entry":
                res = self._process_resident_entry(row=row, log_comments=log_comments)
                self.data_manager.insert_transaction(input_df=valid_input)
                return res
        return

    @log_and_backup()
    def _process_resident_exit(self, row: pd.Series, monthly_factor: float = 30, log_comments: Optional[str] = None):
        """Process the workflow when a resident exits"""

        curr_status = self.data_manager.load_current_status()
        if not curr_status[self.uid].isin([row[self.uid]]).any():
            raise ValueError(f"Inlavid Exit. {self.uid}:{row[self.uid]} does not exist in status")

        curr_bed = curr_status.loc[curr_status[self.uid] == row[self.uid], self.bed_id].squeeze()
        if curr_bed != row[self.bed_id]:
            raise ValueError(f"Invalid Exit. The bedID of exiting {self.uid} does not match this uid in status")


        row_with_calc = self._electricity_adjustment(adj_type="exit", row=row)
        row_with_calc = row_with_calc.rename(columns={"TransDate": "ExitDate","RoomElectricityReading": "ExitElectricityReading",})
        resident_info = self.data_manager.load_residents_info_table(filter_ids=row_with_calc[self.uid].values, filter_cols=[self.uid, "Name","Rent", "Deposit"])
        row_with_calc = row_with_calc.merge(resident_info, on=self.uid, how="left")

        row_with_calc["RentThruDate"] = row["RentThruDate"]
        row_with_calc["PrevDueAmount"] = row["PrevDueAmount"]
        row_with_calc["AdditionalCharges"] = row["AdditionalCharges"]
        row_with_calc["Comments"] = row["Comments"]

        row_with_calc["RentDays"] = (row_with_calc["RentThruDate"] - row_with_calc["LastRentCalcDate"]).dt.days
        row_with_calc["RentDue"] = (row_with_calc["RentDays"] * row_with_calc["Rent"] / monthly_factor)
        row_with_calc["ElectricityCharges"] = (row_with_calc["CumulativeElectConsumption"] * self.confs.elect_rate)

        row_with_calc["TotalAmountDue"] = row_with_calc["RentDue"] + row_with_calc["ElectricityCharges"] + row_with_calc["PrevDueAmount"] + row_with_calc["AdditionalCharges"]
        row_with_calc["NetAmountDue"] = (row_with_calc["TotalAmountDue"] - row_with_calc["Deposit"])
        self.data_manager.insert_final_settlement_record(row_with_calc)
        return row_with_calc


    @log_and_backup()
    def _process_resident_entry(self, row: pd.Series, log_comments: Optional[str] = None):
        """Process the workflow when a resident enters"""

        curr_status = self.data_manager.load_current_status()
        residents_info = self.data_manager.load_residents_info_table()

        uit_at_entering_bed = curr_status.loc[curr_status[self.bed_id]==row[self.bed_id], self.uid].squeeze()
        if not pd.isna(uit_at_entering_bed):
            raise ValueError(f"Invalid Entry. The BedID {row[self.bed_id]} is not empty.")

        if not row[self.uid] in residents_info[self.uid].values:
            raise ValueError(f"{self.uid}: {row[self.uid]} does not exist in databse. Please create an admission or enter valid Enrollment ID")

        if row[self.uid] in curr_status[self.uid].values:
            curr_room = curr_status[curr_status[self.uid] == row[self.uid], self.bed_id].squeeze()
            raise ValueError(f"Invalid Entry. The Enrollment id is already staying in Bed {curr_room}")

        return self._electricity_adjustment(adj_type="entry", row=row)


    def _electricity_adjustment(self, adj_type: str, row: pd.Series):
        """To handle the electricity calulation logic based on entry or exit of the resident
        If exiting, then remvoe the record from current status
        If entering, then add the record to the current status
        It also sets the Last rent and electricity calculation dates to transaction date
        """

        curr_status = self.data_manager.load_current_status()
        room_prev_vals = self._pre_transaction_room_elect_adjustment(
            curr_status=curr_status,
            trans_room=row[self.room_id],
            trans_date=row["TransDate"],
            room_elect_reading=row["RoomElectricityReading"],
        )
        empty_rooms = curr_status[curr_status[self.uid].isna()]
        occupied_room = curr_status[curr_status[self.uid].notna()]
        occupied_room = pd.concat([occupied_room, room_prev_vals[occupied_room.columns]]).drop_duplicates(subset=[self.uid], keep="last")
        curr_status = pd.concat([empty_rooms, occupied_room])
        curr_status.loc[curr_status[self.room_id]==row[self.room_id], "RoomElectricityReading"] = row["RoomElectricityReading"]

        exiting_record = None
        if adj_type == "exit":
            exiting_record = room_prev_vals[room_prev_vals[self.uid] == row[self.uid]].copy()
            cols_to_chng = [col for col in curr_status.columns if col not in [self.room_id, self.bed_id, "RoomElectricityReading", "LastElectricityCalcDate"]]
            curr_status.loc[curr_status[self.uid] == row[self.uid], cols_to_chng] = np.nan

        if adj_type == "entry":
            row_df = row.to_frame().T
            row_df["CumulativeElectConsumption"] = 0
            last_rent_calc_dt = self.data_manager.load_residents_info_table(filter_ids=[row[self.uid]], filter_cols=["RentStartDate"]).squeeze()
            row_df["LastRentCalcDate"] = last_rent_calc_dt - dt.timedelta(days=1)
            row_df["LastElectricityCalcDate"] = row["TransDate"]
            # Updating new entry alredy staying
            curr_status = pd.concat([curr_status, row_df[curr_status.columns]]).drop_duplicates(subset=[self.bed_id], keep="last")

        self.data_manager.update_current_status(new_status=curr_status)
        return exiting_record


    def _pre_transaction_room_elect_adjustment(
        self, curr_status, trans_room, trans_date, room_elect_reading
    ):
        """
        This function handles the calculation before any entry or exit in the room. It gets the uids of
        residents staying in the room. And calculate the total electricity consumption just before the transaction
        and divides the electricity among all residents staying equally before transaction happens
        """
        room_prev_vals = curr_status[curr_status[self.room_id] == trans_room].copy()
        room_prev_vals = room_prev_vals[room_prev_vals[self.uid].notna()]
        room_prev_vals["TransDate"] = pd.to_datetime(trans_date)
        room_prev_vals["NewRoomElectricityReading"] = room_elect_reading
        room_prev_vals["RoomElectricityConsumption"] = (room_prev_vals["NewRoomElectricityReading"]- room_prev_vals["RoomElectricityReading"])
        room_prev_vals["CumulativeElectConsumption"] += (room_prev_vals["RoomElectricityConsumption"] / len(room_prev_vals))

        # room_prev_vals = room_prev_vals.drop(["RoomElectricityReading", "RoomElectricityConsumption"], axis=1)
        room_prev_vals = room_prev_vals.rename(columns={
            "RoomElectricityReading" : "PrevRoomElectricityReading",
            "NewRoomElectricityReading": "RoomElectricityReading"})
        room_prev_vals["PrevElectricityCalcDate"] = room_prev_vals["LastElectricityCalcDate"]
        room_prev_vals["LastElectricityCalcDate"] = room_prev_vals["TransDate"]
        return room_prev_vals

    @log_and_backup()
    def room_meter_change(self, room:str, trans_dt_time: dt.datetime, old_meter_reading: float, new_meter_reading: float, log_comments: Optional[str] = None ):
        curr_status = self.data_manager.load_current_status()
        room_adjusted = self._pre_transaction_room_elect_adjustment(curr_status=curr_status, trans_room=room, trans_date=trans_dt_time, room_elect_reading=old_meter_reading)
        empty_rooms = curr_status[curr_status[self.uid].isna()]
        occupied_room = curr_status[curr_status[self.uid].notna()]
        occupied_room = pd.concat([occupied_room, room_adjusted[occupied_room.columns]]).drop_duplicates(subset=[self.uid], keep="last")
        curr_status = pd.concat([empty_rooms, occupied_room])
        curr_status.loc[curr_status[self.room_id]==room, "RoomElectricityReading"] = new_meter_reading
        self.data_manager.update_current_status(new_status=curr_status)

        return


    def calculate_rent(
        self,
        eom_rent_calc_date,
        update_and_save = False,
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
            raise ValueError("There seems to be some issue. The difference between rent calculation date and last elect record is > {elec_read_days_tolerance} days")

        curr_status = self.data_manager.load_current_status()
        latest_elect = elect_records.tail(1).squeeze().rename("EOMElectReading")

        empty_beds = curr_status[curr_status[self.uid].isna()]
        occupied_beds = curr_status[curr_status[self.uid].notna()]

        if (occupied_beds["LastRentCalcDate"] < prev_eom_rent_calc_date).any():
            raise ValueError(f"Previous month's rent is not yet settled. Still have LastRentCalcDate before {prev_eom_rent_calc_date}")

        if (occupied_beds["LastRentCalcDate"] >= eom_rent_calc_date).any():
            raise ValueError(f"Invalid Calculation: Found LastRentCalcDate >= {eom_rent_calc_date}. Cannot have an entry of resident in new month before setlling prev month rent.")

        # Adding required fields form other tables
        occupied_beds = occupied_beds.merge(latest_elect, left_on=self.room_id, right_index=True, how="left")
        occupied_beds = occupied_beds.merge(residents_info[[self.uid, "Name", "Rent", "Deposit"]], on=self.uid, how="left")


        occupied_beds["RoomElectricityConsumption"] = (occupied_beds["EOMElectReading"] - occupied_beds["RoomElectricityReading"])
        occupied_beds["CumulativeElectConsumption"] += (occupied_beds["RoomElectricityConsumption"] / occupied_beds.groupby(self.room_id)["RoomElectricityConsumption"].transform("count"))
        occupied_beds["PrevElectricityReading"] = occupied_beds["RoomElectricityReading"]
        occupied_beds["RoomElectricityReading"] = occupied_beds["EOMElectReading"]
        occupied_beds["UnitsConsumed"] = occupied_beds["CumulativeElectConsumption"]
        occupied_beds["CumulativeElectConsumption"] = 0
        occupied_beds["ElectricityCharges"] = occupied_beds["UnitsConsumed"] * self.confs.elect_rate
        occupied_beds["PrevElectricityCalcDate"] = occupied_beds["LastElectricityCalcDate"]
        occupied_beds["LastElectricityCalcDate"] = eom_rent_calc_date

        occupied_beds["TransDate"] = eom_rent_calc_date

        # occupied_beds = occupied_beds.drop(["RoomElectricityConsumption","EOMElectReading"], axis=1)
        occupied_beds["RentDays"] = np.where(
            occupied_beds["LastRentCalcDate"]==prev_eom_rent_calc_date,
            monthly_factor,
            (eom_rent_calc_date - occupied_beds["LastRentCalcDate"]).dt.days
        )

        occupied_beds["RentDue"] = occupied_beds["RentDays"] * occupied_beds["Rent"] / monthly_factor
        occupied_beds["TotalAmountDue"] = (occupied_beds["RentDue"] + occupied_beds["ElectricityCharges"])
        occupied_beds["PrevRentCalcDate"] = occupied_beds["LastRentCalcDate"]
        occupied_beds["LastRentCalcDate"] = eom_rent_calc_date

        # Chaning qppropiate columns in the empty beds
        empty_beds["LastElectricityCalcDate"] = eom_rent_calc_date
        empty_beds = empty_beds.merge(latest_elect, left_on=self.room_id, right_index=True, how="left")
        empty_beds["RoomElectricityReading"] = empty_beds["EOMElectReading"]


        new_status = pd.concat([empty_beds[curr_status.columns], occupied_beds[curr_status.columns]])
        report = pd.concat([occupied_beds, empty_beds])
        report = report[rent_history.columns]
        report[self.confs.date_cols_rent_history_tbl] = report[self.confs.date_cols_rent_history_tbl].apply(lambda x : x.dt.strftime("%d-%b-%Y"))

        if update_and_save:
            self.update_and_save_rent_calculation(new_status=new_status, rent_history=report)

        return report

    @log_and_backup()
    def update_and_save_rent_calculation(self, new_status: pd.DataFrame, rent_history: pd.DataFrame):
        self.data_manager.update_current_status(new_status=new_status)
        self.data_manager.insert_rent_history(input_df=rent_history)
