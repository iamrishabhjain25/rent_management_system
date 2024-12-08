import datetime as dt
import os
import traceback
from datetime import datetime
from typing import Optional

import configs
import numpy as np
import pandas as pd
import residentdatabsemanager as RDM
import streamlit as st
from utils import DatabaseHandler, DataManager

st.set_page_config(layout="wide")


class ResidenceManagementStreamlit:
    """Class to handle UI for resident Manager"""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    def contact_input(self, label: str, value=None, length: int = 10, key: Optional[str] = None):
        value = None if pd.isna(value) else value
        value = value if value else "0" * length
        if key is None:
            key = label

        contact = st.text_input(label=label, value=value, key=key)

        if len(contact) != length:
            st.error(f"length of input ({len(contact)}) does not match required length: ({length})")

        if not contact.isdigit():
            raise ValueError(f"only digits allowed in contact number, got({contact})")

        return contact

    def confirm_action_box(self, message):
        st.warning(message)
        confirm = st.button("Continue")
        cancel = st.button("Cancel")
        return confirm, cancel

    def new_admission(self):
        """Insert new resident record"""
        st.header("New Admission")

        generated_uid = self.db_manager.data_manager.generate_new_uid()
        data = {
            "EnrollmentID": st.text_input("Enrollment ID", value=generated_uid),
            "BedID": st.selectbox("Select Bed ID", configs.valid_bedIDs).upper(),
            "DateofAdmission": st.date_input("Date of Admission", value=datetime.today(), format="DD-MM-YYYY"),
            "Name": st.text_input("FulName"),
            "DateofBirth": st.date_input(
                "Date of Birth",
                min_value=datetime.strptime("1950-01-01", "%Y-%m-%d"),
                format="DD-MM-YYYY",
            ),
            "FathersName": st.text_input("Father's Full Name"),
            "ContactNumber": self.contact_input("Contact Number (10 digit)"),
            "OtherContact": self.contact_input("Other Contact Number (10 digit)"),
            "FathersContact": self.contact_input("Father's Contact Number (10 digit)"),
            "MothersContact": self.contact_input("Mother's Contact Number (10 digit)"),
            "AdditionalContact": self.contact_input("Addtional Contact Number (10 digit)"),
            "Email": st.text_input("Email ID", value="abcd.123_efg@xyz.com"),
            "Address": st.text_input(
                "Full Address",
                value="Write Full address with pincode cit and all details",
            ),
            "BloodGroup": st.text_input("Blood Group", value=""),
            "ResidentAadhar": st.text_input("Resident Aadhaar"),
            "FathersAadhar": st.text_input("Father's Aadhaar"),
            "InstituteName": st.text_input("Institute Name"),
            "InstituteAddress": st.text_input("Institute Address"),
            "Course": st.text_input("Course Name (JEE/NEET/IAS)"),
            "Batch": st.text_input("Batch"),
            "InstituteContact": self.contact_input("Institute Contact Number (10 digit)"),
            "InstituteContactOther": self.contact_input("Institute other Contact Number (10 digit)"),
            "InstituteID": st.text_input("Institute ID Number"),
            "GuardianName": st.text_input("Name of Guardian"),
            "GuardianRelation": st.text_input("Relation with Guardian"),
            "GuardianContact1": self.contact_input("Guardian Contact Number (10 digit)"),
            "GuardianContact2": self.contact_input("Guardian other Contact Number (10 digit)"),
            "GuardianAddress": st.text_input("Guardian Address"),
            "Rent": st.number_input("Monthly Rent"),
            "Deposit": st.number_input("Deposit"),
            "RentStartDate": st.date_input("Rent Start Date", format="DD-MM-YYYY"),
        }
        data = pd.DataFrame([data])
        if (data["Rent"] < 5000).any() or (data["Deposit"] < data["Rent"] * 2).any():
            message = "Warning: Either Rent if less than 5,000 or"
            " Deposite is less than twice the rent. Are you sure want to continue ?"
            st.warning(message, icon="⚠️")

        log_comments = st.text_input("Log comments for this Activity")
        if st.button("Submit Admission"):
            try:
                with st.spinner("Processing Admission, please wait"):
                    self.db_manager.data_manager.insert_resident_record(input_df=data, log_comments=log_comments)
                st.success("Admission processed successfully.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())

    def new_electricity_record(self):
        """Insert new electricity record"""

        st.header("New Electricity Record")

        elect_table = self.db_manager.data_manager.load_electricity_table().sort_values(["Date"])
        curr_status = self.db_manager.data_manager.load_current_status()
        curr_reading_status = curr_status[["RoomNo", "RoomElectricityReading"]].drop_duplicates().set_index("RoomNo").squeeze()

        reading_date = st.date_input("Electricity Reading Date", value=datetime.now().date(), format="DD-MM-YYYY")
        reading_time = st.time_input("Electricity Reading Time")
        reading_datetime = datetime.combine(reading_date, reading_time)

        st.warning("Please enter date and time carefully in DD-MM-YYYY format")

        prev_record = elect_table[elect_table["Date"] < reading_datetime].tail(1).squeeze()
        prev_record = curr_reading_status.combine_first(prev_record)
        prev_record["Date"] = np.nan

        if prev_record.empty:
            prev_record = pd.Series(0, index=elect_table.columns)
            prev_record.loc["Date"] = np.nan

        data = {"Date": reading_datetime}
        rooms = self.db_manager.data_manager.load_current_status()
        rooms = rooms[self.db_manager.room_id].unique()

        for a_room in rooms:
            data[a_room] = st.number_input(f"Enter Room {a_room} Meter Reading", value=prev_record[a_room])

        data.update(
            {
                "Meter_1_2A": st.number_input("Enter Main Meter 1 Reading", value=prev_record["Meter_1_2A"]),
                "Meter_2_2B": st.number_input("Enter Main Meter 2 Reading", value=prev_record["Meter_2_2B"]),
                "Meter_3_1A": st.number_input("Enter Main Meter 3 Reading", value=prev_record["Meter_3_1A"]),
                "Meter_4_1B": st.number_input("Enter Main Meter 4 Reading", value=prev_record["Meter_4_1B"]),
                "Meter_5_GA": st.number_input("Enter Main Meter 5 Reading", value=prev_record["Meter_5_GA"]),
                "Meter_6_GB": st.number_input("Enter Main Meter 6 Reading", value=prev_record["Meter_6_GB"]),
                "Meter_7_Basement": st.number_input("Enter Main Meter 7 Reading", value=prev_record["Meter_7_Basement"]),
                "Library": st.number_input("Enter Library Meter Reading", value=prev_record["Library"]),
                "Solar": st.number_input("Enter Solar Meter Reading", value=prev_record["Solar"]),
            }
        )

        data = pd.DataFrame([data])

        series_1 = data.squeeze()
        series_2 = prev_record.reindex(series_1.index)

        invalid_reading = series_1 <= series_2
        if invalid_reading.any():
            st.warning(
                f"Invalid Electricity Reading : Reading for "
                f"({invalid_reading[invalid_reading==1].index.tolist()}) < previous reading."
                "cannot enter this record"
            )
            return

        log_comments = st.text_input("Log comments for this Activity")
        if st.button("Submit Readings"):
            try:
                with st.spinner("Processing, please wait"):
                    self.db_manager.data_manager.insert_electricity_record(input_df=data, log_comments=log_comments)
                st.success("Electricity reading inserted successfully.")
            except Exception as e:
                st.error(f"Error: {e}")

    def update_resident_info(self):
        st.header("Change/Update Resident Info")

        if "resident_data" not in st.session_state:
            st.session_state.resident_data = None

        uid = st.text_input(f"Enter {self.db_manager.uid} to update")

        if st.button("Load Info"):
            if not uid:
                raise ValueError(f"Please enter {self.db_manager.uid} to load resident details and update")

            resident_data = self.db_manager.data_manager.load_residents_info_table(filter_ids=[uid])
            str_cols_to_fill = resident_data.select_dtypes(include=["string", "O"]).columns
            resident_data[str_cols_to_fill] = resident_data[str_cols_to_fill].fillna("")
            if resident_data.empty:
                st.warning("No record found for the entered uid")
            else:
                st.session_state.resident_data = resident_data.squeeze()

        if st.session_state.resident_data is not None:
            resident_data = st.session_state.resident_data

            updated_data = {
                "EnrollmentID": st.text_input("Enrollment ID", value=uid, disabled=True),
                "BedID": st.text_input("Select Bed ID", value=resident_data.loc[self.db_manager.bed_id]),
                "DateofAdmission": st.date_input(
                    "Date of Admission",
                    value=resident_data.get("DateofAdmission").date(),
                ),
                "Name": st.text_input("FulName", value=resident_data.get("Name")),
                "DateofBirth": st.date_input("Date of Birth", value=resident_data.get("DateofBirth").date()),
                "FathersName": st.text_input("Father's Full Name", value=resident_data.get("FathersName")),
                "ContactNumber": self.contact_input("Contact Number", value=resident_data.get("ContactNumber")),
                "OtherContact": self.contact_input("Other Contact Number", value=resident_data.get("OtherContact")),
                "FathersContact": self.contact_input("Father's Contact Number", value=resident_data.get("FathersContact")),
                "MothersContact": self.contact_input("Mother's Contact Number", value=resident_data.get("MothersContact")),
                "AdditionalContact": self.contact_input(
                    "Additional Contact Number",
                    value=resident_data.get("AdditionalContact"),
                ),
                "Email": st.text_input("Email ID", value=resident_data.get("Email")),
                "Address": st.text_input("Full Address", value=resident_data.get("Address")),
                "BloodGroup": st.text_input("Blood Group", value=resident_data.get("BloodGroup")),
                "ResidentAadhar": st.text_input("Resident Aadhaar", value=resident_data.get("ResidentAadhar")),
                "FathersAadhar": st.text_input("Father's Aadhaar", value=resident_data.get("FathersAadhar")),
                "InstituteName": st.text_input("Institute Name", value=resident_data.get("InstituteName")),
                "InstituteAddress": st.text_input("Institute Address", value=resident_data.get("InstituteAddress")),
                "Course": st.text_input("Course Name (JEE/NEET/IAS)", value=resident_data.get("Course")),
                "Batch": st.text_input("Batch", value=resident_data.get("Batch")),
                "InstituteContact": self.contact_input("Institute Contact", value=resident_data.get("InstituteContact")),
                "InstituteContactOther": self.contact_input(
                    "Institute Other Contact",
                    value=resident_data.get("InstituteContactOther"),
                ),
                "InstituteID": st.text_input("Institute ID Number", value=resident_data.get("InstituteID")),
                "GuardianName": st.text_input("Name of Guardian", value=resident_data.get("GuardianName")),
                "GuardianRelation": st.text_input(
                    "Relation with Guardian",
                    value=resident_data.get("GuardianRelation"),
                ),
                "GuardianContact1": self.contact_input("Guardian Contact 1", value=resident_data.get("GuardianContact1")),
                "GuardianContact2": self.contact_input("Guardian Contact 2", value=resident_data.get("GuardianContact2")),
                "GuardianAddress": st.text_input("Guardian Address", value=resident_data.get("GuardianAddress")),
                "Rent": st.number_input("Monthly Rent", value=resident_data.get("Rent")),
                "Deposit": st.number_input("Deposit", value=resident_data.get("Deposit")),
                "RentStartDate": st.date_input("Rent Start Date", value=resident_data.get("RentStartDate").date()),
            }

            updated_data_df = pd.DataFrame([updated_data])

            if (updated_data_df["Rent"] < 5000).any() or (updated_data_df["Deposit"] < updated_data_df["Rent"] * 2).any():
                message = "Warning: Either Rent if less than 5,000 or"
                "Deposite is less than twice the rent. Are you sure want to continue ?"
                st.warning(message, icon="⚠️")

            log_comments = st.text_input("Log comments for this Activity")
            if st.button("Update"):
                try:
                    with st.spinner("Processing, please wait"):
                        self.db_manager.data_manager.edit_resident_record(input_df=updated_data_df, log_comments=log_comments)
                    st.success("Resident record successfully updated in the database.")
                    st.session_state.resident_data = None
                except Exception as e:
                    st.error(f"Error Updating the resident record:{e}")
                    st.code(traceback.format_exc())

    def update_electricity_record(self):
        st.header("Change/Update Electricity Info")
        if "prev_record" not in st.session_state:
            st.session_state.prev_record = None

        elect_records = self.db_manager.data_manager.load_electricity_table().sort_values(["Date"])
        date_map = {}
        for a_dt in elect_records["Date"].unique():
            str_dt = pd.to_datetime(a_dt).strftime("%d-%b-%Y %H:%M:%S")
            date_map[str_dt] = a_dt

        update_dt = st.selectbox("Choose the date to update", options=list(date_map.keys()))
        if st.button("Load Info"):
            if not update_dt:
                raise st.warning("Please enter date to load details and update")

            prev_record = elect_records[elect_records["Date"] == date_map[update_dt]].squeeze()
            latest_before_dt = elect_records[elect_records["Date"] < update_dt].tail(1).squeeze()

            if latest_before_dt.empty:
                latest_before_dt = pd.Series(0, index=elect_records.columns)
                latest_before_dt["Date"] = np.nan

            # st.session_state.latest_before_dt = latest_before_dt
            if prev_record.empty:
                st.warning("No record found")
            else:
                st.session_state.prev_record = prev_record.squeeze()

        if st.session_state.prev_record is not None:
            prev_record = st.session_state.prev_record

            updated_data = {"Date": update_dt}
            rooms = pd.Series(self.db_manager.confs.valid_bedIDs).str.replace(r"\D", "", regex=True).unique().tolist()

            for a_room in rooms:
                updated_data[a_room] = st.number_input(f"Enter Room {a_room} Meter Reading", value=prev_record.get(a_room))

            updated_data.update(
                {
                    "Meter_1_2A": st.number_input(
                        "Enter Main Meter 1 Reading",
                        value=prev_record.get("Meter_1_2A"),
                    ),
                    "Meter_2_2B": st.number_input(
                        "Enter Main Meter 2 Reading",
                        value=prev_record.get("Meter_2_2B"),
                    ),
                    "Meter_3_1A": st.number_input(
                        "Enter Main Meter 3 Reading",
                        value=prev_record.get("Meter_3_1A"),
                    ),
                    "Meter_4_1B": st.number_input(
                        "Enter Main Meter 4 Reading",
                        value=prev_record.get("Meter_4_1B"),
                    ),
                    "Meter_5_GA": st.number_input(
                        "Enter Main Meter 5 Reading",
                        value=prev_record.get("Meter_5_GA"),
                    ),
                    "Meter_6_GB": st.number_input(
                        "Enter Main Meter 6 Reading",
                        value=prev_record.get("Meter_6_GB"),
                    ),
                    "Meter_7_Basement": st.number_input(
                        "Enter Main Meter 7 Reading",
                        value=prev_record.get("Meter_7_Basement"),
                    ),
                    "Library": st.number_input("Enter Library Meter Reading", value=prev_record.get("Library")),
                    "Solar": st.number_input("Enter Solar Meter Reading", value=prev_record.get("Solar")),
                }
            )

            updated_data_df = pd.DataFrame([updated_data])

            invalid_reading = updated_data.squeeze() < latest_before_dt
            if invalid_reading.any():
                st.warning(
                    f"Invalid Electricity Reading : Reading for ({invalid_reading[invalid_reading==1].index.tolist()})"
                    "is less than previous reading. cannot enter this record"
                )
                return

            log_comments = st.text_input("Log comments for this Activity")
            if st.button("Update"):
                try:
                    with st.spinner("Processing Admission, please wait"):
                        self.db_manager.data_manager.edit_electricity_record(input_df=updated_data_df, log_comments=log_comments)
                    st.success("Electricity Record updated successfully.")
                    st.session_state.prev_record = None
                except Exception as e:
                    st.error(f"Error Updating electricity record {e}")
                    st.code(traceback.format_exc())

    def record_activity(self):
        st.header("Resident Exit and entry")
        curr_status = self.db_manager.data_manager.load_current_status().reset_index()
        residets_info = self.db_manager.data_manager.load_residents_info_table()
        occupied_beds = self.db_manager.data_manager.get_occupied_beds()
        empty_beds = self.db_manager.data_manager.get_empty_beds()

        trans_type = st.selectbox("choose transaction type", options=["entry", "exit"])
        bed_options = empty_beds if trans_type == "entry" else occupied_beds
        bed_id = st.selectbox("Select BedID", bed_options)

        prev_due = 0
        additional_charges = 0
        rent_thru_dt = datetime.now().date()
        comments = "No Comments"
        error = ""
        last_elect_reading = 0
        if bed_id:
            last_elect_reading = curr_status.loc[curr_status[self.db_manager.bed_id] == bed_id, "RoomElectricityReading"].squeeze()

        if trans_type == "entry":
            uid = st.text_input("Enter Enrollment ID for the entering resident")
            name = ""
            if uid:
                matching_resident = residets_info[residets_info[self.db_manager.uid] == uid]
                if not matching_resident.empty:
                    name = matching_resident["Name"].iloc[0]
                else:
                    st.warning("No resident found with given enrommlent ID")
            name = st.text_input("Resident Name", value=name, disabled=True)
            rent_thru_dt = np.nan

        if trans_type == "exit":
            matching_row = curr_status[curr_status[self.db_manager.bed_id] == bed_id]
            if not matching_row.empty:
                uid = matching_row[self.db_manager.uid].iloc[0]
                name = residets_info.loc[residets_info[self.db_manager.uid] == uid, "Name"].squeeze()
            else:
                uid = None
                name = None
            uid = st.text_input("Enrollment ID of exiting resident", value=uid, disabled=True)
            name = st.text_input("Name of exiting resident", value=name, disabled=True)
            prev_due = st.number_input("Enter any previous rent/electricity due amount.", value=0)
            additional_charges = st.number_input("Enter any additional charges", value=0)
            rent_thru_dt = st.date_input("Charge Rent until", value=datetime.now().date(), format="DD-MM-YYYY")
            rent_thru_dt = rent_thru_dt.strftime("%d-%b-%Y")
            comments = st.text_input("Enter any comments or remarks.", value="No Comments")

        trans_dt = st.date_input("Enter Exit/Entry Date", value=datetime.now().date(), format="DD-MM-YYYY")
        # st.write("Selectd date in custom format:", trans_dt.strftime("%d-%b-%Y"))
        trans_time = st.time_input("Enter Exit/Entry Date")
        trans_dt_time = datetime.combine(trans_dt, trans_time)

        if trans_dt_time > datetime.now():
            error += "Invalid DateTime : cannot accept future date time for transaction date time. \n"

        trans_elect = st.number_input(
            "Enter room electricity reading at the time of exit/entry",
            value=last_elect_reading,
        )
        if trans_elect < last_elect_reading:
            error += f"Invalid Electricity Reading : Current transaction reading ({trans_elect}) "
            "cannot be less than last electricity reading ({last_elect_reading})"

        data = {
            "TransDate": trans_dt_time.strftime("%d-%b-%Y %H:%M:%S"),
            f"{self.db_manager.bed_id}": bed_id,
            f"{self.db_manager.uid}": uid,
            "RoomElectricityReading": trans_elect,
            "TransType": trans_type,
            "PrevDueAmount": prev_due,
            "AdditionalCharges": additional_charges,
            "RentThruDate": rent_thru_dt,
            "Comments": comments,
        }
        data_df = pd.DataFrame([data])
        st.markdown(
            """
            <div style="margin-top:20px; margin-bottom:20px; font-size:24px; font-weight:bold;">
                Please verify all the details carefully before proceeding.
            </div>
            """,
            unsafe_allow_html=True,
        )
        display_df = data_df.copy()

        display_df = display_df.merge(
            residets_info[[self.db_manager.uid, "Name", "Rent", "Deposit"]],
            on=self.db_manager.uid,
            how="left",
        )
        display_df = display_df[
            [
                "TransDate",
                "RentThruDate",
                "BedID",
                "EnrollmentID",
                "Name",
                "Rent",
                "Deposit",
                "TransType",
                "RoomElectricityReading",
                "PrevDueAmount",
                "AdditionalCharges",
                "Comments",
            ]
        ]
        st.dataframe(display_df)

        if error:
            st.warning(error, icon="⚠️")
            return

        log_comments = st.text_input("Log comments for this Activity")
        if st.button("Process Transaction"):
            try:
                with st.spinner("Processing transaction, please wait..."):
                    exit_details = self.db_manager.process_transaction(input_df=data_df, log_comments=log_comments)
                st.success(f"{trans_type} Transaction processed Successfully")
                if exit_details is not None:
                    st.dataframe(exit_details.T)
            except Exception as e:
                st.error(f"Error Updating electricity record {e}")
                st.code(traceback.format_exc())

    def room_transfer(self):
        st.subheader("Room Transfers")

        # Fetch data
        curr_status = self.db_manager.data_manager.load_current_status().reset_index()
        residents_info = self.db_manager.data_manager.load_residents_info_table()
        occupied_beds = self.db_manager.data_manager.get_occupied_beds()
        empty_beds = self.db_manager.data_manager.get_empty_beds()

        # Instructions for the user
        # st.info("If Swapping, select occupied beds for both 'From BedID' and 'To BedID'. "
        #         "If moving to an empty bed, select only empty beds in 'To BedID'.")

        trans_type = st.selectbox("Type of Transfer", ["Moving to Empty Bed", "Swapping"])
        swap = True if trans_type == "Swapping" else False

        # Transfer Date and Time Inputs
        trans_dt = st.date_input("Room Change/Transfer Date", value=datetime.now().date(), format="DD-MM-YYYY")
        trans_time = st.time_input("Enter Change/Transfer Time")
        trans_dt_time = datetime.combine(trans_dt, trans_time)

        error = ""
        # Error checking for future date
        if trans_dt_time > datetime.now():
            error += "Invalid DateTime: Cannot accept future date/time for the transaction."
            st.error("Invalid DateTime: Cannot accept future date/time for the transaction.")

        # Source Bed Information
        st.subheader("Source Bed Information")
        source_bed_id = st.selectbox("Source (From) BedID", occupied_beds)
        prev_src_reading = curr_status.loc[curr_status[self.db_manager.bed_id] == source_bed_id, "RoomElectricityReading"].squeeze()
        source_elect = st.number_input("Source Room Reading", value=prev_src_reading)

        if source_elect < prev_src_reading:
            st.error("Source elect reading cannot be less than prev reading")
            error += "| Source elect reading cannot be less than prev reading"

        source_uid = curr_status.loc[curr_status[self.db_manager.bed_id] == source_bed_id, self.db_manager.uid].squeeze()
        source_name = residents_info.loc[residents_info[self.db_manager.uid] == source_uid, "Name"].squeeze()
        source_rent = residents_info.loc[residents_info[self.db_manager.uid] == source_uid, "Rent"].squeeze()
        source_deposit = residents_info.loc[residents_info[self.db_manager.uid] == source_uid, "Deposit"].squeeze()

        # Disabled input fields for resident information (read-only)
        st.text_input(f"{self.db_manager.uid} of resident in {source_bed_id}", value=source_uid, disabled=True)
        st.text_input(f"Name of resident in {source_bed_id}", value=source_name, disabled=True)
        st.text_input(f"Current Rent of resident {source_name}", value=str(source_rent), disabled=True)
        st.text_input(f"Current Deposit of resident {source_name}", value=str(source_deposit), disabled=True)

        # Financial information adjustments
        new_source_rent = st.number_input(f"Enter 'New' Rent for {source_name}", value=0)
        new_source_deposit = st.number_input(f"Enter 'New' Deposit for {source_name}", value=0)

        st.warning(f"Change in Rent -> {new_source_rent - source_rent:,.1f}")
        st.warning(f"Change in Deposit -> {new_source_deposit - source_deposit:,.1f}")

        # Destination Bed Information
        st.subheader("Destination Bed Information")
        desti_bed_options = occupied_beds if swap else empty_beds
        desti_bed_id = st.selectbox("Destination (To) BedID", desti_bed_options)

        prev_desti_reading = curr_status.loc[curr_status[self.db_manager.bed_id] == desti_bed_id, "RoomElectricityReading"].squeeze()
        desti_elect = st.number_input("Destination Room Reading", value=prev_desti_reading)

        if desti_elect < prev_desti_reading:
            st.error("Destination elect reading cannot be less than prev reading")
            error += "| Destination elect reading cannot be less than prev reading"

        # If swapping, show the resident info for the destination bed
        desti_uid, desti_rent, desti_deposit = np.nan, np.nan, np.nan
        new_desti_rent, new_desti_deposit = np.nan, np.nan
        if swap:
            desti_uid = curr_status.loc[curr_status[self.db_manager.bed_id] == desti_bed_id, self.db_manager.uid].squeeze()
            desti_name = residents_info.loc[residents_info[self.db_manager.uid] == desti_uid, "Name"].squeeze()
            desti_rent = residents_info.loc[residents_info[self.db_manager.uid] == desti_uid, "Rent"].squeeze()
            desti_deposit = residents_info.loc[residents_info[self.db_manager.uid] == desti_uid, "Deposit"].squeeze()

            st.text_input(f"{self.db_manager.uid} of resident in {desti_bed_id}", value=desti_uid, disabled=True, key="1")
            st.text_input(f"Name of resident in {desti_bed_id}", value=desti_name, disabled=True, key="2")
            st.text_input(f"Current Rent of resident {desti_name}", value=str(desti_rent), disabled=True, key="3")
            st.text_input(f"Current Deposit of resident {desti_name}", value=str(desti_deposit), disabled=True, key="4")

            # Financial adjustment for destination resident
            new_desti_rent = st.number_input(f"Enter 'New' Rent for {desti_name}", value=0, key="5")
            new_desti_deposit = st.number_input(f"Enter 'New' Deposit for {desti_name}", value=0, key="6")

            st.warning(f"Change in Rent -> {new_desti_rent - desti_rent:,.1f}")
            st.warning(f"Change in Deposit -> {new_desti_deposit - desti_deposit:,.1f}")

        comments = st.text_input("Comments")
        data = {
            "TransDate": trans_dt_time,
            "TransType": trans_type,
            "SourceBedId": source_bed_id,
            "SourceEnrollmentID": source_uid,
            "SourceElectReading": source_elect,
            "SourceResidentOldRent": source_rent,
            "SourceResidentNewRent": new_source_rent,
            "SourceResidentOldDeposit": source_deposit,
            "SourceResidentNewDeposit": new_source_deposit,
            "DestinationBedId": desti_bed_id,
            "DestinationEnrollmentID": desti_uid,
            "DestinationElectReading": desti_elect,
            "DestinationResidentOldRent": desti_rent,
            "DestinationResidentNewRent": new_desti_rent,
            "DestinationResidentOldDeposit": desti_deposit,
            "DestinationResidentNewDeposit": new_desti_deposit,
            "Comments": comments,
        }

        if not error:
            if st.button("Process Transfer"):
                try:
                    with st.spinner("Processing transfer, please wait..."):
                        # pd.DataFrame([data]).to_csv("temp_room_trnsfer.csv", index=False)
                        self.db_manager.process_room_transfers(input_df=pd.DataFrame([data]))
                    st.success("Transfer successfully processed")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

        return

    def show_final_adjustment(self, exit_details):
        if not exit_details.empty:
            st.subheader("Final Adjustment")
            for col in exit_details.columns:
                st.write(f"{col}: {exit_details[col].squeeze()}")

    def view_current_tables(self):
        self.apply_custom_css()

        st.header("View DB and tables")

        cursor = self.db_manager.data_manager.db_handler.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [tbl[0] for tbl in cursor.fetchall()]
        df = None

        table_name = st.selectbox("Choose Table name to load", options=tables)

        if table_name in [self.db_manager.confs.current_status_tbl]:
            st.subheader(f"Empty Bed Count : {len(self.db_manager.data_manager.get_empty_beds())}")

        access_code = "##rj_14_hs_3722"
        if table_name in [self.db_manager.confs.rent_history_tbl]:
            pass_ = st.text_input("Enter the password to load table", type="password")
            if pass_:
                if pass_ != access_code:
                    st.error("Incorrect Password. Access Denied")
                    return
                else:
                    st.success("Access Granted")

        residents_info = self.db_manager.data_manager.load_residents_info_table()

        match table_name:
            case self.db_manager.confs.residents_tbl:
                df = self.db_manager.data_manager.load_residents_info_table()
                df[self.db_manager.confs.date_cols_residents_tbl] = df[self.db_manager.confs.date_cols_residents_tbl].apply(
                    lambda x: x.dt.strftime("%d-%b-%Y")
                )
                df[self.db_manager.room_id] = np.where(
                    df[self.db_manager.room_id].isna(),
                    np.nan,
                    df[self.db_manager.room_id].astype(int),
                )

            case self.db_manager.confs.electricity_tbl:
                df = self.db_manager.data_manager.load_electricity_table()
                df[self.db_manager.confs.date_cols_electricity_tbl] = df[self.db_manager.confs.date_cols_electricity_tbl].apply(
                    lambda x: x.dt.strftime("%d-%b-%Y %H:%M:%S")
                )
                df = df.T

            case self.db_manager.confs.transactions_tbl:
                df = self.db_manager.data_manager.load_transactions_table()
                df = df.merge(
                    residents_info[["EnrollmentID", "Name"]],
                    on=self.db_manager.uid,
                    how="left",
                )

                df[self.db_manager.confs.date_cols_transactions_tbl] = df[self.db_manager.confs.date_cols_transactions_tbl].apply(
                    lambda x: x.dt.strftime("%d-%b-%Y")
                )

            case self.db_manager.confs.final_settlement_tbl:
                df = self.db_manager.data_manager.load_final_settlement_table()
                df[self.db_manager.confs.date_cols_final_settlement_tbl] = df[self.db_manager.confs.date_cols_final_settlement_tbl].apply(
                    lambda x: x.dt.strftime("%d-%b-%Y")
                )

            case self.db_manager.confs.rent_history_tbl:
                df = self.db_manager.data_manager.load_rent_history()
                df[self.db_manager.confs.date_cols_rent_history_tbl] = df[self.db_manager.confs.date_cols_rent_history_tbl].apply(
                    lambda x: x.dt.strftime("%d-%b-%Y")
                )

            case self.db_manager.confs.current_status_tbl:
                df = self.db_manager.data_manager.load_current_status()
                df = df.merge(
                    residents_info[
                        [
                            "EnrollmentID",
                            "Name",
                            "ContactNumber",
                            "OtherContact",
                            "FathersContact",
                            "MothersContact",
                            "Course",
                        ]
                    ],
                    on=self.db_manager.uid,
                    how="left",
                )

                df[self.db_manager.confs.date_cols_status_tbl] = df[self.db_manager.confs.date_cols_status_tbl].apply(
                    lambda x: x.dt.strftime("%d-%b-%Y")
                )
                df[self.db_manager.room_id] = np.where(
                    df[self.db_manager.room_id].isna(),
                    np.nan,
                    df[self.db_manager.room_id].astype(int),
                )

            case self.db_manager.confs.logs_tbl:
                df = self.db_manager.data_manager.load_logs()

        if (df is not None) and (not df.empty):
            st.dataframe(df, height=1100, use_container_width=True)

    def apply_custom_css(self):
        st.markdown(
            """
            <style>
                .main .block-container {
                    max-width: 75%;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def change_electricity_meter(self):
        st.subheader("Electricity Meter Change Entry")
        curr_status = self.db_manager.data_manager.load_current_status()
        error = ""

        meter_options = curr_status[self.db_manager.room_id].unique()
        room = st.selectbox("Choose Room Meter to change", options=meter_options)
        prev_reading = curr_status.loc[curr_status[self.db_manager.room_id] == room, "RoomElectricityReading"].iloc[0]

        trans_dt = st.date_input("Enter meter change  Date", value=datetime.now().date(), format="DD-MM-YYYY")
        trans_time = st.time_input("Enter meter change time")
        trans_dt_time = datetime.combine(trans_dt, trans_time)

        if trans_dt_time > datetime.now():
            error += "\n Invalid Datetime : date time cannot be later than current time"

        old_meter_reading = st.number_input("Enter Old Meter electricity reading", value=prev_reading)
        if prev_reading > old_meter_reading:
            error += "\n Invalid Reading. Old reading cannot be less than previous reading of the meter"

        new_meter_reading = st.number_input("Enter New Meter Starting electricity reading", value=0)

        if error:
            st.warning(error, icon="⚠️")
            return

        log_comments = st.text_input("Log comments for this Activity")
        if st.button("Submit"):
            self.db_manager.room_meter_change(
                room=room,
                trans_dt_time=trans_dt_time,
                old_meter_reading=old_meter_reading,
                new_meter_reading=new_meter_reading,
                log_comments=log_comments,
            )
            st.success("Meter changed successfully.")

        return

    def calculate_monthly_rent(self):

        st.subheader("Calculate Monthly Rent")

        curr_dt = datetime.now().date()
        prev_eom = curr_dt.replace(day=1) - dt.timedelta(days=1)
        eom = (pd.Timestamp(curr_dt).to_period("M").end_time).date()

        prev_eom_diff = (curr_dt - prev_eom).days
        eom_diff = (eom - curr_dt).days

        closer_dt = prev_eom if (prev_eom_diff < eom_diff) else eom

        dt_input = st.date_input("Enter month End Date", value=closer_dt, format="DD-MM-YYYY")
        st.warning("Choose only the month end date like 31Oct, 28Feb, 30Jun", icon="⚠️")
        update_and_save = st.selectbox("Update Database and Status ?", options=["No", "Yes"])
        update_and_save = True if update_and_save == "Yes" else False

        if st.button("Calculate"):
            try:
                out = self.db_manager.calculate_rent(dt_input, update_and_save)
                st.success("Rent Calculation Successfull")
                st.dataframe(out)

            except Exception as e:
                st.error(f"Error Updating electricity record {e}")
                st.code(traceback.format_exc())

    def record_payment(self):
        st.subheader("Record Payment")
        curr_status = self.db_manager.data_manager.load_current_status()
        resident_info = self.db_manager.data_manager.load_residents_info_table()
        curr_status = curr_status.merge(resident_info[[self.db_manager.uid, "Name"]], on=self.db_manager.uid, how="left")
        curr_status["options"] = curr_status[self.db_manager.bed_id] + " - " + curr_status["Name"]

        error = ""

        resident_options = curr_status[curr_status["options"].notna()]["options"].values
        resident = st.selectbox("Choose Resident", resident_options)
        resident_uid = curr_status.loc[curr_status["options"] == resident, self.db_manager.uid].squeeze()

        trans_dt = st.date_input("Enter Payment Date", value=datetime.now().date(), format="DD-MM-YYYY")
        trans_time = st.time_input("Enter Payment Time")
        trans_dt_time = datetime.combine(trans_dt, trans_time)

        if trans_dt_time > datetime.now():
            st.error(" Invalid Datetime : date time cannot be later than current time")
            error += " Invalid Datetime : date time cannot be later than current time"

        if pd.isna(resident):
            st.error("Invalid Resident Chosen.")
            error += "Invalid resident chosen"

        prev_due = curr_status.loc[curr_status[self.db_manager.uid] == resident_uid, "PrevDueAmount"].squeeze()
        prev_due = st.text_input("Previous Due", value=str(prev_due), disabled=True)

        pay_amt = st.number_input("Enter Payment Amount")

        log_comments = st.text_input("Comments")

        data = {
            f"{self.db_manager.uid}": resident_uid,
            f"{self.db_manager.bed_id}": curr_status.loc[curr_status[self.db_manager.uid] == resident_uid, f"{self.db_manager.bed_id}"].squeeze(),
            f"{self.db_manager.room_id}": curr_status.loc[curr_status[self.db_manager.uid] == resident_uid, f"{self.db_manager.room_id}"].squeeze(),
            "TransDate": trans_dt_time,
            "PaymentAmount": pay_amt,
            "Comments": log_comments,
        }
        if not error:
            if st.button("Process Payment"):
                try:
                    with st.spinner("Processing, please wait"):
                        self.db_manager.record_payment(row=pd.Series(data), log_comments=log_comments)
                    st.success("Payment processed Successfully")
                except Exception as e:
                    st.error(f"Error Updating electricity record {e}")
                    st.code(traceback.format_exc())

    def create_copy(self):
        st.subheader("Create a copy of the current databse in use.")
        if st.button("Create a copy"):
            with st.spinner("Processing, please wait"):
                self.db_manager.db_handler.copy_and_refresh_db()
            st.success("Successfully created a copy of the databse. Please refresh")


# Main Streamlit UI
def main():
    st.title("Residence Management System")

    db_handler = DatabaseHandler(confs=configs)
    data_manager = DataManager(confs=configs, db_handler=db_handler)
    db_manager = RDM.ResidentManager(confs=configs, db_handler=db_handler, data_manager=data_manager)  # Replace this with your actual db manager
    system = ResidenceManagementStreamlit(db_manager)

    # Sidebar for navigation
    menu = [
        "Save a Copy",
        "New Admission",
        "New Electricity Reading",
        "Update Resident Info",
        "Update Electricity Record",
        "Entry/Exit of Form",
        "Room Transfer",
        "View Current Tables",
        "Calculate Rent",
        "Record Payment",
        "Electricity Meter Change",
        "Undo Last change",
        "View Current Tables",
    ]
    choice = st.sidebar.radio("Choose an Option", menu)

    st.write(f"Curretn Databse in Use -> {db_handler.get_latest_db_path(full_path=False)}")

    if choice == "New Admission":
        system.new_admission()
    elif choice == "Save a Copy":
        system.create_copy()
    elif choice == "New Electricity Reading":
        system.new_electricity_record()
    elif choice == "Update Resident Info":
        system.update_resident_info()
    elif choice == "Update Electricity Record":
        system.update_electricity_record()
    elif choice == "Entry/Exit of Form":
        system.record_activity()
    elif choice == "Room Transfer":
        system.room_transfer()
    elif choice == "View Current Tables":
        system.view_current_tables()
    elif choice == "Calculate Rent":
        system.calculate_monthly_rent()
    elif choice == "Record Payment":
        system.record_payment()

    elif choice == "Electricity Meter Change":
        system.change_electricity_meter()

    if choice == "Undo Last change":
        st.subheader("Undo Recent Changes")
        st.write("This will revert the database to the last backup available.")

        files_list = os.listdir(str(db_manager.db_handler.db_path))
        db_list = [
            file.strip(db_manager.db_handler.db_extension)
            for file in files_list
            if (file.endswith(db_manager.db_handler.db_extension) and file.startswith(db_manager.db_handler.db_filename))
        ]

        st.write(db_list)
        if "undo_warning_displayed" not in st.session_state:
            st.session_state.undo_warning_displayed = False

        if st.button("Undo Recent Changes"):
            st.session_state.undo_warning_displayed = True

        if st.session_state.undo_warning_displayed:
            st.warning("Are you sure you want to undo recent changes? This action cannot be undone.")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Confirm Undo"):
                    try:
                        if db_handler.connection:
                            db_handler.close()
                        # Revert database to last backup
                        db_manager.data_manager.revert_to_last_backup()
                        st.success("Reverted to the last available backup successfully.")
                        st.session_state.undo_warning_displayed = False
                    except Exception as e:
                        st.error(f"Failed to undo changes: {e}")

            with col2:
                if st.button("Cancel"):
                    st.session_state.undo_warning_displayed = False


if __name__ == "__main__":
    main()
