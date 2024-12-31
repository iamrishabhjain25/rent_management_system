import os
import pathlib

# db_path = pathlib.Path(os.getcwd()).resolve() / "store"

db_path = pathlib.Path(os.getcwd()).resolve().parent / "store"
db_filename: str = "ResidentsDatabase"
db_extension: str = ".db"
fl_name_dt_frmt: str = "%Y-%m-%d_%H-%M-%S"

credentials_path = pathlib.Path(os.getcwd()).resolve().parent.parent.parent / "credentials" / "credentials.json"


# UID format: YYYY001
uid_length = 7

# tables names

residents_tbl = "residents_info"
electricity_tbl = "electricity_readings"
current_status_tbl = "status"
transactions_tbl = "transactions"
last_adjustments_tbl = "last_adjustments"
final_settlement_tbl = "final_settlement"
rent_history_tbl = "rent_history"
logs_tbl = "logs"


date_cols_status_tbl = ["LastRentCalcDate", "LastElectricityCalcDate", "TransDate"]
date_cols_residents_tbl = ["DateofAdmission", "DateofBirth", "RentStartDate"]
date_cols_electricity_tbl = ["Date"]
date_cols_transactions_tbl = ["TransDate", "RentThruDate"]
date_cols_final_settlement_tbl = [
    "ExitDate",
    "LastRentCalcDate",
    "LastElectricityCalcDate",
]
date_cols_rent_history_tbl = [
    "TransDate",
    "PrevElectricityCalcDate",
    "LastElectricityCalcDate",
    "PrevRentCalcDate",
    "LastRentCalcDate",
]
date_cols_logs_tbl = ["Date"]


float_cols_residents_tbl = ["Rent", "Deposit"]
float_cols_status_tbl = ["RoomElectricityReading", "CumulativeElectConsumption", "PrevDueAmount", "AdditionalCharges"]
float_cols_electricity_tbl = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "110",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "201",
    "202",
    "203",
    "204",
    "205",
    "206",
    "207",
    "208",
    "209",
    "210",
    "211",
    "212",
    "213",
    "214",
    "215",
    "216",
    "217",
    "Meter_1_2A",
    "Meter_2_2B",
    "Meter_3_1A",
    "Meter_4_1B",
    "Meter_5_GA",
    "Meter_6_GB",
    "Meter_7_Basement",
    "Library",
    "Solar",
]

float_cols_transactions_tbl = [
    "RoomElectricityReading",
    "PrevDueAmount",
    "AdditionalCharges",
    "TransactionAmount",
    "Rent",
    "Deposit",
    "NewBedIDElectReading",
    "NewRent",
    "NewDeposit",
]

float_cols_final_settlement_tbl = [
    "Rent",
    "RentDue",
    "PrevRoomElectricityReading",
    "ExitElectricityReading",
    "RoomElectricityConsumption",
    "CumulativeElectConsumption",
    "ElectricityCharges",
    "PrevDueAmount",
    "AdditionalCharges",
    "TotalAmountDue",
    "Deposit",
    "NetAmountDue"



]


uid_col = "EnrollmentID"
room_col = "RoomNo"
bedId_col = "BedID"


valid_bedIDs = [
    "1A",
    "1B",
    "2A",
    "2B",
    "3A",
    "3B",
    "3C",
    "4A",
    "4B",
    "5A",
    "5B",
    "5C",
    "6A",
    "6B",
    "7A",
    "7B",
    "8A",
    "8B",
    "9A",
    "9B",
    "9C",
    "10A",
    "10B",
    "10C",
    "11A",
    "11B",
    "101A",
    "101B",
    "102A",
    "102B",
    "103A",
    "103B",
    "104A",
    "104B",
    "105A",
    "105B",
    "106A",
    "106B",
    "107A",
    "107B",
    "108A",
    "108B",
    "109A",
    "109B",
    "109C",
    "110A",
    "110B",
    "111A",
    "111B",
    "112A",
    "112B",
    "113A",
    "113B",
    "114A",
    "114B",
    "115A",
    "115B",
    "116A",
    "116B",
    "117A",
    "117B",
    "201A",
    "201B",
    "202A",
    "202B",
    "203A",
    "203B",
    "204A",
    "204B",
    "205A",
    "205B",
    "206A",
    "206B",
    "207A",
    "207B",
    "208A",
    "208B",
    "209A",
    "209B",
    "209C",
    "210A",
    "210B",
    "211A",
    "211B",
    "212A",
    "212B",
    "213A",
    "213B",
    "214A",
    "214B",
    "215A",
    "215B",
    "216A",
    "216B",
    "217A",
    "217B",
]

valid_trans_types = ["entry", "exit"]
triple_rooms = ["3", "5", "9", "10", "109", "209"]

elect_rate = 11.0
