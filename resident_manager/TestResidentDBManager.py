class TestResidentDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Create an instance of the ResidentDatabaseManager
        self.db_manager = ResidentDatabaseManager(path="./test_store/")  # Use a test directory

    def test_invalid_date_format(self):
        # Create a DataFrame with an invalid date format
        invalid_data = pd.DataFrame(
            {
                "EnrollmentID": [1],
                "DateofAdmission": ["2023-02-30"],  # Invalid date
                "DateofBirth": ["2000-12-31"],
                "RentStartDate": ["2023-01-01"],
            }
        )

        with self.assertRaises(ValueError):
            self.db_manager._validate_data(
                invalid_data,
                pd.DataFrame(),
                "EnrollmentID",
                date_cols=["DateofAdmission", "DateofBirth", "RentStartDate"],
                int_cols=[],
                float_cols=[],
                str_cols=[],
            )

    def test_missing_dates(self):
        # Missing DateofAdmission in the input data
        missing_date_data = pd.DataFrame(
            {
                "EnrollmentID": [2],
                "DateofBirth": ["2001-05-10"],
                "RentStartDate": ["2023-05-01"],
            }
        )

        with self.assertRaises(KeyError):
            self.db_manager._validate_data(
                missing_date_data,
                pd.DataFrame(),
                "EnrollmentID",
                date_cols=["DateofAdmission", "DateofBirth", "RentStartDate"],
                int_cols=[],
                float_cols=[],
                str_cols=[],
            )

    def test_future_date(self):
        # Insert a future date in RentStartDate
        future_data = pd.DataFrame(
            {
                "EnrollmentID": [3],
                "DateofAdmission": ["2023-01-01"],
                "DateofBirth": ["1999-12-31"],
                "RentStartDate": ["2050-12-01"],  # Future date
            }
        )

        validated_data = self.db_manager._validate_data(
            future_data,
            pd.DataFrame(),
            "EnrollmentID",
            date_cols=["DateofAdmission", "DateofBirth", "RentStartDate"],
            int_cols=[],
            float_cols=[],
            str_cols=[],
        )

        self.assertEqual(validated_data["RentStartDate"].iloc[0], pd.to_datetime("2050-12-01"))

    def test_duplicate_entries(self):
        data_with_duplicates = pd.DataFrame(
            {
                "EnrollmentID": [4, 4],
                "DateofAdmission": ["2023-01-01", "2023-01-01"],
                "DateofBirth": ["1999-12-31", "1999-12-31"],
                "RentStartDate": ["2023-01-01", "2023-01-01"],
            }
        )

        with self.assertRaises(ValueError):  # Should raise an error due to duplicate EnrollmentID
            self.db_manager._validate_data(
                data_with_duplicates,
                pd.DataFrame(),
                "EnrollmentID",
                date_cols=["DateofAdmission", "DateofBirth", "RentStartDate"],
                int_cols=[],
                float_cols=[],
                str_cols=[],
            )

    def test_empty_dataset(self):
        empty_data = pd.DataFrame(columns=["EnrollmentID", "DateofAdmission", "DateofBirth", "RentStartDate"])

        # This should pass without inserting anything into the database
        validated_data = self.db_manager._validate_data(
            empty_data,
            pd.DataFrame(),
            "EnrollmentID",
            date_cols=["DateofAdmission", "DateofBirth", "RentStartDate"],
            int_cols=[],
            float_cols=[],
            str_cols=[],
        )
        self.assertTrue(validated_data.empty)

    def test_invalid_data_types(self):
        invalid_data = pd.DataFrame(
            {
                "EnrollmentID": ["five"],  # String instead of integer
                "DateofAdmission": ["2023-02-15"],
                "DateofBirth": ["2000-01-01"],
                "RentStartDate": ["2023-03-01"],
            }
        )

        with self.assertRaises(ValueError):
            self.db_manager._validate_data(
                invalid_data,
                pd.DataFrame(),
                "EnrollmentID",
                date_cols=["DateofAdmission", "DateofBirth", "RentStartDate"],
                int_cols=["EnrollmentID"],
                float_cols=[],
                str_cols=[],
            )

    def test_extreme_dates(self):
        extreme_dates_data = pd.DataFrame(
            {
                "EnrollmentID": [5],
                "DateofAdmission": ["1900-01-01"],  # Very old date
                "DateofBirth": ["1999-01-01"],
                "RentStartDate": ["2050-01-01"],  # Far future date
            }
        )

        validated_data = self.db_manager._validate_data(
            extreme_dates_data,
            pd.DataFrame(),
            "EnrollmentID",
            date_cols=["DateofAdmission", "DateofBirth", "RentStartDate"],
            int_cols=[],
            float_cols=[],
            str_cols=[],
        )

        self.assertEqual(validated_data["DateofAdmission"].iloc[0], pd.to_datetime("1900-01-01"))
        self.assertEqual(validated_data["RentStartDate"].iloc[0], pd.to_datetime("2050-01-01"))
