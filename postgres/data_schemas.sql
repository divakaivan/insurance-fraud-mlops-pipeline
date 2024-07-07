CREATE TABLE meaningful_features (
    "NumberOfSuppliments" VARCHAR(255),
    "AgeOfVehicle" VARCHAR(255),
    "AgeOfPolicyHolder" VARCHAR(255),
    "Month" VARCHAR(255),
    "Deductible" INT,
    "MonthClaimed" VARCHAR(255),
    "Make" VARCHAR(255),
    "AddressChange_Claim" VARCHAR(255),
    "PastNumberOfClaims" VARCHAR(255),
    "VehiclePrice" VARCHAR(255),
    "VehicleCategory" VARCHAR(255),
    "Fault" VARCHAR(255),
    "FraudFound_P" INT,
    "FraudFound_P_Prob" INT DEFAULT 2
);

CREATE TABLE model_data_w_dummy (
    "NumberOfSuppliments_none_or_1_to_2" INT,
    "AgeOfVehicle_3_4_years" INT,
    "AgeOfVehicle_more_than_5_years" INT,
    "AgeOfVehicle_new" INT,
    "AgeOfPolicyHolder_18_to_20" INT,
    "AgeOfPolicyHolder_21_to_25" INT,
    "AgeOfPolicyHolder_26_to_30" INT,
    "AgeOfPolicyHolder_31_to_35" INT,
    "AgeOfPolicyHolder_36_to_40" INT,
    "AgeOfPolicyHolder_41_to_65" INT,
    "AgeOfPolicyHolder_over_65" INT,
    "Month_Aug" INT,
    "Month_Dec" INT,
    "Month_Jan_Feb" INT,
    "Month_Jul" INT,
    "Month_Jun" INT,
    "Month_Mar" INT,
    "Month_May" INT,
    "Month_Nov" INT,
    "Month_Oct" INT,
    "Month_Sep" INT,
    "Deductible_400" INT,
    "Deductible_500" INT,
    "Deductible_700" INT,
    "MonthClaimed_Apr" INT,
    "MonthClaimed_Aug" INT,
    "MonthClaimed_Dec" INT,
    "MonthClaimed_Jan_Feb" INT,
    "MonthClaimed_Jul" INT,
    "MonthClaimed_Jun" INT,
    "MonthClaimed_Mar" INT,
    "MonthClaimed_May" INT,
    "MonthClaimed_Nov" INT,
    "MonthClaimed_Oct" INT,
    "MonthClaimed_Sep" INT,
    "Make_BMW" INT,
    "Make_Chevrolet" INT,
    "Make_Dodge" INT,
    "Make_Ford" INT,
    "Make_Honda" INT,
    "Make_Lexus_Ferrari_Porche_Jaguar" INT,
    "Make_Mazda" INT,
    "Make_Mercedes" INT,
    "Make_Mercury" INT,
    "Make_Nisson" INT,
    "Make_Pontiac" INT,
    "Make_Saab" INT,
    "Make_Saturn" INT,
    "Make_Toyota" INT,
    "Make_VW" INT,
    "AddressChange_Claim_2_to_3_years" INT,
    "AddressChange_Claim_4_to_8_years" INT,
    "AddressChange_Claim_no_change" INT,
    "AddressChange_Claim_under_6_months" INT,
    "PastNumberOfClaims_2_to_4" INT,
    "PastNumberOfClaims_more_than_4" INT,
    "PastNumberOfClaims_none" INT,
    "VehiclePrice_40000_to_59000" INT,
    "VehiclePrice_60000_to_69000" INT,
    "VehiclePrice_less_than_20000" INT,
    "VehiclePrice_more_than_69000" INT,
    "VehicleCategory_Sport" INT,
    "VehicleCategory_Utility" INT,
    "Fault_Third_Party" INT,
    "FraudFound_P" INT,
    "FraudFound_P_Prob" INT DEFAULT 2
);
