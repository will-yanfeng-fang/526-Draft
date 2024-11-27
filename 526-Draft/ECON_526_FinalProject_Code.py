"""
ECON 526  - Final Project Python File
Date: December 15, 2024

Collaborated By: Yanfeng Fang, Jiazhe li
Student Number: 52583069, 26337410

Last Edit: Nov 25, 2024

Issues:
1. Maybe change it to fuzzy RDD if I keep the research of 2CP. If OCP, that RDD will be kept sharp
2. Double check the bandwidth to be optimal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns #if error, pip install seaborn in terminal
import statsmodels.formula.api as smf
import os

runDataCleaning = False # change to True to re-generate the labeled dataset
runDataCleaning_NA = False # Second Stage Cleaning, dropping some columns, and then blank cells
runRegression = True # change to True to run regression
runTCP = True # TCP as a placebo
runRobust = False
runML = False # change to True to run machine learning 

### Change 'runDataCleaning' to True to run this part ###
if runDataCleaning:
    # import data set from dta file 
    filename = "dataset_raw_overall.dta" #Changed for more variables. It was controls&othervars_citylevel_CCSY&RECSY.dta
    if not os.path.exists(filename) :
        print(filename + " not found")
        print("Please download the data from the link above.")

    df1 = pd.read_stata(filename)
    data = pd.DataFrame(df1)

    # import city list labeled by geographic location
    filename2 = "CityList_Label.xlsx"
    df2 = pd.read_excel(filename2) # 1 indicates have nearby treated city, 0 means not
    CityList = pd.DataFrame(df2)

    # import divorce and marriange data 
    filename3 = "divorce&marriage_citylevel_CCASY.dta"
    df3 = pd.read_stata(filename3)
    marriage_Data = pd.DataFrame(df3)
    
    # Merge Divorce Data into Data Frame
    for i in range(len(marriage_Data['city_code'])):
        for t in range(len(data['city_code'])):
            if marriage_Data.at[i,'city_code'] == data.at[t,'city_code'] and marriage_Data.at[i,'year'] == data.at[t,'year']:
                data.at[t,'dvrce_cpl'] = marriage_Data.at[i,'dvrce_cpl'] # Number of couples registered for divorce
                data.at[t,'mrgecpl_rstor'] = marriage_Data.at[i,'mrgecpl_rstor'] # Number of couples registered for marriage restoration
                data.at[t,'mrge_cpl'] = marriage_Data.at[i,'mrge_cpl'] # Number of couples registered for marriage
                data.at[t,'mrgepop_re'] = marriage_Data.at[i,'mrgepop_re'] # Number of people registered for remarriage

    # merge the lable of cities into data frame
    for i in range(len(CityList['city_code'])):
        for t in range(len(data['city_name'])):
            # check if city name matches 
            if CityList.at[i,'city_code'] == data.at[t,'city_name']:
                data.at[t,'haveNearCity'] = CityList.at[i,'Treated'] # Whether it has a border city of another province
                data.at[t,'dis_to_bor'] = CityList.at[i,'dis_to_bor'] # The distance to the nearest city of another province
                data.at[t,'OCP_pref'] = CityList.at[i,'OCP_pref'] # Whether the city is in the preferential OCP province
    
    # Fill in all null value with 0, Not Working right now, idk why :(
    for i in range(len(data['OCP_pref'])):
        if data.at[i,'OCP_pref'] == None:
            data.at[i,'OCP_pref'] = 0    
    
    # Get birth rate and append to dataset 
    data['birth_Rate'] = data['birth_pop'] / data['tot_pop'] 
                
    # Export to .xlsx file for future use
    data.to_excel('Dataset_labeled.xlsx', index=False, sheet_name='Dataset') 

    """
    Now we will use the new file createed above to increase effeciency and reduce running time of the code
    """
    df4 = pd.read_excel('Dataset_labeled.xlsx')
    data = pd.DataFrame(df4)

    data_filtered = data[(data['haveNearCity'] == 1)] # filter effective observation
    data.to_excel('df4_data_filtered.xlsx', index=False, sheet_name='Dataset')
    data_filtered.to_excel('df4_data_filtered.xlsx', index=False, sheet_name='Dataset')
    data_filtered_outlier = data_filtered[abs(data_filtered['dis_to_bor'])<=400]

    # I take the natural log of the dis_to_bor
    # Split the data into two DataFrames
    positive_data = data_filtered_outlier[data_filtered_outlier['dis_to_bor'] > 0]
    negative_data = data_filtered_outlier[data_filtered_outlier['dis_to_bor'] < 0]
    # Apply log transformation to the positive values
    positive_data['dis_log'] = np.log(positive_data['dis_to_bor'])  # min pos = 4.44
    # I take the absolute of negative value, then log, then *-1
    negative_data['dis_log'] = np.log(np.abs(negative_data['dis_to_bor']))
    negative_data['dis_log'] = negative_data['dis_log']*(-1)  # max neg = -4.66 

    # I put all the logged data together, name it dis_scaled
    data_filtered_outlier = pd.concat([positive_data, negative_data])
    data_filtered_outlier['dvrcecs_setl_prov_pc'] = data_filtered_outlier['dvrcecs_setl_prov'] / data_filtered_outlier['tot_pop'] 
    data_filtered_outlier.to_excel('Dataset_labeled.xlsx', index=False, sheet_name='Dataset') 

# Second Stage of Data Cleaning, where we should delete the "not-useful" column, and then, drop the empty cells
if runDataCleaning_NA:
    input_file = "Dataset_labeled.xlsx"
    output_file = "Dataset_labeled_dropNA.xlsx"

    # Specify the columns to keep
    columns_to_keep = [
        "year", "prov_code", "city_code", "gdp", "gdppc", "scnd_gdp", "fdi", 
        "cnsmptn_tnpc", "incm_tnpc", "female_hj_aft12", "male_hj_aft12", 
        "emply_org", "emply_tnprvt", "unemply_tn", "cpi_lstyr", "cpi_04", 
        "dvrcecs_setl_prov", "dvrcejdge_no_prov", "dvrcert_rgh", "rstorrt_rgh", 
        "mrgert_rgh", "dvrcert_rgh_robs", "rstorrt_rgh_robs", "ln_rlgdppc", 
        "open", "ln_avghsprice", "ln_cnsmptn", "sex_rto_hjcz", "hsprice_pegrth", 
        "hs_land", "mrge_cpl", "ln_birth", "frtlty_rt", "ln_emply", "emply_rt", 
        "hsprice_yrgrth", "dis_to_bor", "dis_log", "tot_pop", "dvrcecs_setl_prov_pc"
    ]

    # Load the dataset and select specified columns
    data = pd.read_excel(input_file)
    selected_data = data[columns_to_keep]

    # Drop rows with missing values
    cleaned_data = selected_data.dropna()

    # Export the cleaned dataset to a new Excel file
    cleaned_data.to_excel(output_file, index=False)

    print(f"Cleaned dataset saved to {output_file}")

# set up pannel data regression, change 'Regression' to True to run this part
if runRegression:
    # Load the data again so no need to run previous parts, same logic
    data = pd.read_excel('Dataset_labeled_dropNA.xlsx')
    rdd_data = data#[(data['year'].isin([2016]))] # If analysing for specific year for analysis

    # Regression Discontinuity Design
    rdd_data = rdd_data.assign(threshold=(rdd_data["dis_to_bor"] > 0).astype(int))

    # We choose multiple dependent variables
    # significant: scnd_grp
    dependent_vars = [
        "gdp", "gdppc", "scnd_gdp", "fdi", "cnsmptn_tnpc","incm_tnpc","female_hj_aft12","male_hj_aft12",
        "emply_org","emply_tnprvt","unemply_tn","cpi_lstyr","cpi_04","dvrcecs_setl_prov","dvrcejdge_no_prov",
        "dvrcert_rgh","rstorrt_rgh","mrgert_rgh","dvrcert_rgh_robs","rstorrt_rgh_robs","ln_rlgdppc","open",
        "ln_avghsprice","ln_cnsmptn","sex_rto_hjcz","hsprice_pegrth","hs_land", "mrge_cpl","ln_birth","frtlty_rt",
        "ln_emply","emply_rt","hsprice_yrgrth","dvrcecs_setl_prov_pc"
    ]

    # Dictionary to store models
    models = {}

    # Fit the models
    for i, var in enumerate(dependent_vars, start=1):
        formula = f"{var} ~ dis_to_bor * threshold + year + prov_code" #geo & prov control, yr FE added
        models[f"model{i}"] = smf.wls(formula, rdd_data).fit()

    # Initialize an empty list to collect all DataFrames
    all_tables = []

    # Process each model
    for i in range(1, len(dependent_vars) + 1):
        # Get the summary table
        summary_table = models[f"model{i}"].summary().tables[1]
        
        # Convert the table to a string
        table_str = summary_table.as_text()
        
        # Process the text to extract rows
        lines = table_str.splitlines()
        
        # Extract column names and data
        col_names = lines[0].split()  # Get column names
        data = [line.split() for line in lines[1:]]  # Extract rows
        
        # Adjust column alignment (if necessary)
        if len(data[0]) != len(col_names):
            col_names = lines[1].split()  # Try next row as column names
            data = [line.split() for line in lines[2:]]  # Skip the misaligned header
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data, columns=col_names)
            df.insert(0, "Model", f"model{i}")  # Add a column to indicate the model
            all_tables.append(df)  # Collect the DataFrame
        except ValueError as e:
            print(f"Error creating DataFrame for Model {i}: {e}")
            continue  # Skip this model if there's an error

    # Combine all tables into one DataFrame
    combined_df = pd.concat(all_tables, ignore_index=True)
    combined_df.to_excel('All_model_summary.xlsx', index=False, sheet_name='models') 

    
    # Plot the RDD Graph
    def plot_rdd(data=rdd_data, running_var='dis_to_bor', outcome_var='dvrcecs_setl_prov', cutoff=0):

        # Scatter plot with binning
        bin_width = (data[running_var].max() - data[running_var].min()) / 30
        bins = np.arange(data[running_var].min(), data[running_var].max() + bin_width, bin_width)
        
        # Calculate bin means
        bin_means = []
        bin_centers = []
        for i in range(len(bins)-1):
            bin_mask = (data[running_var] >= bins[i]) & (data[running_var] < bins[i+1])
            if bin_mask.sum() > 0:
                bin_means.append(data.loc[bin_mask, outcome_var].mean())
                bin_centers.append((bins[i] + bins[i+1]) / 2)

        # Scatter of binned means
        plt.figure(figsize=(10, 6))
        plt.scatter(bin_centers, bin_means, color='navy', label='Binned Means', alpha=0.7)
        
        # Raw data scatter (more transparent)
        plt.scatter(data[running_var], data[outcome_var], color='lightblue', alpha=0.1)
        
        # Fit separate regressions for each side of the cutoff
        left_data = data[data[running_var] < cutoff]
        right_data = data[data[running_var] >= cutoff]
        
        # Polynomial fit for left side
        left_poly = np.polyfit(left_data[running_var], left_data[outcome_var], 1)
        left_fit = np.poly1d(left_poly)
        
        # Polynomial fit for right side
        right_poly = np.polyfit(right_data[running_var], right_data[outcome_var], 1)
        right_fit = np.poly1d(right_poly)
        
        # Create prediction lines
        left_x = np.linspace(left_data[running_var].min(), cutoff, 100)
        right_x = np.linspace(cutoff, right_data[running_var].max(), 100)
        
        # Plot fitted lines
        plt.plot(left_x, left_fit(left_x), color='red', linestyle='-', linewidth=2, label='Left Fit')
        plt.plot(right_x, right_fit(right_x), color='red', linestyle='-',linewidth=2, label='Right Fit')
        
        # Vertical line at cutoff
        plt.axvline(x=cutoff, color='green', linestyle=':', label='Cutoff')
        
        # Customize plot
        plt.title(f'Regression Discontinuity Plot\n{outcome_var} vs {running_var}')
        plt.xlabel(running_var)
        plt.ylabel(outcome_var)
        plt.legend()
        #print(left_fit(left_x)) #This code can check whether the fit line has issues, if nan, issue is there
        
        return plt
    
    # sort out dependent variables with significance
    dependent_vars_sig = [
        "scnd_gdp", "incm_tnpc", "cpi_04","dvrcecs_setl_prov_pc","open", "ln_avghsprice", "hsprice_pegrth"
    ]
    
    # Find and create directory for plots and save them 
    main_folder = os.getcwd()
    plot_folder = os.path.join(main_folder,"Plots_Out")
    os.makedirs(plot_folder, exist_ok=True)
    
    # plot all variables with significance 
    for i in dependent_vars_sig:
        plot_rdd(rdd_data,'dis_to_bor',i)
        plot_Name = os.path.join(plot_folder,i) + ".png"
        plt.savefig(plot_Name)
        plt.close()

# trying to find out the responsiveness of TCP
if runTCP:
        # Load the data again so no need to run previous parts, same logic
    data = pd.read_excel('Dataset_labeled.xlsx')
    rdd_data = data#[(data['year'].isin([2016]))] # If analysing for specific year for analysis

    # Regression Discontinuity Design
    rdd_data = rdd_data.assign(threshold=(rdd_data["dis_to_bor"] > 0).astype(int))
    rdd_data_TCP = rdd_data[rdd_data['year'].isin([2018, 2019, 2020])]
    rdd_data_TCP.to_excel('Dataset_labeled_TCP.xlsx', index=False, sheet_name='Dataset') 
    dependent_vars = [
        "scnd_gdp", "incm_tnpc", "cpi_04","dvrcecs_setl_prov_pc","open", "ln_avghsprice", "hsprice_pegrth"
    ]

    # Dictionary to store models
    models = {}

    # Fit the models
    for i, var in enumerate(dependent_vars, start=1):
        formula = f"{var} ~ dis_to_bor * threshold + year + prov_code" #geo & prov control, yr FE added
        models[f"model{i}"] = smf.wls(formula, rdd_data_TCP).fit()

    # Initialize an empty list to collect all DataFrames
    all_tables = []

    # Process each model
    for i in range(1, len(dependent_vars) + 1):
        # Get the summary table
        summary_table = models[f"model{i}"].summary().tables[1]
        
        # Convert the table to a string
        table_str = summary_table.as_text()
        
        # Process the text to extract rows
        lines = table_str.splitlines()
        
        # Extract column names and data
        col_names = lines[0].split()  # Get column names
        data = [line.split() for line in lines[1:]]  # Extract rows
        
        # Adjust column alignment (if necessary)
        if len(data[0]) != len(col_names):
            col_names = lines[1].split()  # Try next row as column names
            data = [line.split() for line in lines[2:]]  # Skip the misaligned header
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data, columns=col_names)
            df.insert(0, "Model", f"model{i}")  # Add a column to indicate the model
            all_tables.append(df)  # Collect the DataFrame
        except ValueError as e:
            print(f"Error creating DataFrame for Model {i}: {e}")
            continue  # Skip this model if there's an error

    # Combine all tables into one DataFrame
    combined_df = pd.concat(all_tables, ignore_index=True)
    combined_df.to_excel('All_model_summary_TCP.xlsx', index=False, sheet_name='models') 

    
    # Plot the RDD Graph
    def plot_rdd(data=rdd_data_TCP, running_var='dis_to_bor', outcome_var='dvrcecs_setl_prov', cutoff=0):

        # Scatter plot with binning
        bin_width = (data[running_var].max() - data[running_var].min()) / 30
        bins = np.arange(data[running_var].min(), data[running_var].max() + bin_width, bin_width)
        
        # Calculate bin means
        bin_means = []
        bin_centers = []
        for i in range(len(bins)-1):
            bin_mask = (data[running_var] >= bins[i]) & (data[running_var] < bins[i+1])
            if bin_mask.sum() > 0:
                bin_means.append(data.loc[bin_mask, outcome_var].mean())
                bin_centers.append((bins[i] + bins[i+1]) / 2)

        # Scatter of binned means
        plt.figure(figsize=(10, 6))
        plt.scatter(bin_centers, bin_means, color='navy', label='Binned Means', alpha=0.7)
        
        # Raw data scatter (more transparent)
        plt.scatter(data[running_var], data[outcome_var], color='lightblue', alpha=0.1)
        
        # Fit separate regressions for each side of the cutoff
        left_data = data[data[running_var] < cutoff]
        right_data = data[data[running_var] >= cutoff]
        
        # Polynomial fit for left side
        left_poly = np.polyfit(left_data[running_var], left_data[outcome_var], 1)
        left_fit = np.poly1d(left_poly)
        
        # Polynomial fit for right side
        right_poly = np.polyfit(right_data[running_var], right_data[outcome_var], 1)
        right_fit = np.poly1d(right_poly)
        
        # Create prediction lines
        left_x = np.linspace(left_data[running_var].min(), cutoff, 100)
        right_x = np.linspace(cutoff, right_data[running_var].max(), 100)
        
        # Plot fitted lines
        plt.plot(left_x, left_fit(left_x), color='red', linestyle='-', linewidth=2, label='Left Fit')
        plt.plot(right_x, right_fit(right_x), color='red', linestyle='-',linewidth=2, label='Right Fit')
        
        # Vertical line at cutoff
        plt.axvline(x=cutoff, color='green', linestyle=':', label='Cutoff')
        
        # Customize plot
        plt.title(f'Regression Discontinuity Plot\n{outcome_var} vs {running_var}')
        plt.xlabel(running_var)
        plt.ylabel(outcome_var)
        plt.legend()
        #print(left_fit(left_x)) #This code can check whether the fit line has issues, if nan, issue is there
        
        return plt
    
    # sort out dependent variables with significance
    dependent_vars_sig_TCP = ["scnd_gdp", "incm_tnpc", "cpi_04","dvrcecs_setl_prov_pc","open", "ln_avghsprice", "hsprice_pegrth"]
    
    # Find and create directory for plots and save them 
    main_folder = os.getcwd()
    plot_folder = os.path.join(main_folder,"Plots_Out_TCP")
    os.makedirs(plot_folder, exist_ok=True)
    
    # plot all variables with significance 
    for i in dependent_vars_sig_TCP:
        plot_rdd(rdd_data_TCP,'dis_to_bor',i)
        plot_Name = os.path.join(plot_folder,i) + ".png"
        plt.savefig(plot_Name)
        plt.close()

# Fake cutoff to check robustness
if runRobust:
     # Load the data again so no need to run previous parts, same logic
    data = pd.read_excel('Dataset_labeled_dropNA.xlsx')

    # Adjust `dis_to_bor` for robustness check
    # DEPENDS ON WHICH PROVINCES WE USE FOR ROBUSTNESS CHECK, DELETE THE _ AT THE END
    data['dis_scale_'] = data['dis_to_bor'].apply(lambda x: None if x > 0 else x + 150)

    # Create 'dis_scale_negative': subtract 150 from non-positive values
    data['dis_scale'] = data['dis_to_bor'].apply(lambda x: None if x <=0 else x - 150)


    # Additional calculated column
    data['dvrcecs_setl_prov_pc'] = data['dvrcecs_setl_prov'] / data['tot_pop']

    # Save updated dataset
    data.to_excel('Dataset_labeled_robust.xlsx', index=False, sheet_name='Dataset_robust')

    # Prepare for Regression Discontinuity Design
    rdd_data_robust = data.assign(threshold=(data["dis_to_bor"] > 0).astype(int))

    dependent_vars = [
        "scnd_gdp", "incm_tnpc", "cpi_04","dvrcecs_setl_prov_pc","open", "ln_avghsprice", "hsprice_pegrth"
    ]

    # Dictionary to store models
    models = {}

    # Fit the models
    for i, var in enumerate(dependent_vars, start=1):
        formula = f"{var} ~ dis_scale * threshold + year + prov_code" #geo & prov control, yr FE added
        models[f"model{i}"] = smf.wls(formula, rdd_data_robust).fit()

    # Initialize an empty list to collect all DataFrames
    all_tables = []

    # Process each model
    for i in range(1, len(dependent_vars) + 1):
        # Get the summary table
        summary_table = models[f"model{i}"].summary().tables[1]
        
        # Convert the table to a string
        table_str = summary_table.as_text()
        
        # Process the text to extract rows
        lines = table_str.splitlines()
        
        # Extract column names and data
        col_names = lines[0].split()  # Get column names
        data = [line.split() for line in lines[1:]]  # Extract rows
        
        # Adjust column alignment (if necessary)
        if len(data[0]) != len(col_names):
            col_names = lines[1].split()  # Try next row as column names
            data = [line.split() for line in lines[2:]]  # Skip the misaligned header
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data, columns=col_names)
            df.insert(0, "Model", f"model{i}")  # Add a column to indicate the model
            all_tables.append(df)  # Collect the DataFrame
        except ValueError as e:
            print(f"Error creating DataFrame for Model {i}: {e}")
            continue  # Skip this model if there's an error

    # Combine all tables into one DataFrame
    combined_df = pd.concat(all_tables, ignore_index=True)
    combined_df.to_excel('All_model_summary_robust.xlsx', index=False, sheet_name='models') 

    
    # Plot the RDD Graph
    def plot_rdd(data=rdd_data_robust, running_var='dis_scale', outcome_var='dvrcecs_setl_prov', cutoff=0):

        # Scatter plot with binning
        bin_width = (data[running_var].max() - data[running_var].min()) / 30
        bins = np.arange(data[running_var].min(), data[running_var].max() + bin_width, bin_width)
        
        # Calculate bin means
        bin_means = []
        bin_centers = []
        for i in range(len(bins)-1):
            bin_mask = (data[running_var] >= bins[i]) & (data[running_var] < bins[i+1])
            if bin_mask.sum() > 0:
                bin_means.append(data.loc[bin_mask, outcome_var].mean())
                bin_centers.append((bins[i] + bins[i+1]) / 2)

        # Scatter of binned means
        plt.figure(figsize=(10, 6))
        plt.scatter(bin_centers, bin_means, color='navy', label='Binned Means', alpha=0.7)
        
        # Raw data scatter (more transparent)
        plt.scatter(data[running_var], data[outcome_var], color='lightblue', alpha=0.1)
        
        # Fit separate regressions for each side of the cutoff
        left_data = data[data[running_var] < cutoff]
        right_data = data[data[running_var] >= cutoff]
        
        # Polynomial fit for left side
        left_poly = np.polyfit(left_data[running_var], left_data[outcome_var], 1)
        left_fit = np.poly1d(left_poly)
        
        # Polynomial fit for right side
        right_poly = np.polyfit(right_data[running_var], right_data[outcome_var], 1)
        right_fit = np.poly1d(right_poly)
        
        # Create prediction lines
        left_x = np.linspace(left_data[running_var].min(), cutoff, 100)
        right_x = np.linspace(cutoff, right_data[running_var].max(), 100)
        
        # Plot fitted lines
        plt.plot(left_x, left_fit(left_x), color='red', linestyle='-', linewidth=2, label='Left Fit')
        plt.plot(right_x, right_fit(right_x), color='red', linestyle='-',linewidth=2, label='Right Fit')
        
        # Vertical line at cutoff
        plt.axvline(x=cutoff, color='green', linestyle=':', label='Cutoff')
        
        # Customize plot
        plt.title(f'Regression Discontinuity Plot\n{outcome_var} vs {running_var}')
        plt.xlabel(running_var)
        plt.ylabel(outcome_var)
        plt.legend()
        #print(left_fit(left_x)) #This code can check whether the fit line has issues, if nan, issue is there
        
        return plt
    
    # sort out dependent variables with significance
    dependent_vars_sig = [
        "scnd_gdp", "incm_tnpc", "cpi_04","dvrcecs_setl_prov_pc","open", "ln_avghsprice", "hsprice_pegrth"
    ]
    
    # Find and create directory for plots and save them 
    main_folder = os.getcwd()
    plot_folder = os.path.join(main_folder,"Plots_Out_robust")
    os.makedirs(plot_folder, exist_ok=True)
    
    # plot all variables with significance 
    for i in dependent_vars_sig:
        plot_rdd(rdd_data_robust,'dis_scale',i)
        plot_Name = os.path.join(plot_folder,i) + ".png"
        plt.savefig(plot_Name)
        plt.close()
#####################

# Model Estimation by Machine Learning 
if runML:
    print("See ML_Testing.py, Seems not working :(")