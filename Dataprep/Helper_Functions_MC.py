#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns

def simulation_demonstration(df_list, simulations, plot_original=False, plot_dummied=False, plot_transformed=False, plot_transformed_simulation = False, plot_back_transformed_simulation = False,return_diff=False, return_dfs=False):
    '''The function that conducts the full transformation, simulation and back-transformation. As it was designed for a 
    specific experiment, the parameters "df_list" and "simulations" are intended to be lists of several dataframes and 
    simulation numbers to ease the use for single and multiple values for these parameters. 
    
    The parameters starting with "plot_" give the plots of the datasets at different stages of the simulation. 
    
    "return_diff" gives two dataframes that compares the absolute difference of the correlation matrices for each given 
    dataframe at diffferent stages of the simulation as well as for each number of simulation. Requires the dataframes
    to have names assigned to them.
    
    "return_dfs" returns the final simulated dataframes.'''
    
    # Initialize lists to store the results
    orr_trans=[]
    trans_clean=[]
    
    orr_corr_sim=[]
    orr_back_trans=[]
    
    total_full_sims=[]
    
    # Loop through the dataframes
    for count, df in enumerate(df_list):
        
        column_order = df.columns

        # Plot the original data
        if plot_original==True :
            print('Dataframe ' + str(count+1) + ' out of ' + str(len(df_list)) + ' Dataframes')
            for col in df.columns:

                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):

                    # If it's numeric, plot a histogram
                    sns.histplot(data=df, x=col)

                else:
                    # If it's not numeric, plot a countplot
                    sns.countplot(data=df, x=col)

                # Show the plot
                plt.show()
            continue

        
        # Perform one-hot encoding
        enc_df=pd.get_dummies(df.select_dtypes('object'), drop_first=False, prefix_sep='___')
        df_num = df.select_dtypes(exclude="object_").join(enc_df)
        
        # Calculate the correlation matrix of the original data
        corr_orig = df_num.corr()

        # Plot the data after one-hot encoding
        if plot_dummied==True:
            print('Dataframe ' + str(count+1) + ' out of ' + str(len(df_list)) + ' Dataframes')
            df_num.hist(figsize=(20, 20), bins=20, xlabelsize=8, ylabelsize=8)
            plt.show()
            continue

        #power transform (and standardize), that is, make it more Gaussian 
        pt = PowerTransformer()
        df_num_trans=pt.fit_transform(df_num)
        df_num_trans = pd.DataFrame(data=df_num_trans, columns=df_num.columns)

        
        # Plot the data after power transformation
        if plot_transformed==True:
            print('Dataframe ' + str(count+1) + ' out of ' + str(len(df_list)) + ' Dataframes')
            df_num_trans.hist(figsize=(20, 20), bins=20, xlabelsize=8, ylabelsize=8)
            plt.show()
            continue
            
        
         # Calculate the correlation matrix of the transformed data
        corr_trans = df_num_trans.corr()
        
        # Calculate the difference in correlation matrices before and after transformation
        orr_trans.append(np.mean(np.absolute(np.subtract(corr_orig.values ,corr_trans.values))))

        # Simulate random nubers between 0 and 1 for each value of simulations
        x_uncor=[]
        for i in simulations:
            x_uncor.append(np.random.normal(0, 1, (len(df_num.columns), i)))

        # Clean correlation matrix by adding a marginal number to the diagonal
        # It works because the source of negative eigenvalues (incrementally close to 0) is so-called "numeric fuzz" that arises when using python float values, see:
        # https://pythonnumericalmethods.berkeley.edu/notebooks/chapter09.03-Roundoff-Errors.html#:~:text=This%20has%20a%20side%20effect,is%20called%20round%2Doff%20error.
        C = np.diag([0.1] * corr_trans.shape[0])
        corr_trans_clean = corr_trans+C

        # Calculate the difference in correlation matrices before and after adding the small number to the diagonal
        trans_clean.append(np.mean(np.absolute(np.subtract(corr_trans_clean.values ,corr_trans.values))))

        # Correlate monte carlo simulations and save datasets and their correlation matrices
        L = np.linalg.cholesky(corr_trans_clean)
        x_cor=[]
        x_data=[]
        for simul in x_uncor:
            y = np.dot(L, simul)
            y=pd.DataFrame(y).T
            y.columns = df_num.columns
            x_data.append(y)
            y= y.corr()
            x_cor.append(y)

        # Plot the simulations after correlating them
        if plot_transformed_simulation==True:
            print('Dataframe ' + str(count+1) + ' out of ' + str(len(df_list)) + ' Dataframes')
            for i, simul in zip(x_data, simulations):
                print(str(simul) + 'Simulations')
                i.hist(figsize=(20, 20), bins=20, xlabelsize=8, ylabelsize=8)
                plt.show()
                continue

        # Check how close the simulated correlation matrices get to the original correlation matrix
        for i in x_cor:
            orr_corr_sim.append(np.mean(np.absolute(np.subtract(i.values ,corr_orig.values))))


        # Transform simulated datasets back in order to get the original distribution (including un-standardizing)
        x_data2=[]
        for i in x_data:
            back_transform= pt.inverse_transform(i)
            back_transform = pd.DataFrame(back_transform, columns=df_num.columns)
            x_data2.append(back_transform)

        # Check differences in correlation between original correlation and simulated, back-transformed
        for i in x_data2:
            orr_back_trans.append(np.mean(np.absolute(np.subtract(i.corr().values ,corr_orig.values))))

        #Plot back-transformed simulations
        if plot_back_transformed_simulation==True:
            print('Dataframe ' + str(count+1) + ' out of ' + str(len(df_list)) + ' Dataframes')
            for i, simul in zip(x_data2, simulations):
                print(str(simul) + 'Simulations')
                i.hist(figsize=(20, 20), bins=20, xlabelsize=8, ylabelsize=8)
                plt.show()
                continue

        # Initialize lists of column groups
        dummy_groups = list(df.select_dtypes("object").columns)
        binaries=[]
        numerics=[]

        # Loop through all numeric column names and check if it is a binary or not
        for col in df.select_dtypes(exclude="object_").columns:

            # Select the column and get its minimum and maximum values
            min_value = df_num[col].min()
            max_value = df_num[col].max()

            # Check if the minimum and maximum values are equal to 0 and 1, respectively
            if min_value == 0 and max_value == 1 and len(df_num[col].unique()) == 2:
                # If so, append the column name to the list of selected columns
                binaries.append(col)
            else :
                numerics.append(col)
        # Initalize lists of sub-dataframes
        df_non_cats=[]
        df_rounded_dummies=[]
        full_sims=[]

        # Save mean values of binaries for proper rounding
        mean_values=df[binaries].mean()


        # Loop though correlated simulations
        for i in x_data2:

            # Fill missing values
            for feature in i.columns:
                i[feature].fillna(i[feature].mean(), inplace=True)

            # Merge non-categorical columns together    
            df_non_cat = pd.concat([i[numerics], i[binaries]], axis=1)

            # Round binaries
            for feature in df_non_cat[binaries]:
                df_non_cat[feature] = (df_non_cat[feature] > mean_values[feature]).astype(int)

            # Set highest number of each dummy per categorical column to 1 and rest to 0
            empty= pd.DataFrame()
            for cat in dummy_groups:
                part_df= i.iloc[:, i.columns.str.startswith(cat)]
                part_df_rounded = part_df.apply(lambda x: pd.Series([1 if i == x.max() else 0 for i in x], index=part_df.columns), axis=1)
                empty = pd.concat([empty, part_df_rounded], axis=1)

            # De-dummy categorical columns
            cats_sim = pd.from_dummies(empty, sep='___')


            # Merge non-categorical sub-data with categorical-data
            full_sim = df_non_cat.join(cats_sim)
            full_sim = full_sim.reindex(columns=column_order)
            full_sims.append(full_sim)
        
        # Save full simulations in list    
        total_full_sims.append(full_sims)


    #Return dataframe with differences between correlation matrices
    if return_diff == True:
        df_names = ['Small', 'German', 'Deloitte', 'Large']  
        # Create output dataframe
        single_matrix_diffs = pd.DataFrame(
        {'Datasets' : df_names,
         'original vs. transformed': orr_trans,
         'transformed vs. cleaned': trans_clean,})
        single_matrix_diffs.set_index('Datasets', inplace = True)    

        #Set MultiIndex to ensure proper comparison between datasets and simulations
        index = pd.MultiIndex.from_product([simulations, df_names], names=['number of simulations', 'dataset'])

        # Create output dataframe
        sim_matrix_diffs = pd.DataFrame({'original vs. simulated & correlated': orr_corr_sim,
                                         'simulated, correlated & back-transformed': orr_back_trans},
                                         index=index)

        return single_matrix_diffs, sim_matrix_diffs
    
    # Return final simulated dataframes as a list
    if return_dfs == True:
        return total_full_sims
    
def al_splitting(orr_df, full_sim):
    '''perform splitting for Active Learning. 
    Inputs are:
        orr_df: original dataset
        full_sim: full Monte Carlo Simulation of the given dataframe
        
    Outputs are:
        df_start: dataset of accepted cases for the start
        df_al_splits: list of dataframes for active learning consisting of the original dataset
        df_al_mc_splits: list of dataframes for active learning consisting of the original and simulated cases'''
    simulated_bads = full_sim[full_sim['BAD'] == 1]

    simulated_bads_sample = simulated_bads.sample(n=(round(0.2/0.8*0.9*len(orr_df))), random_state=888)
    
    # Split the dataframe into independent and dependent variables and shuffle
    sets = train_test_split(orr_df, test_size=0.1, random_state=888) 
    df_al, df_start = sets

    df_al_mc = pd.concat([df_al,simulated_bads_sample])


    # Shuffle the Dataframe and convert it to an array
#    df_al_array = df_al.sample(frac=1, random_state=888).values

#    # Split the array into parts of specified size
#    splits = np.array_split(df_al_array, 9)

    # Convert each split back to a DataFrame and store it in a list
#    df_al_splits = [pd.DataFrame(split, columns=orr_df.columns) for split in splits]




    # Shuffle the Dataframe and convert it to an array
#    df_al_mc_array = df_al_mc.sample(frac=1, random_state=888).values
    df_al_mc = df_al_mc.sample(frac=1, random_state=888)

    # Split the array into parts of specified size
#    splits = np.array_split(df_al_mc_array, 9)

    # Convert each split back to a DataFrame and store it in a list
#    df_al_mc_splits = [pd.DataFrame(split, columns=orr_df.columns) for split in splits]
    
    return df_start, df_al, df_al_mc


# In[ ]:




