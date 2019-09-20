# StructuredData.py
from General.Core import *
from General.Layers import *
from General.Learner import *
from General.LossesMetrics import *
from General.Optimizer import *

# Problem Description:
# Given input variables X_1,...,X_m and wish to predict value of an output variable 
# (i.e. dependent variable) Y. Inputs X_i may be categorical or real-valued variables, 
# or some combination of the two types. Output variable Y may be either categorical or 
# real-valued. All variables should have a natural conceptual meaning, and the inputs 
# are generally assumed to be heterogeneous. 

# NOTE: In the description, docstrings, and functions that follow we generally  
# refer to real-valued variables as 'continuous' or 'cont', even if they only
# take integer values. Categorical variables are often referred to as 'cat'.

# Example: Car insurance
# X_1 = age of cutomer (continuous) 
# X_2 = zipcode of customer (categorical)
# X_3 = total amount of claims by customer in past year (continuous)
# X_4 = whether of not customer had an accident in past year (categorical, "yes" or "no") 
# Possible Output 1: Y = total amount of claims by customer in next year (continuous)
# Possible Output 2: Y = whether or not customer has accident in next year (categorical)

# In order to make predictions we are given n data points of (input + output) 
# of form (x_1,...,x_m,y) corresponding to realizations of variables (X_1,...,X_m,Y).
# The starting point for constructing and training models in Section 2 of this file 
# StructuredData.py is the function ProcessDataFrame, which assumes the data is in a 
# pandas DataFrame of following form:

#  (index)   age zipcode  claims_previous_year  accident_previous_year  claims_following_year
# 'Person1'  25  12345    0.00                  'No'                    603.00
# 'Person2'  37  37373    0.00                  'No'                    0.00
# 'Person3'  46  46928    452.00                'Yes'                   337.25 
#    .
#    .
#    .
# 'Person_n' 62  20385    246.30                'Yes'                   925.05
 
# Actually, and more precisely, seperate DataFrames of the above form for train and validation data 
# should be passed into the function ProcessDataFrame. And (optionally) a DataFrame of the above form
# with the output variable column removed can also be passed into the function ProcessDataFrame
# for test data.

# NOTE 1: Normally the data will be given in .csv files (or possibly excel files) and will have to 
#         be read in by the user to create a pandas DataFrame. However, often the user may also
#         want to do some feature engineering on the given data. That is, the given input variables 
#         in the .csv file may be (X1,...,Xm), but the user may want to create from them new input variables
#         (X1_hat, ..., Xk_hat) to use as the inputs. For instance, an input variable might be the date of a 
#         business transaction, but the user may want to extract from the date the corresponding <day_of_week>
#         and <month_of_year> variables, to use explicitly as inputs. Various methods for exploratory data
#         analysis and feature engineering are contained in Section 1 of this file StructuredData.py.

#         HOWEVER, IT IS ASSUMED THAT ALL FEATURE ENGINEERING HAS BEEN DONE PRIOR TO PASSING 
#         A DATAFRAME INTO THE FUNCTION ProcessDataFrame. 

# NOTE 2: In many instances when dealing with real data there may be missing values for some of the 
#         input variables for some of the data points. For instance, in the car insurance example 
#         we may not know the amount of claims the previous year for a particular person, because
#         that person had car insurance with a different company the previous year. The ProcessDataFrame
#         function allows for missing values in some of the inputs. (The docstring for that 
#         function explains how missing values are dealt with.)

# OUTLINE:

# Section (1) - EDA AND FEATURE ENGINEERING
# (1.1) - EDA Part 1: Plotting Variable Distributions and Relationships
# (1.2) - EDA Part 2: Numerical measures of association between variables
# (1.3) - Feature Engineering Tools

# Section (2) - DATAOBJ CONSTRUCTION AND MODELING
# (2.1) - DataObj Construction
# (2.2) - Models


# SECTION 1 - EDA AND FEATURE ENGINEERING

# (1.1) EDA Part 1: Plotting Variable Distributions and Relationships

def get_variable_names(df,variables):
    """Simple helper function for other plotting functions below.
    If <variables> is a list of column names in <df>, returns them unchanged.
    If <variables> is a list of integer indices of columns, converts to list of column names. 
    """ 
    
    columns = list(df.columns)
    for i in range(len(variables)):
        if type(variables[i])==int: variables[i]=columns[variables[i]]
    return variables 

def plot_distributions(df,var_type,variables,num_cols=4):
    """Function to plot distributions of either continuous OR categorical variables 
       in a pandas dataframe. For continuous variables plots histograms with kernel 
       denisty etimates overlayed, for categorical variables plots bar graphs. 
    
    Arguments:
    df: A pandas dataframe 
    var_type: Either 'cont' or 'cat'. 
    variables: List of column names of variables in df to plot distributions of. If a list of 
               integers is given instead these are used as the indices of the columns.
               If var_type == 'cont', all variables in list must be continuous or at least real-valued.
               If var_type == 'cat', all variables in list must be categorical or integer-valued. 
    num_cols: Number of histograms or bar graphs plotted on each row.
    """
    
    variables, L = get_variable_names(df,variables), len(variables)
    
    if var_type == 'cont':
        means = {variables[i]:df[variables[i]].mean() for i in range(len(variables))}
        stds = {variables[i]:df[variables[i]].std() for i in range(len(variables))}
    
    num_rows = int(np.ceil(L/num_cols))
    plt.figure(figsize=(6*num_cols,5*num_rows))    
    for i in PBar(range(L)):
        plt.subplot(num_rows, num_cols, i+1)
        if var_type == 'cont':
            sns.distplot( df[variables[i]][df[variables[i]].notnull()] )       
            mean, std = '{:.2f}'.format(means[variables[i]]), '{:.2f}'.format(stds[variables[i]])
            plt.title(variables[i] + '  mean=' + mean + ' std=' + std, fontsize=15)
        if var_type == 'cat':
            sns.countplot(x = df[variables[i]][df[variables[i]].notnull()])       
            plt.title(variables[i], fontsize=15)
        plt.tight_layout()
    
def plot_dependence(df,variables,depend_var,types,s=None,num_cols=4):
    """Plots dependence of <depend_var> on each variable in <variables>. Type of plots 
       depends on whether <variables> and <depend_var> are categorical or continuous. 
   
    Arguments:
    df: A pandas dataframe. 
    variables: List of column names of variables in df. Must be either all categorical/integer-valued 
               OR all continuous/real-valued. Should not contain <depend_var>.
    depend_var: Name of a column in df, must be continuous or categorical. 
    types: Either 'ContCont', 'ContCat', 'CatCont', or 'CatCat'.           
           ContCont: Use if depend_var and variables both continuous. 
                     In this case, plots are scatter plots with depend_var on y-axis. 
            ContCat: Use if depend_var is continous and variables are categorical.
                     In this case, plots are vertical violin plots with depend_var on y_axis. 
            CatCont: Use if depend_var is categorical and variables are continuous.
                     In this case, plots are horizontal violin plots with depend_var on y_axis.
             CatCat: Use if depend_var and variables are both categorical. 
                     In this case, plots are stacked bar charts with each variables[i] on x_axis.            
    s: size of markers for scatter plots if types 'ContCont'.
    num_cols: Number of plots on each row.
    """

    variables, L = get_variable_names(df,variables), len(variables)    
    num_rows = int(np.ceil(L/num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols,5*num_rows))
    for i,ax in enumerate(PBar(axes.flat)):
        if i >= L: 
            continue
        elif types == 'ContCont':
            DF = df[df[variables[i]].notnull() & df[depend_var].notnull()]
            DF.plot.scatter(x=variables[i],y=depend_var,s=s,ax=ax)
        elif types == 'ContCat': 
            DF = df[df[variables[i]].notnull() & df[depend_var].notnull()]
            sns.violinplot(x=variables[i],y=depend_var,data=DF,ax=ax)
        elif types == 'CatCont': 
            DF = df[df[variables[i]].notnull() & df[depend_var].notnull()]
            sns.violinplot(x=variables[i],y=depend_var,data=DF,orient='h',ax=ax)
        elif types == 'CatCat': 
            DF = df[df[variables[i]].notnull() & df[depend_var].notnull()]
            crosstab = pd.crosstab(DF[variables[i]],DF[depend_var])
            crosstab.index, crosstab.columns = list(crosstab.index), list(crosstab.columns)
            crosstab.plot.bar(stacked=True,ax=ax)
            ax.set_xlabel(variables[i])
            ax.set_ylabel('count')
        ax.set_title(variables[i],fontsize=15)
     
    plt.tight_layout()

def plot_pairs(df,var_types,variables,s=None):
    """Plots distribution of pairs of variables in df. 
    
    Arguments:
    df: A pandas dataframe. 
    var_types: Either 'Cont','Cat', or 'ContCat'
    variables: If var_types == 'Cont', variables is a list of names of continuous variables in df.
               If var_types == 'Cat', variables is a list of names of categorical variables in df.
               If var_types == 'ContCat', variables = [vars0,vars1] where:
               vars0 is is a list of names of continuous variables in df.
               vars1 is a list of names of categorical variables in df.
               (In all cases lists of integer indices of columns may be used instead of column names.)
    s: size of markers for scatter plots if types 'Cont'.
    """
    
    if var_types in ['Cont','Cat']:
        variables, L = get_variable_names(df,variables), len(variables)
        fig, axes = plt.subplots(L, L, figsize=(6*L,5*L))
        
        for i,j in PBar(list(itertools.product(range(L),range(L)))):
            
            ax = axes[i][j]
            DF = df[df[variables[i]].notnull() & df[variables[j]].notnull()]                    
                
            if i==j and var_types == 'Cont':
                sns.distplot(DF[variables[i]],ax=ax) 
                ax.set_title(variables[i],fontsize=15)    
            elif i==j and var_types == 'Cat':
                sns.countplot(DF[variables[i]],ax=ax)
                ax.set_title(variables[i],fontsize=15)
            elif var_types == 'Cont':
                DF.plot.scatter(x=variables[i],y=variables[j],s=s,ax=ax)
                ax.set_title(variables[i]+' vs. '+variables[j],fontsize=15)
            elif var_types == 'Cat':
                crosstab = pd.crosstab(DF[variables[i]],DF[variables[j]])
                crosstab.index, crosstab.columns = list(crosstab.index), list(crosstab.columns)
                crosstab.plot.bar(stacked=True,ax=ax)
                ax.set_xlabel(variables[i])
                ax.set_ylabel('count')
                ax.set_title(variables[i]+' vs. '+variables[j],fontsize=15) 
                
        plt.tight_layout()
            
    if var_types == 'ContCat':
        vars0 = get_variable_names(df,variables[0])
        vars1 = get_variable_names(df,variables[1])
        L0,L1 = len(vars0),len(vars1)
        fig, axes = plt.subplots(L1, L0, figsize=(6*L1,5*L0))
        
        for i,j in PBar(list(itertools.product(range(L1),range(L0)))):
            ax = axes[i][j]
            DF = df[df[vars1[i]].notnull() & df[vars0[j]].notnull()]
            sns.violinplot(x=vars1[i],y=vars0[j],data=DF,ax=ax)
            ax.set_title(vars1[i]+' vs. '+vars0[j],fontsize=15)
            
        plt.tight_layout()

# (1.2) EDA Part 2: Numerical measures of association between variables  

def missing_value_counts(df):
    """Returns counts of missing values in columns of a df."""
    counts = df.isnull().astype(int).sum()
    return counts[counts>0]

def entropy(df,X):
    """Computes entropy of a categorical variable X in a pandas dataframe df.
       (Assumes no missing values for X in df.) """
    p = df[X].value_counts()/len(df[X])
    return -np.sum(np.log(p)*p)
    
def joint_entropy(df,X,Y):
    """Computes joint entropy of categorical variables X,Y in a pandas dataframe df.
       (Assumes no missing values for X,Y in df.)"""    
    f = np.array(pd.crosstab(df[X],df[Y])).flatten() #frequencies of pairs
    p = f/f.sum()                                    #normalize frequenicies to probabilities
    p = np.maximum(p,1e-20)                          #for numerical stability, in case any p[i] == 0.
    return -np.sum(np.log(p)*p)
    
def normed_mutual_info(df,X,Y,symmetric):
    """Computes 'normed_mutual_info' of categorical variables X,Y in a pandas DataFrame df. 
       If symmetric==False: normed_mutual_info(X,Y) = I(X;Y)/H(Y)
       If symmetric==True: normed_mutual_info(X,Y) = 0.5*( I(X;Y)/H(X) + I(X;Y)/H(Y) ).
       (Assumes no missing values for X,Y in df.) """
    H_X, H_Y, H_XY = entropy(df,X), entropy(df,Y), joint_entropy(df,X,Y)
    I_XY = H_X + H_Y - H_XY
    if symmetric == False: return I_XY/H_Y
    if symmetric == True: return 0.5*(I_XY/H_X + I_XY/H_Y)
    
def correlation_ratio(df,X,Y):
    """ Computes correlation ratio of variables X,Y in a pandas DataFrame df.
        X is categorical, Y is numeric. (Assumes no missing values for X,Y in df.) """       
    mean,var = df[Y].mean(),df[Y].var()
    cat_means = df.groupby(X)[Y].mean()
    cat_counts = df.groupby(X)[Y].count()
    cat_var = (cat_counts*(cat_means - mean)**2).sum()/cat_counts.sum()
    return np.sqrt(cat_var/var)

def max_correlation_ratio(df,X,Y):
    """Computes the following quantity: max_i[ abs((mean(Y|X=x_i) - mean(Y))/std(Y)) ].
       Here X is categorical, Y is numeric, and the max is taken over all possible 
       categories x_i for the variable X.
       
       NOTE: This quantity differs from the standard correlation ratio in that it measures
       the maximum difference between the categorical means and overall mean, instead of 
       the average difference weighted by category frequencies. This can be important 
       if there are some rare categories for X, that have notably different statistics for Y.
       In such cases, you don't want to ignore the variable X for predicting Y, eventhough
       the standard correlation ratio may be very small. 
       """    
    mean,std = df[Y].mean(),df[Y].std()
    cat_means = df.groupby(X)[Y].mean()
    return np.max( np.abs((cat_means - mean)/std) )  
    
def abs_max_correlation(df,X,Y): 
    """ Computes the 'absolute maximum correlation' of numeric variables X,Y 
        in a pandas DataFrame df.
        
    This is defined as: 
    abs_max_corr = max(abs(c1),abs(c2),abs(c3),abs(c4))           
    where c1 = corr(X,Y), c2 = corr(X,Y2), c3 = corr(X2,Y), c4 = corr(X2,Y2)         
    and X2 = abs(X - mean(X)), Y2 = abs(Y - mean(Y)).         
    (Assumes no missing values for X,Y in df.) 
        
    NOTE: This is a significantly more robust measure of the amount of dependence between numeric variables
          then the standard correlation. Correlation measures the linear dependence between variables X and Y. 
          It is still usually fairly large if X and Y have a monotonic, but not linear, relationship. However, 
          it is not always good for picking out non-monotonic relationships. For example, if Y tends to 
          be large when X takes extreme values (far above or below its mean), this measure will show 
          significant dependence, even though correlation may not. 
    """    
    x,y = df[X],df[Y]
    x2,y2 = (x - x.mean()).abs(), (y - y.mean()).abs()
    c1,c2,c3,c4 = x.corr(y), x.corr(y2), x2.corr(y), x2.corr(y2)
    return max(abs(c1),abs(c2),abs(c3),abs(c4))    
    
def get_association(df,X,Y,Type):
    """Computes an unsigned measure of the association strength between variables X,Y in a
       pandas DataFrame df, using one of several possible measures specified by Type.
       Output is always a number in range [0,1], with 0 for independent variables and 1 
       for maximum dependence **.
       
    Possible values of 'Type' are:
    1. 'abs_correlation' or 'abs_max_correlation', for X,Y both numeric. 
    2. 'correlation_ratio' or 'max_correlation_ratio', for X categorical and Y numeric. 
    3. 'mutual_info_symmetric' or 'mutual_info_asymmetric', for X,Y both categorical.    
    
    ** Raw output of function max_correlation_ratio (defined above) is in range [0,infinity),
    instead of [0,1]. The function get_association actually outputs min(max_correlation_ratio/3,1) 
    if Type == 'max_correlation_ratio'. A value of 1 (i.e. max_correlation_ratio=3), indicates
    a 3 std deviation difference between the mean of Y for a particular category of X, and 
    overall mean of Y. We truncate this 3 standard deviation difference as 'maximal dependence'.
    """
    
    if X == Y: return 1.0
    DF = df[df[X].notnull() & df[Y].notnull()][[X,Y]]
    if len(DF)==0: return 0.0
    elif len(DF[X].value_counts())==1 or len(DF[Y].value_counts())==1: return 0.0   
    elif Type == 'abs_correlation': return np.abs(DF[X].corr(DF[Y]))
    elif Type == 'abs_max_correlation': return abs_max_correlation(DF,X,Y)
    elif Type == 'correlation_ratio': return correlation_ratio(DF,X,Y)
    elif Type == 'max_correlation_ratio': return min(max_correlation_ratio(DF,X,Y)/3, 1)
    elif Type == 'mutual_info_asymmetric': return normed_mutual_info(DF,X,Y,symmetric=False)
    elif Type == 'mutual_info_symmetric': return normed_mutual_info(DF,X,Y,symmetric=True)
    
def associations_dependent(df,Type,variables,depend_var,reverse=False,plot=True):
    """Computes association of specified variables in a dataframe with a single 'dependendent' 
       variable. Results returned as a pd.Series and (optionally) also plotted a bar graph.
    
    Arguments: 
    df: A pandas dataframe.
    variables: List of names of variables in df. Must be all categorical OR all continuous/real-valued. 
    depend_var: Name of dependent variable in df to plot associations with.
    Type: The type of association to compute, same as for function get_association.  
    plot: If True, plots bar graph of associations.
    """    
    associations = []
    for var in PBar(variables): 
        if reverse==False: associations.append( get_association(df,var,depend_var,Type) )
        if reverse==True: associations.append( get_association(df,depend_var,var,Type) )
    associations = pd.Series(associations,index=variables)
   
    figsize=(len(associations)/2,max(len(associations)/4,4))
    if plot: associations.plot.bar(title='Associations with '+depend_var,figsize=figsize)
    return associations
                                    
def associations_pairs(df,Type,cat_vars=None,cont_vars=None,plot=True):
    """ Computes association between pairs of a specified collection of variables in a dataframe.
        Results returned in dataframe form and (optionally) also plotted as a heatmap. 
    
    Arguments: 
    df: A pandas dataframe.
    cat_vars: A list of names of categorical variables in df.
    cont_vars: A list of names of continuous/real-valued variables in df. 
    Type: The type of association to compute, same as for function get_association.
    plot: If True, plots heat map of associations.
    
    Output:
    Returns a pandas dataframe of associations between pairs using given 'Type'. 
    If cat_vars == None: associations are for each pair of variables in cont_vars.
    If cont_vars == None: association are for each pair of variables in cat_vars.
    If cat_vars and cont_vars are both given: associations are for each pair of variables 
    X_i,Y_j with X_i in cat_vars and Y_j in cont_vars.   
    """ 
    
    print('computing associations')
    
    if cat_vars and cont_vars:
        Assoc = np.zeros((len(cat_vars),len(cont_vars)))
        for i in PBar(range(len(cat_vars))):
            for j in range(len(cont_vars)):
                Assoc[i,j] = get_association(df, cat_vars[i], cont_vars[j], Type)
        Assoc = pd.DataFrame(Assoc,index=cat_vars,columns=cont_vars)
    
    elif cont_vars is None:
        Assoc = np.zeros((len(cat_vars),len(cat_vars)))
        for i in PBar(range(len(cat_vars))):
            for j in range(len(cat_vars)):
                Assoc[i,j] = get_association(df, cat_vars[i], cat_vars[j], Type)
        Assoc = pd.DataFrame(Assoc,index=cat_vars,columns=cat_vars)
                
    elif cat_vars is None:
        Assoc = np.zeros((len(cont_vars),len(cont_vars)))
        for i in PBar(range(len(cont_vars))):
            for j in range(len(cont_vars)):
                Assoc[i,j] = get_association(df, cont_vars[i], cont_vars[j], Type)
        Assoc = pd.DataFrame(Assoc,index=cont_vars,columns=cont_vars)

    if plot:  
        # For plotting, associations along diagonal are set to 0 when we have all cat_vars 
        # or all cont_vars. This means that if all other associations are relatively small 
        # (e.g. 0.2 or 0.3), then the differences still show up well visually in heatmap. 
        print('plotting heatmap')
        
        if cat_vars and cont_vars:
            Assoc2 = Assoc
            h,w = len(cat_vars),1.5*len(cont_vars)
            heatmap = sns.heatmap(Assoc2,cbar_kws={"shrink": .5},cmap='Reds',fmt='.2f',square=True,
                              linewidths=.5,annot=True,xticklabels=cont_vars,yticklabels=cat_vars)
        elif cont_vars is None:
            Assoc2 = Assoc - pd.DataFrame(np.identity(len(cat_vars)),index=cat_vars,columns=cat_vars)
            h,w = len(cat_vars),1.5*len(cat_vars)
            heatmap = sns.heatmap(Assoc2,cbar_kws={"shrink": .5},cmap='Reds',fmt='.2f',square=True,
                              linewidths=.5,annot=True,xticklabels=cat_vars,yticklabels=cat_vars)
        elif cat_vars is None:
            Assoc2 = Assoc - pd.DataFrame(np.identity(len(cont_vars)),index=cont_vars,columns=cont_vars)
            h,w = len(cont_vars),1.5*len(cont_vars)
            heatmap = sns.heatmap(Assoc2,cbar_kws={"shrink": .5},cmap='Reds',fmt='.2f',square=True,
                              linewidths=.5,annot=True,xticklabels=cont_vars,yticklabels=cont_vars)
            
        heatmap.figure.set_size_inches(h,w)
        sns.despine()
    
    return Assoc
            
# (1.3) Feature Engineering Tools            
            
def add_datepart(df,date_column='Date',start=None):
    """Add various date-part variables such as 'week','month','year',... corresponding to a date variable
       in a pandas dataframe. Also, adds a variable 'days_elapsed', which is the number of days since 
       the specified 'start'. If 'start' == None, the earliest Timestamp in the date_column is used. 
       The added variables are joined directly into df. """
    
    df[date_column] = pd.to_datetime(df[date_column])
    
    df['week'] = df[date_column].dt.week
    df['month'] = df[date_column].dt.month
    df['year'] = df[date_column].dt.year
    
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['dayofmonth'] = df[date_column].dt.day
    df['dayofyear'] = df[date_column].dt.dayofyear    
    
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    
    if start is None: start = min(df[date_column])
    df['days_elapsed'] = (df[date_column] - pd.to_datetime(start))/np.timedelta64(1,'D')

def get_TimeBeforeAfter(df,event_col,index_col=None,groupby_col=None,keep_cols=[],timescale=1):
    """
    Function returns a new dataframe with columns containing amount of time 
    before and after occurences of events in event_col. 
    
    Arguments:
    df: A pandas dataframe.
    event_col: A col in df with 0-1 entries indicating if an event of a certain type occured for a given datapoint. 
    index_col: Normally a col of df such as 'Date' or 'Time' indicating the timestamp of each datapoint. 
               If index_col == None, then df.index is used for time of occurences, which may simply be 
               integers instead of Timestamps. 
    groupby_col: If groupby_col == None, then times before/after events are calculated for entire dataframe df. 
                 Otherwise, times before/after are calculated separately for each distinct label in groupby_col.
    keep_cols: List of additional columns from df to keep in returned object dfBeforeAfter.  
    timescale: Times before/after events are measured in units of timescale. For example, if index_col == 'Date',
               you might set timescale = np.timedelta(1,'D') in order to measure time in units of days. 
                              
    NOTE 1: There cannot be any col of df called 'index', if there is rename it 'Index' or something similar.
    NOTE 2: If groupby_col is None, the indices in df.index or index_col (if given) must be unique. 
            If groupby_col is 'G', indices in df.index/index_col for each label g in 'G' must be unique. 
    """
    
    if groupby_col:
        
        BeforeAfter_dfs = []
        grouped_dfs = df.groupby(groupby_col)
        for label,X in PBar(grouped_dfs):
            BeforeAfter_dfs.append(get_TimeBeforeAfter(X.copy(),event_col,index_col,None,keep_cols+[groupby_col],timescale))
        return pd.concat(BeforeAfter_dfs)
        
    if groupby_col is None:
        
        if index_col is None: 
            df['index'] = df.index.copy()
            index_col = 'index'
        
        before, last_event = [], None  
        dfBefore = df[[index_col,event_col]+keep_cols].sort_values(index_col,ascending=True)
        for i in range(len(dfBefore)):
            if last_event is None: before.append( np.nan )
            else: before.append( (dfBefore[index_col].iloc[i] - last_event)/timescale )
            if dfBefore[event_col].iloc[i] == 1: 
                last_event = dfBefore[index_col].iloc[i]
        dfBefore[event_col+'Before'] = before
        if event_col not in keep_cols: 
            dfBefore.drop(event_col,axis=1,inplace=True)
        
        after, last_event = [], None
        dfAfter = df[[index_col,event_col]].sort_values(index_col,ascending=False)
        for i in range(len(dfAfter)):
            if last_event is None: after.append( np.nan )
            else: after.append( (last_event - dfAfter[index_col].iloc[i])/timescale )
            if dfAfter[event_col].iloc[i] == 1: 
                last_event = dfAfter[index_col].iloc[i]
        dfAfter[event_col+'After'] = after 
        dfAfter.drop(event_col,axis=1,inplace=True)

        if index_col == 'index': df.drop('index',axis=1,inplace=True)
        dfBeforeAfter = dfBefore.join(dfAfter.set_index(index_col),on=index_col)
        return dfBeforeAfter    

def reverse_datetime(datetimes):
    """Simple helper function for the function get_RollingStats. The input 
       datetimes is a datetime index sorted in the order (newest,...,oldest)."""    
    
    diffs = datetimes.map(lambda x: datetimes[0] - x)
    start = pd.Timestamp('01/01/2000')
    reverse = diffs.map(lambda x: start + x)
    return reverse
    
def get_RollingStats(df,columns,window_size,stat_types,index_col=None,groupby_col=None,keep_cols=[]):
    
    """ Function returns a new dataframe with rolling statistics of specified 
        numeric columns for a specified window_size.
       
       Arguments:
       df: A pandas dataframe.
       columns: A list of numeric columns in dataframe to compute rolling statistics for. 
       window_size: Length of window to compute rolling statistics over. For example: 
                    If window_size=3, 3 datapoints in each window.  
                    If window_size='5D' window is of length 5 days before/after current datapoint.
                    Similarly for '5H'=5 hours, '5S'=5 seconds, '5min' = 5 minutes.
                    (When using such window_sizes, index or index_col must contain pd Timestamps).
       stat_types: A list containing 1 or more of following: 'Sum','Min','Max','Mean','Std','Count'. 
                   Foward and Backward rolling statistics of each such type are computed 
                   for each specified column over a window of specified length.       
       index_col: Normally a col of df such as 'Date' or 'Time' indicating the Timestamp of each datapoint. 
                  If index_col == None, then df.index is used for time of occurences, which may simply 
                  be integers instead of Timestamps.               
       groupby_col: If groupby_col == None, then rolling stats are calculated for entire dataframe df. 
                    Otherwise, rolling stats are calculated separately for each distinct label in groupby_col. 
       keep_col: Always leave as default [], only for internal use to preserve groupby_col when self-calling.
       
       NOTE: There cannot be any col of df called 'index', if there is rename it 'Index' or something similar.
       """
    
    if groupby_col:
        
        AllRollingStat_dfs = []
        grouped_dfs = df.groupby(groupby_col)
        for label,X in PBar(grouped_dfs):
            AllRollingStat_dfs.append(get_RollingStats(X,columns,window_size,stat_types,index_col,None,[groupby_col]))
        return pd.concat(AllRollingStat_dfs)    
    
    if groupby_col is None:
        
        df = df.copy()
        groupbycol = keep_cols[0] if keep_cols else None
        if index_col: df.set_index(index_col,inplace=True)         
        RollingBwd = df[columns].sort_index(ascending=True)
        RollingFwd = df[columns].sort_index(ascending=False)
        
        if type(RollingFwd.index[0]) == pd.Timestamp:
            true_fwd_index = copy.deepcopy(RollingFwd.index)
            RollingFwd.index = reverse_datetime(RollingFwd.index)
        
        RollingStatDFs = []        
        for st in stat_types: 
            
            if st == 'Sum':
                X1 = RollingBwd.rolling(window_size,min_periods=1).sum()
                X2 = RollingFwd.rolling(window_size,min_periods=1).sum()
            elif st == 'Min':
                X1 = RollingBwd.rolling(window_size,min_periods=1).min()
                X2 = RollingFwd.rolling(window_size,min_periods=1).min()
            elif st == 'Max':
                X1 = RollingBwd.rolling(window_size,min_periods=1).max()
                X2 = RollingFwd.rolling(window_size,min_periods=1).max()
            elif st == 'Mean':
                X1 = RollingBwd.rolling(window_size,min_periods=1).mean()
                X2 = RollingFwd.rolling(window_size,min_periods=1).mean()
            elif st == 'Std':
                X1 = RollingBwd.rolling(window_size,min_periods=2).std()
                X2 = RollingFwd.rolling(window_size,min_periods=2).std()
            elif st == 'Count':
                X1 = RollingBwd.rolling(window_size,min_periods=1).count()
                X2 = RollingFwd.rolling(window_size,min_periods=1).count()
            
            if type(RollingFwd.index[0]) == pd.Timestamp: X2.index = true_fwd_index
            X1.columns = [c+'RollBwd'+st for c in X1.columns]
            X2.columns = [c+'RollFwd'+st for c in X2.columns]
            RollingStatDFs += [X1,X2]            
        
        RollingStatDFs = RollingStatDFs[0].join(RollingStatDFs[1:])     
        if groupbycol: 
            RollingStatDFs[groupbycol] = df[groupbycol] 
            RollingStatDFs['index'] = RollingStatDFs.index.copy()
        return RollingStatDFs
    
    
# SECTION 2 - DATAOBJ CONSTRUCTION AND MODELING     

# (2.1) - DataObj Construction

def ProcessDataFrame(df, cat_vars, cont_vars, output_var, scale_cont, fill_missing = 'median', 
                     category_labels = None, unknown_category = True):
    
    """ Function to pre-process a pandas DataFrame of structured data 
        before training a neural network with it. 
        
        NOTE: This function modifies the original dataframe df in place as part of the 
              processing, eventhough the outputs are seperate processed components of df. 
              If you want to preserve the original dataframe df, pass in df.copy() instead. 
   
Arguments:

df: A pandas DataFrame of form given above in car insurance example at beginning of this file. 

cat_vars: A list of the column names of categorical variables in <df>. 
          Each var must contain only string values and possibly np.nans, OR
          only integer values and possibly np.nans. Integers may be in float form (e.g. 7.0) 
          with a dtype of float, np.float32, or np.float64 for the var.             

cont_vars: A list of the column names of continuous/real-valued variables in <df>.
           These may be stored as float, int, or other numeric type in <df>.

output_var: * If <df> is for test data, set output_var = None. 
            * If <df> is for train or validation data, set ouput_var to be the 
              column name of the output variable in <df>. 
              
(NOTE: output_var should be included in either the list cat_vars or the list cont_vars.)
              
scale_cont: * If <scale_cont> == 'No' then function does no rescaling of continuous variables. 
            * If <scale_cont> == 'by_df', then function rescales all continous input variables
              in <df> to have (empirical) mean = 0 and std = 1, in the returned object <xcont_df>. 
            * If <scale_cont> is a dictionary object with entries of form << cont_var_name: [mean,std] >>, 
              then each continous input variable of <df> is rescaled in the returned object <xcont_df>, 
              by subtracting given mean and then dividing by given std.
              
fill_missing: Method to fill in missing values for continous variables.
              Choices are 'mean','median', or c (where c is a constant, given as float or integer). 
              Default is 'median'. 
              
category_labels: * If 0 categorical variables OR running function on a DataFrame of training 
                   data, then leave as the default, <category_labels> = None.
                 * Otherwise, use as <category_labels> the output of the 
                   function ProcessDataFrame when run on the training DataFrame. 
                   
unknown_category: If unknown_category == True, adds an extra 'unknown' category to all 
                  categorical variables and calls missing entries 'unknown'. 
  
              
Output: 
Returns 5 objects: xcat_df, xcont_df, y, scaling_values, category_labels 

xcat_df:  A dataframe whose columns correspond to the columns of <df> with categorical input variables. 
          For each column the following processing steps are applied:
          * If <unknown_category> == True: Existing categories are renamed 1,2,3... and 
            all missing values are replaced by another category 0. 
          * If <unknown_category> == False: Existing categories are renamed 0,1,2...
            (Assumes no missing values). 
             
xcont_df: A dataframe whose columns correspond to the columns of <df> with continuous input variables.
          The following processing steps are done to these columns:
          (1) All missing values of continuous variables from <df> are replaced by either mean, median, 
              or a constant, as specified by <fill_missing>. 
          (2) Continuous variables rescaled (or not) according to the value of <scale_cont>. 
              This rescaling is done after the filling in of any missing values in step (1).

y: * If <df> is for test data (so that there is no <output_var> column) then y = None. 
   * If <df> is for training or validation then y is the <output_var> column of <df> in np.array form.  
   * If <output_var> is categorical, categories also renamed 0,1,2, ... in y.  
    
scaling_values: * If <scale_cont> == 'No', then is equal to None. 
                * If <scale_cont> != 'No', then is a dictionary with 
                  entries of form << cont_var_name: [mean,std]. >>
                               
category_labels: A list with entries which are dictionaries. 
                 The ith dictionary gives the labels for the ith categorical variable. 
                 The order of categorical variables is:
                 First, the input categorical variables in same order as <xcat_df>.
                 Last, the output variables if it is categorical. 
                 
                 Example: Assume ith cagtegorical variable is <accident_last_year> and 'unknown' has 
                 label 0, 'yes' has label 1, and 'no' has label 2 in the returned dataframe <xcat_df>. 
                 Then the dictionary would be: {'unknown':0, 'yes':1, 'no':2}.      
"""
    
    # Make lists of categorical and continuous input variables
    xcat_vars, xcont_vars = cat_vars.copy(), cont_vars.copy()
    if output_var in cat_vars: xcat_vars.remove(output_var)
    if output_var in cont_vars: xcont_vars.remove(output_var)
    
    # Ensure proper types of cat and cont variables in df.
    # All cat_vars changed to pd.Categorical with categories that are strings.
    # If there are any np.nan's in cat_vars values, they are relabelled as the string 'nan'. 
    for var in cont_vars: 
        df[var] = df[var].astype('float32')
    
    var2idx = {list(df.columns)[i]:i for i in range(len(df.columns))}
    for var in cat_vars:
        if df[var].dtype in [float,'float32','float64']:
            m = max(df[var][df[var].notnull()]) + 1
            df[var].fillna(m,inplace=True)
            df[var] = df[var].astype(int)           
            df.iloc[np.where(df[var] == m)[0],var2idx[var]] = 'nan'            
            df[var] = df[var].astype(str).astype('category')
        else: 
            df[var] = df[var].astype(str).astype('category')
    
    # Check if need to construct dictionary for scaling_values and list for category_labels. 
    # If so, these variables are initialized as empty dictionary and empty list. 
    need_catlabels = False
    if category_labels == None: 
        category_labels = []
        need_catlabels = True    
    if len(xcont_vars) > 0 and scale_cont == 'by_df': scaling_values = {}        
    elif len(xcont_vars) > 0 and type(scale_cont) == dict: scaling_values = scale_cont    
    else: scaling_values = None
   
    # Output variable y
    if output_var == None: 
        y = None    
    elif output_var in cont_vars: 
        y = np.array(df[output_var])
    elif output_var in cat_vars:
        if need_catlabels == True:
            y_cats = df[output_var].unique()
            y_cat_labels = {y_cats[i]:i for i in range(len(y_cats))} 
            y = np.array( df[output_var].cat.rename_categories(y_cat_labels) ).astype('int64')
        else:
            y = np.array( df[output_var].cat.rename_categories(category_labels[-1]) ).astype('int64')
    
    # Construct xcat_df. 
    # (Along the way, the list category_labels also built if necessary.)
    if len(xcat_vars) > 0: 
        xcat_df = df.reindex(columns=xcat_vars) 
        for j,var in enumerate(xcat_vars):
            if need_catlabels == True and unknown_category == True:
                var_cats = list(xcat_df[var].cat.categories)
                if 'nan' in var_cats: var_cats.remove('nan') 
                Dict = {var_cats[i]:i+1 for i in range(len(var_cats))} 
                Dict['unknown'] = 0
                category_labels.append(Dict)
            elif need_catlabels == True and unknown_category == False:
                var_cats = list(xcat_df[var].cat.categories)
                Dict = {var_cats[i]:i for i in range(len(var_cats))} 
                category_labels.append(Dict)
            else: 
                Dict = category_labels[j]
        
            if unknown_category == True:
                xcat_df[var] = xcat_df[var].cat.add_categories(['unknown'])
                for value in xcat_df[var].unique():
                    if value not in Dict: xcat_df[var].replace(value,'unknown',inplace=True)
            
            xcat_df[var] = xcat_df[var].astype(str).astype('category') #to avoid issue with rename_categories method
            xcat_df[var] = xcat_df[var].cat.rename_categories(Dict)
            xcat_df[var] = xcat_df[var].astype('int64') 
    
    else: xcat_df = None
        
    if (need_catlabels == True) and (output_var in cat_vars): 
        category_labels.append(y_cat_labels)
        
    # Construct xcont_df. 
    # (Along the way, the dictionary scaling_values also built if necessary.) 
    if len(xcont_vars) > 0:
        xcont_df = df.reindex(columns=xcont_vars)
        if fill_missing == 'median': 
            xcont_df = xcont_df.fillna(xcont_df.median())
        elif fill_missing == 'mean': 
            xcont_df = xcont_df.fillna(xcont_df.mean())
        else:
            filler = pd.Series(fill_missing,index = xcont_vars)
            xcont_df = xcont_df.fillna(filler)     
        if scale_cont == 'by_df':
            for var in xcont_vars:
                mean = xcont_df[var].mean()
                std = xcont_df[var].std()
                xcont_df[var] = (xcont_df[var] - mean)/std
                scaling_values[var] = [mean,std]
        elif type(scale_cont) == dict:
            for var in xcont_vars:
                mean = scale_cont[var][0]
                std = scale_cont[var][1]
                xcont_df[var] = (xcont_df[var] - mean)/std
                
    else: xcont_df = None 
    
    # Return the output
    return xcat_df, xcont_df, y, scaling_values, category_labels

class StructuredDataset(Dataset):
    
    """
    A class for a dataset of structured data (can be either train, val, or test).  
    
    Arguments for initialization:
    xcat_df, xcont_df, y: these are the outputs of function ProcessDataFrame. 
    target_type: type of the output variable, either 'cont' or 'cat'. 
    
    Attributes:
    target_type: Same as input.
    n_cat, n_cont: number of categorical and continuous input variables.
    x_cat, x_cont, y: Same as inputs xcat_df, xcont_df, y except that x_cat and x_cont 
                      are in form of numpy arrays. If any of these inputs is None, 
                      corresponding attribute is an array of zeros.    
    """
    
    def __init__(self,xcat_df,xcont_df,y,target_type): 
        
        self.target_type = target_type
        L = len(xcat_df) if (xcat_df is not None) else len(xcont_df)
        self.y = y if (y is not None) else np.zeros(L).astype('float32')
        
        if xcat_df is not None: 
            self.n_cat = xcat_df.shape[1] 
            self.x_cat = np.array(xcat_df)
        else:
            self.n_cat = 0
            self.x_cat = np.zeros((L,1),'int64')
        
        if xcont_df is not None: 
            self.n_cont = xcont_df.shape[1]
            self.x_cont = np.array(xcont_df)
        else:
            self.n_cont = 0
            self.x_cont = np.zeros((L,1),'float32')
                        
    def __len__(self):
        return len(self.x_cat)
 
    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_cont[idx], self.y[idx]
    
    def y_range(self):
        return [np.min(self.y),np.max(self.y)]
        
def StructuredDataCollater(batch): 
    
    """
    Collate function for batch of items returned from StructuredDataset, 
    using __getitem__(idx) function.
    
    Input: 'batch' is a list of elements, each of form, (x_cat[idx], x_cont[idx], y[idx]). 
    Output: returns [xcat_batch,xcont_batch], y_batch where:
            * xcat_batch is a (bs x n_cat') torch.LongTensor
            * xcont_batch is a (bs x n_cont') torch.FloatTensor
            * ybatch is a 1d torch.FloatTensor/torch.LongTensor for, resprectively, 
              a cont/cat output variable. 
    
    Here n_cat' = max(self.n_cat,1) where self.n_cat is defined as in StructuredDataset.
         n_cont' = max(self.n_cont,1) where self.n_cont is defined as in StructuredDataset. 
    """
    
    xcat = TEN( np.array([z[0] for z in batch]), GPU=False )
    xcont = TEN( np.array([z[1] for z in batch]), GPU=False )
    y = TEN( np.array([z[2] for z in batch]), GPU=False )
    return [xcat,xcont], y
        
class StructuredDataObj(object):
    
    """ Class for a structured data object encompassing the datasets and corresponding dataloaders 
    for train, validation, and (optionally) test data, along with a bit of extra information. 
    
    Arguments for initialization:
    train_ds: training dataset of class StructuredDataset 
    val_ds: validation dataset of class StructuredDataset
    test_ds (optional): test dataset of class StructuredDataset 
    category_labels: output of function ProcessDataFrame (applied to original training DataFrame <df>) 
    scaling_values: output of function ProcessDataFrame (applied to original training DataFrame <df>) 
    bs: the batch size to use for dataloaders 
    num_workers (optional): numper of CPU's to use in parrallel for data loading
    
    Attributes:
    target_type: type of the output variable, either 'cont' or 'cat'.
    train_ds, val_ds, test_ds, category_labels, scaling_values, bs, num_workers: all exact same as inputs.
    train_dl, val_dl, test_dl: dataloaders for train, val, and test datasets. 
    """    
        
    def __init__(self, train_ds, val_ds, category_labels, scaling_values,
                 bs, num_workers=6, test_ds = None):
        
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.category_labels = category_labels
        self.scaling_values = scaling_values
        self.bs = bs
        self.num_workers = num_workers
        self.target_type = train_ds.target_type
        
        self.train_dl = DataLoader(train_ds, batch_size=bs, collate_fn=StructuredDataCollater,
                                   num_workers=num_workers, shuffle=True, pin_memory=True)
        
        self.val_dl = DataLoader(val_ds, batch_size=bs, collate_fn=StructuredDataCollater, 
                                 num_workers=num_workers, shuffle=False, pin_memory=True)
        
        if self.test_ds:
            self.test_dl = DataLoader(test_ds, batch_size=bs, collate_fn=StructuredDataCollater,
                                      num_workers=num_workers, shuffle=False, pin_memory=True)
        
    @classmethod
    def from_dataframes(cls, train_df, val_df, cat_vars, cont_vars, output_var, bs,
                        fill_missing = 'median', scale_cont = True, unknown_category = True,
                        num_workers = 6, test_df = None):
        
        if output_var in cat_vars: target_type = 'cat'
        if output_var in cont_vars: target_type = 'cont'
            
        if scale_cont == True:
            
            xcat_df, xcont_df, y, scaling_values, category_labels = \
            ProcessDataFrame(train_df, cat_vars, cont_vars, output_var,'by_df', 
                             fill_missing, None, unknown_category)
            train_ds = StructuredDataset(xcat_df,xcont_df,y,target_type)
            
            xcat_df, xcont_df, y, scaling_values, category_labels = \
            ProcessDataFrame(val_df, cat_vars, cont_vars, output_var, scaling_values, 
                             fill_missing, category_labels, unknown_category)
            val_ds = StructuredDataset(xcat_df,xcont_df,y,target_type)
            
            if type(test_df) == pd.DataFrame:
                xcat_vars, xcont_vars = cat_vars.copy(), cont_vars.copy()
                if output_var in cat_vars: xcat_vars.remove(output_var)
                if output_var in cont_vars: xcont_vars.remove(output_var)
                xcat_df, xcont_df, y, scaling_values, category_labels = \
                ProcessDataFrame(test_df, xcat_vars, xcont_vars, None, scaling_values, 
                                 fill_missing, category_labels, unknown_category)
                test_ds = StructuredDataset(xcat_df,xcont_df,y,target_type)
            else: test_ds = None            
        
        if scale_cont == False:
            
            xcat_df, xcont_df, y, scaling_values, category_labels = \
            ProcessDataFrame(train_df, cat_vars, cont_vars, output_var, 'No', 
                             fill_missing, None, unknown_category)
            train_ds = StructuredDataset(xcat_df,xcont_df,y,target_type)
            
            xcat_df, xcont_df, y, scaling_values, category_labels = \
            ProcessDataFrame(val_df, cat_vars, cont_vars, output_var, 'No', 
                             fill_missing, category_labels, unknown_category)
            val_ds = StructuredDataset(xcat_df,xcont_df,y,target_type)
            
            if type(test_df) == pd.DataFrame:
                xcat_vars, xcont_vars = cat_vars.copy(), cont_vars.copy()
                if output_var in cat_vars: xcat_vars.remove(output_var)
                if output_var in cont_vars: xcont_vars.remove(output_var)
                xcat_df, xcont_df, y, scaling_values, category_labels = \
                ProcessDataFrame(test_df, xcat_vars, xcont_vars, None, 'No', 
                                 fill_missing, category_labels, unknown_category)
                test_ds = StructuredDataset(xcat_df,xcont_df,y,target_type)             
            else: test_ds = None
                
        return cls(train_ds, val_ds, category_labels, scaling_values, bs, num_workers, test_ds)


# (2.2) - Models

def embedding_dim(n):
    "Returns a 'reasonable' embedding dimension d, for an embedding of n classes."
    if n>=2 and n<=8: return int(np.ceil(n/2))
    elif n>=9 and n<=12: return 5
    elif n>=13 and n<=18: return 6
    elif n>=19 and n<=27: return 7
    elif n>=28 and n<=100 : return int(np.ceil(n/4))
    elif n>100: return 25 
    
class StructuredDataNet(nn.Module):
    
    """ Class for a pytorch neural network model to learn from structured data.
    
    Embeddings are done for categorical variables, and then output of these embeddings
    combined with values of continuous variables is passed through a (user specified)
    fully connected network of arbitrary depth. 
    
    Arguments for Initialization:
    
    n_cat: The number of categorical input variables.
    n_cont: The number of continuous input variables.
    
    category_labels: output of function ProcessDataFrame (applied to original training DataFrame <df>) 
    
    emb_sizes: A list of tuples of form (c,d) where the ith tuple has:
               c = number of categories of ith input categorical variable.
               d = dimension of the embedding of ith input categorical variable. 
               (Ordering of categorical variables is same as in xcat_df) 
               
               If <emb_sizes> == 'default', the function embedding_dim() is used to 
               calculate all embedding dimensions.
                   
    fc_layer_sizes: A list specifying the sizes of the fully connected layers. 
                    For example if fc_layer_sizes = [50,20,10] then:
                    fc_layer1 has N inputs, 50 outputs
                    fc_layer2 has 50 inputs, 20 outpus
                    fc_layer3 has 20 inputs, 10 outputs
                    10 output activations from the network
                 
                    Here N = (sum of embedding dimesions of all categorical variables) + n_cont.
                 
    dropout_levels: A tuple (emb_drop, cont_drop, other_drop) where:
                    - emb_drop is the dropout prob to use for embeddings.
                    - cont_drop is the dropout prob to use for each continuous input variable.
                    - other_drop = [d1=0,d2,d3,...d_n] with n = len(fc_layer_sizes).
                      d_i = dropout prob applied BEFORE passing through ith linear layer of fully connected network.
                      Note: d1 = 0, since dropout seperately applied to embedding and continuous variables,
                            before passing through 1st linear layer.
                            
                    Continuing example above, if dropout_levels = (0.01,0.02,[0,0.3,0.1]) then
                    dropout of 0.01 applied to embeddings
                    dropout of 0.02 applied to each continuous input variable 
                            (i.e. each such variable set to 0, with prob 0.02)
                    0 additional dropout before 1st linear layer
                    dropout of 0.3 before 2nd linear layer
                    dropout of 0.1 before 3rd linear layer 
                    
                    Default is dropuout_levels = None, which is interpreted same as (0,0,[0,...,0]).
                          
    target_type: The type of the output variable, either 'cont' or 'cat'. 
   
    output_range: * If target_type == 'cont' and you want to ensure the network only returns values
                    of the output variable in the range [a,b], then set output_range = [a,b].
                  * If target_type == 'cat' OR target_type == 'cont', but you do not want to ensure
                    the output variable stays within a prespecified range, then leave as default 
                    output_range = None. 
    """

    def __init__(self, target_type, n_cat, n_cont, category_labels, fc_layer_sizes, 
                 emb_sizes = 'default', output_range=None, dropout_levels = None):
        
        super().__init__()
        self.n_cat, self.n_cont = n_cat, n_cont
        if dropout_levels == None: dropout_levels = (0,0,None)
        
        # define batchnorm and dropout for continuous inputs 
        self.cont_bn = nn.BatchNorm1d(n_cont)
        self.cont_drop = nn.Dropout(dropout_levels[1])
        
        # define embeddings
        if emb_sizes == 'default': 
            if target_type == 'cont': cat_sizes = [len(Dict) for Dict in category_labels]
            else: cat_sizes = [len(Dict) for Dict in category_labels[0:-1]] 
            emb_sizes = [(c,embedding_dim(c)) for c in cat_sizes]
        self.embeddings = nn.ModuleList([EmbeddingDrop(c,d,dropout_levels[0],std=1/d**0.5,max_norm=1.5) for c,d in emb_sizes])
        
        #define fully connected part of the network (called 'head')
        total_emb_dim = sum(d for c,d in emb_sizes)
        layer_sizes = [total_emb_dim + n_cont] + fc_layer_sizes 
        if target_type == 'cont' and output_range: final_activ = 'sigmoidal'
        else: final_activ = None
            
        if target_type == 'cat':    
            self.head = FullyConnectedNet(layer_sizes, dropout_levels[2], final_activ, output_range, pre_bn=False)             
        elif target_type == 'cont':
            fc = FullyConnectedNet(layer_sizes, dropout_levels[2], final_activ, output_range, pre_bn=False)
            self.head = nn.Sequential(fc, Flatten1d())
        
        # define layer_groups and param_groups
        self.layer_groups = [nn.ModuleList([self.embeddings,self.cont_bn,self.cont_drop]),self.head]
        self.param_groups = separate_bn_layers(self.layer_groups)
                        
    def forward(self, xcat_batch, xcont_batch):
        
        if self.n_cat > 0:
            cat_inputs = [emb(xcat_batch[:,i]) for i,emb in enumerate(self.embeddings)]                 
            cat_inputs = torch.cat(cat_inputs,dim=1)
        if self.n_cont > 0:
            cont_inputs = self.cont_drop(self.cont_bn(xcont_batch))
                    
        if self.n_cat == 0: combined_inputs = cont_inputs
        elif self.n_cont == 0: combined_inputs = cat_inputs
        else: combined_inputs = torch.cat([cat_inputs, cont_inputs], dim=1)
        
        return(self.head(combined_inputs))
    
    @classmethod
    def from_dataobj(cls, data, fc_layer_sizes, emb_sizes='default', 
                     output_range=None, dropout_levels=None):
        
        category_labels = data.category_labels
        target_type = data.target_type
        n_cat = data.train_ds.n_cat
        n_cont = data.train_ds.n_cont
        
        return cls(target_type, n_cat, n_cont, category_labels, fc_layer_sizes, 
                   emb_sizes, output_range, dropout_levels)
    
class StructuredDataEnsembleNet(nn.Module):
    
    """Class for an ensemble model for structured data.
    
       Arguments for Initilization:
       
       models: A list of models, all trained on same structured data dataset. 
               Models do not have to have same architecture.           
       weights: A list of weights for averaging the outputs of models, should sum to 1. 
                If weights is None, all model weights are equal.          
       correction: 'cat' or None. If 'cat', softmax is applied to each model's
                    outputs before averaging, for use with target_type = 'cat'.
       
       Output:
       Returns a single model, which given an input x returns a weighted 
       average of the outputs of the individual models (with softmax correction 
       of 'cat' outputs if specified.)
       
       """
    
    def __init__(self,models,weights=None,correction=None):
        
        super().__init__()        
        n = len(models)
        if weights: self.weights = weights
        else: self.weights = [1/n]*n
        self.correction = correction
        self.models = nn.ModuleList(models)
        self.layer_groups = models
        self.param_groups = separate_bn_layers(self.layer_groups)
        
    def forward(self,xcat,xcont):
        if self.correction is None:
            return sum(self.weights[i]*m(xcat,xcont) for i,m in enumerate(self.models))
        elif self.correction == 'cat':
            return sum(self.weights[i]*F.log_softmax(m(xcat,xcont),dim=1).exp() for i,m in enumerate(self.models))
        