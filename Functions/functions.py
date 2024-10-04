import matplotlib
#General things:

import pandas as pd

def nb_of_fields (name_csv_file, delim):
    import csv
    fields=[]
    with open(name_csv_file, newline ='') as csvfile:
        data = csv.reader(csvfile, delimiter=delim)
        fields = next(data)
        min_counter = len(fields)
        max_counter = min_counter
        for row in data:
            nb_fields = len (row)
            if nb_fields > max_counter:
                max_counter = nb_fields
            elif nb_fields < min_counter:
                min_counter = nb_fields
    return([min_counter, max_counter])

def Zscore_norm(col):
    mean = col.mean()
    std = col.std()
    return (col - mean) / std

def normalize_values(df, norm_func):
    """Normalize only the numeric columns using the specified normalization function."""
    df_norm = pd.DataFrame()
    for column in df:
        if pd.api.types.is_numeric_dtype(df[column]):  # Check if the column is numeric
            df_norm[column] = norm_func(df[column])
        else:
            df_norm[column] = df[column]  # Leave non-numeric columns as they are
    return df_norm


def determine_country(website):
    part1=substrings=website.split('//')
    if len(part1)>1:
        part2=part1[1].split('/')
    else:
        part2=part1[0].split('/')
    part3=part2[0].split('.')
    return part3[-1]

#Make an array of all the values in the given table, respecting the frequencies
def repeated_array(array):
    import numpy as np
    result = np.repeat(array[:,0],array[:,1])
    return result

# Frequencies
def all_freq(x):
    t_abs = x.value_counts(dropna=False).sort_index()
    t_rel = (x.value_counts(dropna=False, normalize=True).sort_index()*100).round(1)
    t_abs_cum = x.value_counts(dropna=False).sort_index().cumsum()
    t_rel_cum = (x.value_counts(dropna=False, normalize=True).sort_index().cumsum()*100).round(1)
    return pd.DataFrame({'abs freq':t_abs,'rel freq':t_rel,'abs cum freq':t_abs_cum,'relcum freq':t_rel_cum})

def nb_of_classes(x):
    import math
    import statistics as stat
    x=x.dropna()
    n=len(x)
    sturges=math.ceil(1+math.log2(n)) # Sturges
    print('Sturges:',sturges)
    b=3.5*stat.stdev(x)/(n**(1/3))
    scott=math.ceil((x.max()-x.min())/b) # Scott
    print('Scott:',scott)
    sqrt=math.ceil(math.sqrt(n))
    print('Sqrt(n):',sqrt)

#Graphs
def spider_graph(column):
    import matplotlib.pyplot as plt
    import math

    x=column
    t=x.value_counts()
    categories= t.index
    values= t.values.tolist()
    values +=values[:1] #add end point equal to start point
    n= len (t)
    m =max(values)
    angles =[k/float(n)*2* math.pi for k in range(n)]
    angles+= angles[: 1]

    plt. figure()
    ax=plt.subplot( 111,polar=True)
    plt.xticks(angles[:-1],categories, color ='grey', size=8)
    ax.set_rlabel_position (0)
    plt.yticks([ k/4*m for k in range(4)],[ k/4*m for k in range(4)],color='grey',size=7)
    plt.ylim (0,m)
    plt.plot ( angles,values, linewidth=1,linestyle='solid')

    plt.fill ( angles,values ,'b',alpha=0.1)
    plt.show()
    return


#Mean,average etc

import math as m
def median_categorical(data):
    d = data.dropna()
    n = len(d)
    middle = m.floor(n/2)
    return d.sort_values().reset_index(drop=True)[middle]


# Weighted mean
def weighted_mean(credits, score):
    return sum(credits*score)/sum(credits)

# Geometric mean
def geo_mean(delta):
    import numpy as np
    return np.exp(np.mean(np.log(delta)))

# Harmonic mean
def harmonic_mean(data):
    from scipy import stats
    return stats.hmean(data)



## Functie care face toate mediile

def central(column):
    data = column.dropna()
    mode = data.mode()
    med = data.median()
    mean = data.mean()
    geo = geo_mean(data)
    harm = harmonic_mean(data)
    return pd.DataFrame({'Mode':mode,'Median':med,'Mean':mean.round(2),'G-mean':geo.round(2),'H-mean': harm.round(2)})

#Calculates the mode for a given array with scores and freq

def mode(vector):
    import numpy as np
    maximum=np.max(vector[:,1])
    for i in range(0,len(vector)):
        if vector[i,1]==maximum:
            mode = vector[i][0]
    return mode

#Calculates the median for a given array with scores and freq
def median(vector):
    import numpy as np
    # Repeat each health score based on the number of times it appears in the array
    repeated_scores = np.repeat(vector[:,0], vector[:,1])
    # Calculate the median of the repeated scores
    return np.median(repeated_scores)

#Calculates the mean for a given array with scores and freq

def calculate_average(vector):
    import numpy as np
    # Repeat each health score based on the number of times it appears in the array
    repeated_scores = np.repeat(vector[:,0], vector[:,1])
    # Calculate the median of the repeated scores
    return np.mean(repeated_scores)



## Get outliers funtion
def get_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    I = Q3 - Q1
    low = Q1 - 1.5 * I
    high = Q3 + 1.5 * I
    outliers = data[(data < low) | (data > high)]
    return outliers

def get_extreme_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    I = Q3 - Q1
    low = Q1 - 3 * I
    high = Q3 + 3 * I
    outliers = data[(data < low) | (data > high)]
    return outliers

def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    outliers_mask = ((df < low) | (df > high)).any(axis=1)
    df_outliers_removed = df.loc[~outliers_mask]
    return df_outliers_removed

def remove_outliers_column(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    outliers_mask = (df[column] < low) | (df[column] > high)
    df_outliers_removed = df.loc[~outliers_mask]
    return df_outliers_removed

def dispersion (series):
    s = series.dropna()
    import numpy as np
    if type(s[0]) == np.int32 or type(s[0]) == np.float32 or type(s[0]) == np.float64 or type(s[0]) == np.int64:
        from scipy import stats
        d_range = s.max() - s.min()
        d_IQR = s.quantile(0.75) - s.quantile(0.25)
        d_mad = s.mad()
        d_var = s.var()
        d_std = s.std()
        print(" Range IQR MAD Var std")
        print ('%5.2f' %d_range + ' ' + '%5.2f' %d_IQR + ' '
               +'%5.2f' %d_mad + ' ' +'%5.2f' %d_var + ' ' +'%5.2f' %d_std)
    else:
        print ('Range = ' + s.min() + ' ' + s.max())


# Coherence
#Avand formula dreptei de coordonate x si y, determin a si b
# y=a+bx etc..

def general_regression(x, y, degree=1, exp=False,log=False):
    import numpy as np
    import math
    data = pd.DataFrame({'x':x,'y':y})
    data.reset_index(drop=True, inplace=True)
    func=lambda x:x # def func(x): return[x]
    inv_func=lambda x:x
    if (exp):
        func=np.exp
        inv_func=np.log
    if(log):
        x=np.log(data['x'])
        data.x=x
        degree=1
    sy = data.y.std()
    model = np.polyfit(x, inv_func(y), degree)
    line = np.poly1d(model)
    predict = lambda x:func(line(x))
    data['y_pred'] = pd.Series(predict(x))
    se = math.sqrt(((data.y_pred-data.y)**2).mean())
    R2 = 1-(se**2)/(sy**2)
    result = [se, R2, predict]
    index = ['se', 'R2', 'predict']
    for i in range(1,len(model)+1):
        result = np.append(result, model[-i])
        index += chr(i+96) # to obtain the characters a,b,...
    result = pd.Series(result)
    result.index = index
    return result

def min_max_corr (corr_mat):
    n = corr_mat.shape[0]
    min= 1.0
    max = 0.0
    for i in range(0,n):
        for j in range(0,i):
            if (abs(corr_mat.iloc[i,j]) < abs(min)):
                min = corr_mat.iloc[i,j]
            if (abs(corr_mat.iloc[i,j]) > abs(max)):
                max = corr_mat.iloc[i,j]
    return min, max

#Datermine the best fit R2. The biggest one is the best one. If the R is 1, all the points will be on the line which is 100% fit.

def best_fit_Rsquare(x,y):
    import numpy as np
    overview=[]
    used_model=[]
    result_1=general_regression(x,y,1)
    overview=np.append(overview,result_1[1])
    used_model=np.append(used_model,'linear')
    result_2=general_regression(x,y,2)
    overview=np.append(overview,result_2[1])
    used_model=np.append(used_model,'quadratic')
    result_3=general_regression(x,y,3)
    overview=np.append(overview,result_3[1])
    used_model=np.append(used_model,'cubic')
    result_e=general_regression(x,y,exp=True)
    overview=np.append(overview,result_e[1])
    used_model=np.append(used_model,'Exponential')
    result_l = general_regression(x, y, log=True)
    overview=np.append(overview,result_l[1])
    used_model=np.append(used_model,'Logarithmic')
    overview=pd.Series(overview)
    overview.index=used_model
    return[overview]

# draw the graph with the line and the blue dots

def regression_complete(x,y,degree=1,linecol='red',errorcol='#FFFF0080',exp=False,log=False):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    plt.scatter(x,y)
    min=x.min()
    max=x.max()
    reg_result=general_regression(x,y,degree,exp,log)
    print(reg_result)
    se=reg_result.se
    predict=reg_result.predict
    xr=np.arange(min,max,(max-min)/100)
    if log:
        yr=predict(np.log(xr))
    else:
        yr=predict(xr)
    plt.fill_between(xr,yr-se,yr+se,color=errorcol)
    plt.plot(xr,yr,color=linecol)
    plt.show()

#Forcasting
import math
def naiveForecasting(past):
    if (len(past)<1):
        return math.nan
    return past[len(past)-1]

# Example of how to call the function
# forecast = naiveForecasting
# forecast(revenues)

def naiveForecastingTimes(past, amount):
    if len(past) < amount:
        return math.nan
    predicted = []
    forecast = naiveForecasting(past)
    for i in range(0, amount):
        next_val = forecast
        predicted.append(next_val)
        past = past + [next_val]  # Modified this line
    return predicted

def averageForecasting(past):
    if (len(past)<1):
        return math.nan
    return pd.Series(past).mean()


# Example of how to call the function
# forecast = averageForecasting
# forecast(revenues)


def movingAverageForecasting(period):
    def result(past):
        n = len(past)
        if (n < period):
            return math.nan
        return pd.Series(past[(n-period):n]).mean()
    return result

# Example of how to call the function
# forecast = movingAverageForecasting(4)
# forecast(revenues) #OR: movingAverageForecasting(4)(revenues)

def movingAverageForecastingMultipleDays(period):
    def result(past, forecast_days):
        n = len(past)
        if n < period:
            return [math.nan] * forecast_days
        forecasts = []
        for _ in range(forecast_days):
            average = pd.Series(past[(n - period):n]).mean()
            forecasts.append(average)
            past = pd.concat([past, pd.Series([average])], ignore_index=True)
            n += 1
        return forecasts
    return result
#This is how I call this function:
# movingAverage=f.movingAverageForecastingMultipleDays(5)
# print(movingAverage(df["downloads"]),3)



import numpy as np

# Solve this equation
# 181 = a1.82 + a2.176 + a3.282 + a4.445

###Linear Combination
def calculateWeights(period, past):
    n = len(past)
    if (n<2*period):
        return math.nan
    v = past[(n-2*period):(n-period)]
    for i in range(2,period+1):
        v = np.append(v, past[(n-2*period+i-1):(n-period+i-1)])
    M = np.array(v).reshape(period, period)
    v = past[(n-period):n]
    return np.linalg.solve(M, v)

def linearCombinationForecasting(period):
    def result(past):
        n = len(past)
        if (n<2*period):
            return math.nan
        a = calculateWeights(period,past)
        return (past[(n-period):n]*a).sum()
    return result

#How to apply
# >>> forecast = linearCombinationForecasting(4)
# >>> forecast(revenues)

def linearCombinationForecastingMultipleDays(period):
    def result(past, forecast_days):
        n = len(past)
        if n < 2 * period:
            return [math.nan] * forecast_days
        forecasts = []
        for _ in range(forecast_days):
            a = calculateWeights(period, past)
            forecast = (past[(n - period):n] * a).sum()
            forecasts.append(forecast)
            past = pd.concat([past, pd.Series([forecast])], ignore_index=True)
            n += 1
        return forecasts
    return result

def calculatePreviousForecasting(past, predictor):
    predicted = []
    n = len(past)
    for i in range(0,n):
        predicted = predicted + [predictor(past[0:i])]
    return predicted


def calculateErrors(past,forecast):
    predicted = calculatePreviousForecasting(past, forecast)
    errors = pd.Series(predicted) - past
    return errors

def modelsNaiveForecasting(df):
    mae=calculateErrors(df,naiveForecasting(df)).abs().mean()
    rmse=math.sqrt((calculateErrors(df,naiveForecasting(df))**2).mean())
    mape=(calculateErrors(df,naiveForecasting(df))/df).abs().mean()

def modelsAverageForecasting(df):
    mae=calculateErrors(df,averageForecasting(df)).abs().mean()
    rmse=math.sqrt((calculateErrors(df,averageForecasting(df))**2).mean())
    mape=(calculateErrors(df,averageForecasting(df))/df).abs().mean()

def modelTrend(df):
    predicted = pd.Series(trendForecastingModel(df))
    errors = predicted - df
    results = []
    MAE = errors.abs().mean()
    results.append(MAE)
    RMSE = math.sqrt((errors ** 2).mean())
    results.append(RMSE)
    MAPE = (errors / df).abs().mean()
    results.append(MAPE)
    metrics = ['MAE', 'RMSE', 'MAPE']
    # Create DataFrame with results
    df_results = pd.DataFrame([results], columns=metrics)

    return df_results

def reliability_overview(df, period):
    models = ['Naive', 'Average', 'Moving Average', 'Linear Combination','Trend Estimation']
    metrics = ['MAE', 'RMSE', 'MAPE']
    results = []

    # Naive Forecasting
    mae_naive = calculateErrors(df, naiveForecasting).abs().mean()
    rmse_naive = math.sqrt((calculateErrors(df, naiveForecasting) ** 2).mean())
    mape_naive = (calculateErrors(df, naiveForecasting) / df).abs().mean()
    results.append([mae_naive, rmse_naive, mape_naive])

    # Average Forecasting
    mae_avg = calculateErrors(df, averageForecasting).abs().mean()
    rmse_avg = math.sqrt((calculateErrors(df, averageForecasting) ** 2).mean())
    mape_avg = (calculateErrors(df, averageForecasting) / df).abs().mean()
    results.append([mae_avg, rmse_avg, mape_avg])

    # Moving Average Forecasting
    moving_avg = movingAverageForecasting(period)
    mae_ma = calculateErrors(df, moving_avg).abs().mean()
    rmse_ma = math.sqrt((calculateErrors(df, moving_avg) ** 2).mean())
    mape_ma = (calculateErrors(df, moving_avg) / df).abs().mean()
    results.append([mae_ma, rmse_ma, mape_ma])

    # Linear Combination Forecasting
    linear_combination = linearCombinationForecasting(period)
    mae_lc = calculateErrors(df, linear_combination).abs().mean()
    rmse_lc = math.sqrt((calculateErrors(df, linear_combination) ** 2).mean())
    mape_lc = (calculateErrors(df, linear_combination) / df).abs().mean()
    results.append([mae_lc, rmse_lc, mape_lc])

    # Trend estimation
    predicted = pd.Series(trendForecastingModel(df))
    errors = predicted - df
    MAE = errors.abs().mean()
    RMSE = math.sqrt((errors ** 2).mean())
    MAPE = (errors / df).abs().mean()
    results.append([MAE, RMSE, MAPE])

    # Create DataFrame with results
    df_results = pd.DataFrame(results, columns=metrics, index=models)

    return df_results

def seasonalDecompositionPredict(datapoints,period,model,time):
    import statsmodels.tsa.seasonal as smts
    result=smts.seasonal_decompose(datapoints,model=model,period=period)
    reg=general_regression(range(0,len(result.trend)-math.floor(period/2)*2),pd.Series(result.trend).dropna(),1)
    print(reg)
    trendFactor=reg.predict(time-math.floor(period/2))
    seasonalIndex=(time-1)%period
    seasonalFactor=result.seasonal[seasonalIndex]
    if model=='additive':
        forecast=trendFactor+seasonalFactor
    else:
        forecast=trendFactor*seasonalFactor
    return forecast

def seasonalDecompositionModel(df,model,period):
    import statsmodels.tsa.seasonal as smts
    result=smts.seasonal_decompose(df, model=model, period=period)
    predicted = result.trend + result.seasonal
    errors = predicted - df
    results = []
    MAE = errors.abs().mean()
    results.append(MAE)
    RMSE = math.sqrt((errors ** 2).mean())
    results.append(RMSE)
    MAPE = (errors / df).abs().mean()
    results.append(MAPE)
    metrics = ['MAE', 'RMSE', 'MAPE']
    # Create DataFrame with results
    df_results = pd.DataFrame([results], columns=metrics)

    return df_results

def smooth(x, period):
    result = []
    for i in range(0, len(x)-period+1):
        result = result + [np.mean(x[i:i+period])]
    return result

def findTrend(x, period):
    result = smooth(x, period)
    nan = [math.nan] * int(period/2)
    if (period % 2 == 0):
        result = smooth(result, 2)
    result = nan + result + nan
    return result


# def trendForecastingModel(past):
#     n = len(past)
#     x = pd.Series(range(0,n))
#     y = pd.Series(past)
#     reg = general_regression(x, y)
#     return reg

def trendForecastingModel(past):
    n = len(past)
    x = pd.Series(range(0, n))
    y = pd.Series(past)
    reg = general_regression(x, y)
    return reg.predict(x)


def autocorrelation(x):
    n = len(x)
    variance = np.var(x)
    x = x - np.mean(x)
    autocorr = np.correlate(x, x, mode='full')[-n:]
    autocorr /= (variance * np.arange(n, 0, -1))
    return autocorr


#Decision trees
def entropy(column: pd.Series, base=None):
    # Determine the fractions for all column values
    fracties = column.value_counts(normalize=True, sort=False)
    base = 2 if base is None else base
    return -(fracties * np.log(fracties) / np.log(base)).sum()

def information_gain(df: pd.DataFrame, s: str, target: str):
    # calculate entropy of parent table
    entropy_parent = entropy(df[target])
    child_entropies = []
    child_weights = []
    # compute entropies of child tables
    for (label, p) in df[s].value_counts().items():
        child_df = df[df[s] == label]
        child_entropies.append(entropy(child_df[target]))
        child_weights.append(int(p))
    # calculate the difference between parent entropy and weighted child entropies
    return entropy_parent - np.average(child_entropies, weights=child_weights)

# This is how you can call the function
# for label in stud.drop(labels='App', axis=1).columns:
#     print('{}: {}'.format(label, f.information_gain(stud, label, 'App')))

def best_split(df:pd.DataFrame, target: str):
    # retrieve all non target column labels (the features)
    features = df.drop(axis=1,labels=target).columns
    # calculate the information gains for these features
    gains = [information_gain(df,feature,target) for feature in features]
    # return column with highest information gain
    return features[np.argmax(gains)], max(gains)

# Example of how to call it
# print(f.best_split(stud,'App'))

#Example:
# entropy(simpsons.gender)
# 0.9910760598382221
# >>>
# for label in simpsons.drop(labels= gender', axis=1).columns:
# print('{}: {}'.format(label, information_gain(simpsons, label, 'gender')))
# hair length: 0.45165906291896163
# weight: 0.5900048960119098
# age: 0.07278022578373256
#      >>>
#      best_split(simpsons, gender)
#      ('weight', 0.5900..)

##PCA
def pca_visual(dataset, category_data):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    datasetZ = StandardScaler().fit_transform(dataset)
    pca_dim = min(datasetZ.shape[1], datasetZ.shape[0])
    pcamodel = PCA(n_components= pca_dim)
    principalComponents = pcamodel.fit_transform(datasetZ)
    #Visualize of the coefficient
    row_labels=['PC{}'.format(i) for i in range(1,pca_dim+1)]
    aij=pd.DataFrame(data=pcamodel.components_, columns=dataset.columns, index=row_labels)
    pd.set_option('display.max_columns', None) # to see all the columns
    print(aij)
    #visualize the explained variance
    print('Explained variance in %:', pcamodel.explained_variance_ratio_)
    labels_bar = ['PC{}'.format(i) for i in range(1,pca_dim+1)]
    plt.figure()
    plt.bar(labels_bar, pcamodel.explained_variance_ratio_)
    plt.title('PCA: Variance ratio')
    plt.xlabel('PCi')
    plt.ylabel('%')
    plt.show()
    #Graph of PC1 and PC2
    principalDf = pd.DataFrame()
    principalDf['PC1'] = principalComponents[:,0]
    principalDf['PC2'] = principalComponents[:,1]
    principalDf['cat'] = category_data
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('2-dim PCA')
    labels_plot = set(principalDf['cat'])
    markers = list(matplotlib.markers.MarkerStyle.markers.keys())
    for m, l in zip(markers, labels_plot):
        indices = np.where(principalDf['cat'] == l)[0] # return a tuple therefore [0]
        ax.scatter(principalDf.loc[indices, 'PC1'], principalDf.loc[indices, 'PC2'],
           marker=m, s=50)
    ax.legend(labels_plot)
    ax.grid(True)
    plt.show()