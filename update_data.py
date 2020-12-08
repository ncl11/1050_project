import sched, time
from datetime import date, timedelta, datetime
import pandas as pd
from bs4 import BeautifulSoup
import requests
import gtab
import numpy as np
import datetime
import logging
from io import StringIO
import utils

logger = logging.Logger(__name__)
utils.setup_logger(logger, 'data.log')

DOWNLOAD_PERIOD = 604800
#DOWNLOAD_PERIOD = 600

def add_target(df_weekly):
    '''Matches a row with the next weeks 'Weekly' columns which is target variable.'''  
    df_temp = pd.read_csv('3_mo_weekly.csv', sep='\t')
    df_weekly = df_temp.append(df_weekly, ignore_index=True)
    df_weekly['Date'] = df_weekly['Date'].astype('datetime64[ns]')
    #from datetime import timedelta
    df_weekly['Y'] = df_weekly['Date'].apply(lambda x: x + timedelta(days=7))
    df_weekly['Week + 1'] = pd.Series(np.zeros(df_weekly.shape[0]))
    for movie in df_weekly['Release'].unique():
        for date in df_weekly[df_weekly['Release'] == movie]['Date']:
            df_weekly.loc[(df_weekly['Release'] == movie) & (df_weekly['Y'] == date), 'Week + 1'] = float(df_weekly.loc[(df_weekly['Release'] == movie) & (df_weekly['Date'] == date)]['Weekly'])
            df_weekly['Week + 1'].fillna(0, inplace=True)
    return df_weekly


def to_weekly(df):
    '''Combines daily into weekly data, divides by 7 where appropriate, removes data points with 'Weekly' == 0'''
        #convert date column into datetime object
    df['Date'] = df['Date'].astype('datetime64[ns]')
    #convert daily data to weekly
    df_weekly = df.groupby("Release").resample('W-Mon', label='left', closed = 'left', on='Date').sum().reset_index().sort_values(by='Date')
    df_weekly['Avg TD'] = df_weekly['TD'].apply(lambda x: x/7)
    df_weekly = df_weekly.drop('TD', axis=1)
    df_weekly['Avg YD'] = df_weekly['YD'].apply(lambda x: x/7)
    df_weekly = df_weekly.drop('YD', axis=1)
    df_weekly['Weekly'] = df_weekly['Daily']  
    df_weekly = df_weekly.drop('Daily', axis=1)
    df_weekly['Weekly %+-YD'] = df_weekly['%+-YD'] 
    df_weekly['Weekly %+-LW'] = df_weekly['%+-LW'] 
    df_weekly['Avg Theatre'] = df_weekly['Theatre'].apply(lambda x: x/7)
    df_weekly = df_weekly.drop('Theatre', axis=1)
    df_weekly['Avg per Theatre Avg'] = df_weekly['Avg'].apply(lambda x: x/7)
    df_weekly = df_weekly.drop('Avg', axis=1)
    df_weekly['Avg To Date'] = df_weekly['To Date'].apply(lambda x: x/7)
    df_weekly = df_weekly.drop('To Date', axis=1)
    df_weekly = df_weekly[df_weekly.Weekly != 0]
    
    return df_weekly
    

def edit_df(df):
    """Removes $,%,- and converts str to int or float"""
    df['Daily'] = df.Daily.apply(lambda x: x.strip('$'))
    df['%+-YD'] = df['%+-YD'].apply(lambda x: x.strip('%'))
    df['%+-LW'] = df['%+-LW'].apply(lambda x: x.strip('%'))
    df['Avg'] = df['Avg'].apply(lambda x: x.strip('$'))
    df['To Date'] = df['To Date'].apply(lambda x: x.strip('$'))
    df['Distributor'] = df.Distributor.apply(lambda x: x[0:-2])

    df = df.replace(',','', regex=True)
    df = df.replace('-', '0') #SHOUOLD BE NAN
    df = df.replace('<0.1', '0')
    
    df['Theatre'] = df['Theatre'].astype(float)
    df['Days'] = df['Days'].astype(float)
    df['Daily'] = df.Daily.astype(float)
    df['%+-YD'] = df['%+-YD'].astype(float)
    df['%+-LW'] = df['%+-LW'].astype(float)
    df['Avg'] = df['Avg'].astype(float)
    df['To Date'] = df['To Date'].astype(float)
    df['TD'] = df['TD'].astype(float)
    df['YD'] = df['YD'].astype(float)
    return df

def google_trends(df, d1, d2):
    """Pulls movie data from google trends and merges"""
    df.loc[:, 'google trends'] = 0
    for movie in df['Release'].unique():
        try:
            t = gtab.GTAB();
            t.set_options(pytrends_config={"timeframe": f"{str(d1)} {str(d2)}"}); 
            query = t.new_query(movie);
            #print(query)
            for date in query.index:
                print(query.loc[date, 'max_ratio'])
                df.loc[df['Release'].eq(movie) & df['Date'].eq(date), 'google trends'] = query.loc[date, 'max_ratio']
        except:
            continue
    return df

def get_data():
    '''Scrapes data from boxofficemojo with beautiful soup'''
    
    d2 = datetime.date.today() - timedelta(days=1)
    d1 = d2 - timedelta(days=6)

    dd = [d1 + timedelta(days=x) for x in range((d2-d1).days + 1)][::-1]
    rows = 0
    df = pd.DataFrame(columns = ['Date','TD', 'YD', 'Release', 'Daily', '%+-YD', '%+-LW', 'Theatre', 'Avg', 'To Date', 'Days', 'Distributor'])

    for date in dd:
        try:
            source = requests.get('https://www.boxofficemojo.com/date/'+str(date)+'/').text
            soup = BeautifulSoup(source, 'lxml')
            table = soup.find('table')
            data = table.find_all('tr')
            master_list = []
            for row in data:
                row_list = [date]
                for entry in row.find_all('td'):
                    if entry.text == 'false' or entry.text == 'true':
                        continue
                    row_list.append(entry.text)

                if len(row_list) == 12:
                    master_list.append(row_list)

            for i in range(len(master_list)):
                df.loc[rows] = master_list[i]
                rows += 1
        except:
            continue
            
    df = google_trends(df, d1, d2)
    df = edit_df(df) 
    df_weekly = to_weekly(df)
    df_weekly = add_target(df_weekly)

        
    df_weekly['Date'] = df_weekly['Date'].astype('datetime64[ns]')
    df_weekly.to_csv('3_mo_weekly.csv', index=False, sep='\t')
    print('DONE')
    
    
def main_loop(timeout=DOWNLOAD_PERIOD):
    scheduler = sched.scheduler(time.time, time.sleep)
    print('WORKING . . . ')

    def _worker():
        try:
            get_data()
        except Exception as e:
            logger.warning("main loop worker ignores exception and continues: {}".format(e))
        scheduler.enter(timeout, 1, _worker)    # schedule the next event

    scheduler.enter(0, 1, _worker)              # start the first event
    scheduler.run(blocking=True)


if __name__ == '__main__':    
    main_loop()