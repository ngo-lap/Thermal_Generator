import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup


def get_df_from_epex(start: pd.Timestamp, country='FR'):
    """This will take the start date timestamp, returns a dataframe of the week that ends in that day along with the
    time stamp for the next day in the past (the day start - 8)."""

    # Get the url
    url_init = 'https://www.epexspot.com/en/market-data/dayaheadauction/auction-table/'
    url_var = start.strftime('%Y-%m-%d')
    url_end = '/FR'
    url = url_init + url_var + url_end

    # Get the response :
    page = requests.get(url)

    # BeautifulSoup object :
    soup = BeautifulSoup(page.content, 'html.parser')

    # Use the selector
    soup_list_1 = soup.select('table.list.hours.responsive tr.no-border')

    # Data for France, Originally, we expected to find only 24 'tr'-type objects representing 24 rows of data per day,
    # but we got 72 instead. This is because the page also includes data for 2 additional countries (Germany & Switzerland).
    # All 3 countries' data are put in different tabs so that at the first glance we cant observe it

    if country == 'FR':
        country_tag = soup_list_1[:24]
    elif country == 'DE/AT':
        country_tag = soup_list_1[24:24*2]
    elif country == 'CH':
        country_tag = soup_list_1[24*2:]
    else:
        raise ValueError('The country code is not recognized: must be \'FR\', \'DE/AT\' or \'CH\' ')

    country_dict = {}

    for hour in country_tag:
        ex = hour.find_all('td')
        ex_list = [elem.text for elem in ex]
        name_hour = ex_list[0].replace(" ", "")[1:3]  # strip the whitespaces, take the first number as the hour column
        country_dict[name_hour] = ex_list[:-8:-1]  # Be careful with the \n commands

    week_interval = pd.DatetimeIndex(start=start, freq='-1D', periods=7)
    epex_tempo = pd.DataFrame(index=week_interval, data=country_dict)
    next_day_stamp = (week_interval[-1] - pd.to_timedelta(1, unit='D'))

    return epex_tempo, next_day_stamp


def process_datetime_df(df_agg: pd.DataFrame):

    """This function further process the datetime index series. Instead of only the date, now it will be up to hour."""
    df_melted = pd.melt(df_agg.reset_index(), id_vars='index', var_name='Hour', value_name='DAM')
    df_melted['date_str'] = df_melted['index'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df_melted['date_str'] = df_melted['date_str'] + ' ' + df_melted.Hour
    df_melted['Time'] = pd.to_datetime(df_melted['date_str'], format='%Y-%m-%d %H')
    df_melted.index = df_melted['Time']
    df_melted.drop(['index', 'date_str', 'Time', 'Hour'], axis=1, inplace=True)
    df_melted.sort_index(inplace=True)

    # Process the invalid data points
    df_melted.fillna(method='ffill', inplace=True)
    df_melted[df_melted['DAM'].str.contains('–')] = np.nan
    df_melted.fillna(method='ffill', inplace=True)
    df_melted = df_melted.astype('float')

    return df_melted


def epex_date(start: pd.Timestamp, country='FR', nbr_weeks=3):
    """This function will perform nbr_weeks times the get_df_from_epex function and returns the aggregate dataframe """
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup

    df_list = []
    df_agg = pd.DataFrame()

    for t in range(nbr_weeks):

        df, next_stamp = get_df_from_epex(start=start, country=country)
        df_list.append(df)
        start = next_stamp
        df_agg = df_agg.append(df)

    df_agg_hour = process_datetime_df(df_agg)

    return df_agg, df_agg_hour


if __name__ == '__main__':

    start_date = pd.Timestamp('2015-12-31')
    df, df_hour = epex_date(start=start_date, country='DE/AT', nbr_weeks=52)
    df_hour.info()
