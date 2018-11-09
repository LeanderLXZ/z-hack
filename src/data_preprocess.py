import time
import pandas as pd


# Load CSV Files Using Pandas
def load_csv(file_path):

    f = pd.read_csv(file_path, index_col='FDATE', parse_dates=['FDATE'],
                    usecols=['FDATE', 'CONTNUM', 'CONTPRICE'])
    f['VOLUME'] = f['CONTNUM'] * f['CONTPRICE']

    return f


def preprocess(df, mode):

    if mode == 'd':
        suffix = 'DAY'
        df_grouped = df.groupby(df.index)
    elif mode == 'm':
        suffix = 'MONTH'
        df_grouped = df.groupby([lambda x: x.month, lambda x: x.year])
    elif mode == 'y':
        suffix = 'YEAR'
        df_grouped = df.groupby(lambda x: x.year)
    else:
        raise ValueError

    df_preprocessed = pd.DataFrame()
    df_preprocessed['CONTNUM_' + suffix] = df_grouped['CONTNUM'].sum()
    df_preprocessed['VOLUME_' + suffix] = df_grouped['VOLUME'].sum()
    df_preprocessed['CONTPRICE_' + suffix] = df_grouped['CONTPRICE'].mean()

    if mode == 'd':
        df_preprocessed.index = df_preprocessed.index.rename('DAY')
    if mode == 'm':
        df_preprocessed.index = df_preprocessed.index.rename(['MONTH', 'YEAR'])
    if mode == 'y':
        df_preprocessed.index = df_preprocessed.index.rename('YEAR')

    return df_preprocessed


if __name__ == '__main__':

    time_ = time.time()

    df_day = []
    df_month = []
    df_year = []

    for year in ['2009', '2010', '2011', '2012', '2013']:
        df_loaded = load_csv(
            '../data/source_data/z_hack_transaction_{}.csv'.format(year))
        df_day.append(preprocess(df_loaded, 'd'))
        df_month.append(preprocess(df_loaded, 'm'))
        df_year.append(preprocess(df_loaded, 'y'))

    df_total_day: pd.DataFrame = pd.concat(df_day)
    df_total_month: pd.DataFrame = pd.concat(df_month)
    df_total_year: pd.DataFrame = pd.concat(df_year)

    df_total_day.to_csv('../data/source_data/total_day.csv')
    df_total_month.to_csv('../data/source_data/total_month.csv')
    df_total_year.to_csv('../data/source_data/total_year.csv')

    print('Finished! Using time: {:.2f}s'.format( time.time() - time_))
