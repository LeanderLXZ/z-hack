import time
import utils
from os.path import join
import pandas as pd

source_path = '../data/source_data'
preprocessed_path = '../data/preprocessed_data'


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
    df_preprocessed['CONTPRICE_' + suffix] = \
        df_preprocessed['VOLUME_' + suffix] / \
        df_preprocessed['CONTNUM_' + suffix]
    # df_preprocessed['CONTPRICE_' + suffix + '_AVG'] = \
    #     df_grouped['CONTPRICE'].mean()

    if mode == 'd':
        df_preprocessed.index = df_preprocessed.index.rename('DAY')
    if mode == 'm':
        df_preprocessed.index = df_preprocessed.index.rename(['MONTH', 'YEAR'])
    if mode == 'y':
        df_preprocessed.index = df_preprocessed.index.rename('YEAR')

    return df_preprocessed


def output_rows(df):

    df = df.iloc


if __name__ == '__main__':

    time_ = time.time()

    utils.thick_line()
    print('Start preprocessing...')

    utils.check_dir([preprocessed_path])

    df_day = []
    df_month = []
    df_year = []

    for year in ['2009', '2010', '2011', '2012', '2013']:
        print('Preprocessing data in {}...'.format(year))
        df_loaded = load_csv(
            join(source_path, 'z_hack_transaction_{}_new.csv'.format(year)))
        df_day.append(preprocess(df_loaded, 'd'))
        df_month.append(preprocess(df_loaded, 'm'))
        df_year.append(preprocess(df_loaded, 'y'))

    print('Concatenating data...')
    df_total_day: pd.DataFrame = pd.concat(df_day)
    df_total_month: pd.DataFrame = pd.concat(df_month)
    df_total_year: pd.DataFrame = pd.concat(df_year)

    print('Writing data to CSV...')
    df_total_day.to_csv(join(preprocessed_path, 'total_day.csv'))
    df_total_month.to_csv(join(preprocessed_path, 'total_month.csv'))
    df_total_year.to_csv(join(preprocessed_path, 'total_year.csv'))

    utils.thin_line()
    print('Finished! Using time: {:.2f}s'.format( time.time() - time_))
    utils.thick_line()
