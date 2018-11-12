import time
import pandas as pd
from models import utils
from os.path import join
from config import cfg

source_path = '../data/source_data'
preprocessed_path = '../data/preprocessed_data'


# Load CSV Files Using Pandas
def load_csv(file_path):

    f = pd.read_csv(file_path, index_col='FDATE', parse_dates=['FDATE'],
                    usecols=['FDATE', 'CONTNUM', 'CONTPRICE'])
    f['VOLUME'] = f['CONTNUM'] * f['CONTPRICE']

    return f


def preprocess(df, sample_mode):

    if sample_mode == 'day':
        df_grouped = df.groupby(
            [lambda x: x.year, lambda x: x.month, lambda x: x.day])
    elif sample_mode == 'month':
        df_grouped = df.groupby([lambda x: x.year, lambda x: x.month])
    elif sample_mode == 'year':
        df_grouped = df.groupby(lambda x: x.year)
    else:
        raise ValueError('Wrong Sample Mode Name!')

    suffix = utils.get_suffix(sample_mode)
    df_preprocessed = pd.DataFrame()
    df_preprocessed['CONTNUM_' + suffix] = df_grouped['CONTNUM'].sum()
    df_preprocessed['VOLUME_' + suffix] = df_grouped['VOLUME'].sum()
    df_preprocessed['CONTPRICE_' + suffix] = \
        df_preprocessed['VOLUME_' + suffix] / \
        df_preprocessed['CONTNUM_' + suffix]
    # df_preprocessed['CONTPRICE_' + suffix + '_AVG'] = \
    #     df_grouped['CONTPRICE'].mean()

    if sample_mode == 'day':
        df_preprocessed.index = df_preprocessed.index.rename(
            ['YEAR', 'MONTH', 'DAY'])
    if sample_mode == 'month':
        df_preprocessed.index = df_preprocessed.index.rename(['YEAR', 'MONTH'])
    if sample_mode == 'year':
        df_preprocessed.index = df_preprocessed.index.rename('YEAR')

    return df_preprocessed


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
        df_day.append(preprocess(df_loaded, 'day'))
        df_month.append(preprocess(df_loaded, 'month'))
        df_year.append(preprocess(df_loaded, 'year'))

    print('Concatenating data...')
    df_total_day: pd.DataFrame = pd.concat(df_day)
    df_total_month: pd.DataFrame = pd.concat(df_month)
    df_total_year: pd.DataFrame = pd.concat(df_year)

    print('Writing data to CSV...')
    df_total_day.to_csv(join(cfg.preprocessed_path, 'total_day.csv'))
    df_total_month.to_csv(join(cfg.preprocessed_path, 'total_month.csv'))
    df_total_year.to_csv(join(cfg.preprocessed_path, 'total_year.csv'))

    utils.thin_line()
    print('Finished! Using time: {:.2f}s'.format(time.time() - time_))
    utils.thick_line()
