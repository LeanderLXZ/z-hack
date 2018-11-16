import time
import pandas as pd
from pandas import datetime as dt
from models import utils
from os.path import join
from config import cfg


class PreProcess(object):

    def __init__(self):
        pass

    @staticmethod
    def load_csv(file_path):
        df = pd.read_csv(file_path,
                              index_col='FDATE',
                              parse_dates=['FDATE'],
                              usecols=['FDATE', 'CONTNUM', 'CONTPRICE'])
        df['VOLUME'] = df['CONTNUM'] * df['CONTPRICE']
        return df

    @staticmethod
    def save_to_csv(df, sample_mode):
        utils.check_dir([cfg.preprocessed_path])
        df.to_csv(
            join(cfg.preprocessed_path, 'total_{}.csv'.format(sample_mode)))

    @staticmethod
    def preprocess(df, sample_mode):

        if sample_mode == 'day':
            df_grouped = df.groupby(df.index)
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
            df_preprocessed.index = df_preprocessed.index.rename('DATE')
        if sample_mode == 'month':
            df_preprocessed.index = df_preprocessed.index.rename(['YEAR', 'MONTH'])
        if sample_mode == 'year':
            df_preprocessed.index = df_preprocessed.index.rename('YEAR')

        return df_preprocessed

    def preprocess_years(self):

        df_day = []
        df_month = []
        df_year = []
        for year in ['2009', '2010', '2011', '2012', '2013']:
            print('Preprocessing data in {}...'.format(year))
            df_loaded = self.load_csv(
                join(cfg.source_path,
                     'z_hack_transaction_{}_new.csv'.format(year)))
            df_day.append(self.preprocess(df_loaded, 'day'))
            df_month.append(self.preprocess(df_loaded, 'month'))
            df_year.append(self.preprocess(df_loaded, 'year'))

        print('Concatenating data...')
        df_total_day: pd.DataFrame = pd.concat(df_day)
        df_total_month: pd.DataFrame = pd.concat(df_month)
        df_total_year: pd.DataFrame = pd.concat(df_year)

        print('Filling missing data...')
        start = dt(2009, 1, 4)
        end = dt(2013, 12, 31)
        df_day_miss = df_total_day.reindex(
            pd.bdate_range(start, end).rename('DATE'))
        df_day_mff = df_day_miss.fillna(method='ffill')
        df_day_mbf = df_day_miss.fillna(method='bfill')
        df_day_all = df_total_day.reindex(
            pd.date_range(start, end).rename('DATE'))
        df_day_aff = df_day_all.fillna(method='ffill')
        df_day_abf = df_day_all.fillna(method='bfill')

        print('Writing data to CSV...')
        self.save_to_csv(df_total_day, 'day')
        self.save_to_csv(df_total_month, 'month')
        self.save_to_csv(df_total_year, 'year')
        self.save_to_csv(df_day_miss, 'work_day_miss')
        self.save_to_csv(df_day_mff, 'work_day_mff')
        self.save_to_csv(df_day_mbf, 'work_day_mbf')
        self.save_to_csv(df_day_all, 'work_day_all')
        self.save_to_csv(df_day_aff, 'work_day_aff')
        self.save_to_csv(df_day_abf, 'work_day_abf')


if __name__ == '__main__':

    time_ = time.time()
    utils.thick_line()
    print('Start preprocessing...')

    PreProcess().preprocess_years()

    utils.thin_line()
    print('Finished! Using time: {:.2f}s'.format(time.time() - time_))
    utils.thick_line()
