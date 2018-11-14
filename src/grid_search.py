import time
import numpy as np
import pandas as pd
from os.path import join
from models import utils
from main_ts import Training
from config import cfg


class GridSearch(object):

    def __init__(self, sample_mode, select_col):
        self.T = Training(sample_mode, select_col)
        self.sample_mode = sample_mode

    @staticmethod
    def _generate_grid_pairs(param_grid):

        n_param = len(param_grid)
        n_value = 1
        param_name = []
        for i in range(n_param):
            param_name.append(param_grid[i][0])
            n_value *= len(param_grid[i][1])

        param_value = np.zeros((n_param, n_value)).tolist()
        global value_list
        global value_col
        value_list = []
        value_col = 0

        def generate_value_matrix_(idx_param):
            idx_param_next = idx_param + 1
            for value in param_grid[idx_param][1]:
                global value_list
                value_list.append(value)
                if idx_param_next < n_param:
                    generate_value_matrix_(idx_param_next)
                else:
                    global value_col
                    for i_row, row in enumerate(param_value):
                        row[value_col] = value_list[i_row]
                    value_col += 1
                value_list.pop()
        generate_value_matrix_(0)

        grid_pairs = []
        for i_param_value in range(n_value):
            grid_search_tuple_dict = {}
            param_info = ''
            for i_param in range(n_param):
                param_name_i = param_name[i_param]
                param_value_i = param_value[i_param][i_param_value]
                param_info += '_' + utils.get_simple_param_name(
                    param_name_i) + '-' + str(param_value_i)
                grid_search_tuple_dict[param_name_i] = param_value_i
            grid_pairs.append((grid_search_tuple_dict, param_info))

        return grid_pairs

    def grid_search(self, param_grid,
                    save_every_result=False, append_info=None):

        start_time = time.time()

        df_total = pd.read_csv(
            join(cfg.source_path, 'z_hack_submit_new_with_cost.csv'),
            index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
        forest_num = len(df_total) - 1

        df_valid = pd.DataFrame(index=range(30))

        idx = 0

        for i, grid_i in enumerate(param_grid):

            task_time = time.time()

            utils.thick_line()
            print('Grid Searching Task {}...'.format(i))


            grid_pairs = self._generate_grid_pairs(grid_i)
            print('Pairs in the Task: {}'.format(len(grid_pairs)))
            utils.thin_line()

            for grid_search_tuple_dict, param_info in grid_pairs:

                grid_time = time.time()

                print('Grid Searching idx_{}: param_{}...'.format(
                    idx, param_info))

                model_name = grid_search_tuple_dict['model_name']
                start_year = grid_search_tuple_dict['start_year']
                valid_range = grid_search_tuple_dict['valid_range']
                frequency = grid_search_tuple_dict['frequency']
                hw_seasonal = grid_search_tuple_dict['hw_seasonal']

                data_range = {'train_start': '{}-1-4'.format(start_year),
                              'valid_start': valid_range[0],
                              'valid_end': valid_range[1]}

                if append_info is None:
                    append_info = ''

                pred_final, cost, pred_valid = self.T.train(
                    model_name=model_name,
                    freq=frequency,
                    forest_num=forest_num,
                    seasonal=hw_seasonal,
                    data_range=data_range,
                    save_result=save_every_result,
                    append_info='_' + str(idx) + append_info)

                utils.save_ts_log_to_csv(
                    log_path=cfg.log_path,
                    grid_search_tuple_dict=grid_search_tuple_dict,
                    cost=cost,
                    idx=idx,
                    append_info='_' + self.sample_mode + append_info)

                pred_final = np.append(pred_final, cost)
                df_total[str(idx)] = pred_final

                if len(pred_valid) < 29:
                    pred_valid = \
                        np.append(pred_valid, np.zeros(29 - len(pred_valid)))
                pred_cost_valid = np.append(pred_valid, cost)
                df_valid[str(idx)] = pred_cost_valid

                idx += 1

                print('Done! Using {:.2f}s...'.format(time.time() - grid_time))

            utils.thin_line()
            print('Task {} Done! Using {:.2f}s...'.format(
                i, time.time() - task_time))
            utils.thick_line()

        utils.check_dir([cfg.result_path])
        df_total = df_total.stack().unstack(0)
        df_total.to_csv(join(
            cfg.log_path,
            'all_results_{}_{}.csv'.format(self.sample_mode, append_info)))
        df_valid.to_csv(join(
            cfg.log_path,
            'all_valid_{}_{}.csv'.format(self.sample_mode, append_info)))

        utils.thick_line()
        print('All Task Done! Using {:.2f}s...'.format(
            time.time() - start_time))
        utils.thick_line()


if __name__ == '__main__':

    parameter_grid = [[['model_name', ('arima', 'stl', 'ets', 'hw')],
                       ['start_year', (2009, 2010, 2011, 2012)],
                       ['valid_range', [('2013-12-2', '2013-12-31'), ('2013-1-4', '2013-1-31')]],
                       ['frequency', (5, 10, 15, 20, 25, 30)],
                       ['hw_seasonal', ['multiplicative']]],
                      [['model_name', ('arima', 'stl', 'ets', 'hw')],
                       ['start_year', [2013]],
                       ['valid_range', [('2013-12-2', '2013-12-31')]],
                       ['frequency', (5, 10, 15, 20, 25, 30)],
                       ['hw_seasonal', ['multiplicative']]]]

    GS = GridSearch(sample_mode='day', select_col='CONTPRICE')
    GS.grid_search(parameter_grid, save_every_result=True, append_info='')
