import numpy as np
import pandas as pd
import math
import yaml
import copy
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import argparse

class SimData:
    def __init__(self, args, config_path = 'config.yaml'):
        self.data = self.gen_data_from_config(config_path)
        self.temp_cols = ['Panel_Temp', 'OBC_Temp', 'Bat_Temp']
        self.current_cols = ['OBC_I1', 'OBC_I3', 'OBC_I8', 'Radio_I8']
        # generate sys failure, if there is a reset the indexing must be handled first then other faults can be added later
        sys_resets = self.gen_faults(sys_reset = True, forced = args.fault)
        if np.array(np.where(sys_resets!=None)).T.shape[0] > 0:
            self.add_faults(sys_resets, sys_reset = True)
        faults = self.gen_faults(forced = args.fault)
        if np.array(np.where(faults!=None)).T.shape[0] > 0:
            self.add_faults(faults)
        # self.data.iloc[:,:-1].to_csv('data/sim_data.csv', index=False)
        self.data.to_csv(self.outfile, index=False)
        print('Save successful')

    def sys_reset(self, ind, data, fault_array):
        for error_type in fault_array:
            if error_type == self.noclock_name:
                # clock continues but there is a break in data, resulting in a jump in the clock value
                duration = int(np.random.normal(float(self.faults['Reset']['noclock']['mean']), float(self.faults['Reset']['noclock']['std'])))
                if (ind in data.index.tolist()) and (duration>0) and (len(data.loc[ind+1:ind+duration]) > 0) and (max(data.index) > ind+duration):
                    new_ind = data.loc[ind+duration].name
                    data=data.drop(range(ind+1, ind+duration))
                    # data.loc[ind, "Fault"] = 1
                    # data = self.add_to_cell(data, ind, self.noclock_name, "Fault_Type")
                    # data = self.add_to_cell(data, ind, "clk", "Fault_Col")
                    data.loc[new_ind, "Fault"] = 1
                    data = self.add_to_cell(data, new_ind, self.noclock_name, "Fault_Type")
                    data = self.add_to_cell(data, new_ind, "clk", "Fault_Col")
                    tqdm.write(f'\nInd: {ind}, Dur: {duration}')
            elif error_type == self.clock_name:
                # clock resets to 0
                duration = int(np.random.normal(float(self.faults['Reset']['clock']['mean']), float(self.faults['Reset']['clock']['std'])))
                if (ind in data.index.tolist()) and (duration>0) and (len(data.iloc[ind+1:ind+duration]) > 0) and (max(data.index) > ind+duration):
                    new_ind = data.loc[ind+duration].name
                    data=data.drop(range(ind+1, ind+duration))
                    # data.loc[ind, "Fault"] = 1
                    # data = self.add_to_cell(data, ind, self.clock_name, "Fault_Type")
                    # data = self.add_to_cell(data, ind, "clk", "Fault_Col")
                    # new_ind = data.iloc[ind+1].name
                    data.loc[new_ind:, 'clk'] = np.array(range(0, len(data.loc[new_ind:])))
                    data.loc[new_ind, "Fault"] = 1
                    data = self.add_to_cell(data, new_ind, self.clock_name, "Fault_Type")
                    data = self.add_to_cell(data, new_ind, "clk", "Fault_Col")
                    tqdm.write(f'\nInd: {ind}, Dur: {duration}')
        return data

    def ion_fault(self, ind, data, fault_array):
        for error_type in fault_array:
            if error_type == self.random_name:
                if data.loc[ind, 'Fault'] == 1:
                    continue
                if len(self.faults['IRF']['random']['columns']) == 0:
                    col = np.random.choice(range(data.shape[1]-4))
                    data.iloc[ind, col] = np.random.uniform(self.faults['IRF']['random']['min'], self.faults['IRF']['random']['max'])
                    data = self.add_to_cell(data, ind, data.columns[col], "Fault_Col")
                else:
                    cols = self.faults['IRF']['random']['columns']
                    for col in cols:
                        data.loc[ind, col] = np.random.uniform(self.faults['IRF']['random']['min'], self.faults['IRF']['random']['max'])
                        data = self.add_to_cell(data, ind, col, "Fault_Col")
                data.loc[ind, "Fault"] = 1
                data = self.add_to_cell(data, ind, self.random_name, "Fault_Type")
            elif error_type == self.stale_name:
                duration = self.duration_selector(self.faults['IRF']['stale']['mean'], self.faults['IRF']['stale']['std'], ind, data)
                if duration == -1:
                    continue
                if len(self.faults['IRF']['stale']['columns']) == 0:
                    col = np.random.choice(range(data.shape[1]-4))
                    value = data.iloc[ind, col]
                    data.iloc[ind:ind+duration, col] = value
                    for j in range(ind,ind+duration):
                        data = self.add_to_cell(data, j, data.columns[col], "Fault_Col")
                        data = self.add_to_cell(data, j, self.stale_name, "Fault_Type")
                else:
                    cols = self.faults['IRF']['stale']['columns']
                    for col in cols:
                        value = data.loc[ind, col]
                        data.loc[ind:ind+duration, col] = value
                        for j in range(ind,ind+duration):
                            data = self.add_to_cell(data, j, col, "Fault_Col")
                            data = self.add_to_cell(data, j, self.stale_name, "Fault_Type")
                data.loc[ind:ind+duration-1, "Fault"] = 1
            elif error_type == self.zeros_name:
                duration = self.duration_selector(self.faults['IRF']['zeros']['mean'], self.faults['IRF']['zeros']['std'], ind, data)
                if duration == -1:
                    continue
                if len(self.faults['IRF']['zeros']['columns']) == 0:
                    col = np.random.choice(range(data.shape[1]-4))
                    data.iloc[ind:ind+duration, col] = 0
                    for j in range(ind,ind+duration):
                        data = self.add_to_cell(data, j, data.columns[col], "Fault_Col")
                        data = self.add_to_cell(data, j, self.zeros_name, "Fault_Type")
                else:
                    cols = self.faults['IRF']['zeros']['columns']
                    for col in cols:
                        data.loc[ind:ind+duration, col] = 0
                        for j in range(ind,ind+duration):
                            data = self.add_to_cell(data, j, data.columns[col], "Fault_Col")
                            data = self.add_to_cell(data, j, self.zeros_name, "Fault_Type")
                data.loc[ind:ind+duration-1, "Fault"] = 1
        return data

    def thermal(self, ind, data, fault_array):
        for error_type in fault_array:
            if error_type == self.hot_name:
                duration = self.duration_selector(self.faults['Thermal']['hot']['mean'], self.faults['Thermal']['hot']['std'], ind, data)
                if duration == -1:
                    continue
                if len(self.faults['Thermal']['hot']['columns']) == 0:
                    temp_col = np.random.choice(self.temp_cols)
                    data.loc[ind:ind+duration, temp_col] = self.faults['Thermal']['hot']['max']
                    for j in range(ind,ind+duration):
                        data = self.add_to_cell(data, j, temp_col, "Fault_Col")
                        data = self.add_to_cell(data, j, self.hot_name, "Fault_Type")
                else:
                    cols = self.faults['Thermal']['hot']['columns']
                    for temp_col in cols:
                        data.loc[ind:ind+duration, temp_col] = self.faults['Thermal']['hot']['max']
                        for j in range(ind,ind+duration):
                            data = self.add_to_cell(data, j, temp_col, "Fault_Col")
                            data = self.add_to_cell(data, j, self.hot_name, "Fault_Type")
                data.loc[ind:ind+duration-1, "Fault"] = 1
            elif error_type == self.failure_name:
                duration = self.duration_selector(self.faults['Thermal']['failure']['mean'], self.faults['Thermal']['failure']['std'], ind, data)
                if duration == -1:
                    continue
                stale = np.random.choice([True, False], p=[float(self.faults['Thermal']['failure']['stale_prob']), 1-float(self.faults['Thermal']['failure']['stale_prob'])])
                if stale:
                    if len(self.faults['Thermal']['failure']['columns']) == 0:
                        temp_col = data.columns.get_loc(np.random.choice(self.temp_cols))
                        value = data.iloc[ind, temp_col]
                        data.iloc[ind:ind+duration, temp_col] = value
                        for j in range(ind, ind+duration):
                            data = self.add_to_cell(data, j, data.columns[temp_col], "Fault_Col")
                            data = self.add_to_cell(data, j, self.failure_name, "Fault_Type")
                    else:
                        cols = self.faults['Thermal']['failure']['columns']
                        for col in cols:
                            value = data.loc[ind, col]
                            data.loc[ind:ind+duration, col] = value
                            for j in range(ind, ind+duration):
                                data = self.add_to_cell(data, j, col, "Fault_Col")
                                data = self.add_to_cell(data, j, self.failure_name, "Fault_Type")
                    data.loc[ind:ind+duration-1, "Fault"] = 1
                else:
                    if data.loc[ind, 'Fault'] == 1:
                        continue
                    if len(self.faults['Thermal']['failure']['columns']) == 0:
                        temp_col = np.random.choice(self.temp_cols)
                        val = np.random.choice([np.random.normal(self.faults['Thermal']['failure']['low'],
                                                                 self.faults['Thermal']['failure']['low_std']),
                                                np.random.normal(self.faults['Thermal']['failure']['high'],
                                                                 self.faults['Thermal']['failure']['high_std']),
                                                self.faults['Thermal']['failure']['min'],
                                                self.faults['Thermal']['failure']['max']])
                        if val < self.faults['Thermal']['failure']['min']:
                            val = self.faults['Thermal']['failure']['min']
                        elif val > self.faults['Thermal']['failure']['max']:
                            val = self.faults['Thermal']['failure']['max']
                        data.loc[ind, temp_col] = val
                        data = self.add_to_cell(data, ind, temp_col, "Fault_Col")
                        data = self.add_to_cell(data, ind, self.failure_name, "Fault_Type")
                    else:
                        cols = self.faults['Thermal']['failure']['columns']
                        for temp_col in cols:
                            val = np.random.choice([np.random.normal(self.faults['Thermal']['failure']['low'],
                                                                     self.faults['Thermal']['failure']['low_std']),
                                                    np.random.normal(self.faults['Thermal']['failure']['high'],
                                                                     self.faults['Thermal']['failure']['high_std']),
                                                    self.faults['Thermal']['failure']['min'],
                                                    self.faults['Thermal']['failure']['max']])
                            if val < self.faults['Thermal']['failure']['min']:
                                val = self.faults['Thermal']['failure']['min']
                            elif val > self.faults['Thermal']['failure']['max']:
                                val = self.faults['Thermal']['failure']['max']
                                data.loc[ind, temp_col] = val
                            data = self.add_to_cell(data, ind, temp_col, "Fault_Col")
                            data = self.add_to_cell(data, ind, self.failure_name, "Fault_Type")
                    data.loc[ind, "Fault"] = 1
            elif error_type == self.loss_name:
                #TODO: this can be done better with an HMM
                duration = self.duration_selector(self.faults['Thermal']['loss']['mean'], self.faults['Thermal']['loss']['std'], ind, data)
                if duration == -1:
                    continue
                # null temp
                temp_col = np.random.choice(self.temp_cols)
                data.loc[ind:ind+duration, temp_col] = -999
                # remove heater
                # find heater values around the ecl mean and given 2 std
                data_duration = data.loc[ind:ind+duration]
                data_duration = data_duration[data_duration['Heater_I'].between(self.heater['ecl']['mean'] - 2*self.heater['ecl']['std'], self.heater['ecl']['mean'] + 2*self.heater['ecl']['std'])]
                if len(data_duration) > 0:
                    duration_heater = max(data_duration.index.tolist()) - min(data_duration.index.tolist())
                    start_ind = min(data_duration.index.tolist())
                    data.loc[start_ind:start_ind+duration_heater, 'Heater_I'] = np.random.normal(self.heater['sun']['mean'], self.heater['sun']['std'], duration_heater+1)
                    # current down to 0 during eclipse
                    current_col = np.random.choice(self.current_cols)
                    data.loc[start_ind:start_ind+duration_heater, current_col] = 0
                    for j in range(start_ind,start_ind+duration_heater):
                        data = self.add_to_cell(data, j, current_col, "Fault_Col")
                        data = self.add_to_cell(data, j, "Heater_I", "Fault_Col")
                for j in range(ind,ind+duration):
                    data = self.add_to_cell(data, j, temp_col, "Fault_Col")
                    data = self.add_to_cell(data, j, self.loss_name, "Fault_Type")
                data.loc[ind:ind+duration-1, "Fault"] = 1
            elif error_type == self.component_name:
                duration = self.duration_selector(self.faults['Thermal']['component']['mean'], self.faults['Thermal']['component']['std'], ind, data)
                if duration == -1:
                    continue
                temp_col = data.columns.get_loc(np.random.choice(self.temp_cols))
                #take current value and decrease to low or increase to high over the duration
                value = data.iloc[ind, temp_col]
                direction = np.random.choice([1, -1])
                if direction == 1:
                    end_value = float(self.faults['Thermal']['component']['max'])
                else:
                    end_value = float(self.faults['Thermal']['component']['min'])
                data.iloc[ind:ind+duration, temp_col] = np.random.normal(np.linspace(value, end_value, duration), float(self.faults['Thermal']['component']['value_std']))
                for j in range(ind,ind+duration):
                    data = self.add_to_cell(data, j, data.columns[temp_col], "Fault_Col")
                    data = self.add_to_cell(data, j, self.component_name, "Fault_Type")
                data.loc[ind:ind+duration-1, "Fault"] = 1
        return data

    def overcurrent(self, ind, data, fault_array):
        for error_type in fault_array:
            if error_type == self.over_name:
                duration = self.duration_selector(self.faults['Current']['over']['mean'], self.faults['Current']['over']['std'], ind, data)
                if duration == -1:
                    continue
                if len(self.faults['Current']['over']['columns']) == 0:
                    current_col = np.random.choice(self.current_cols)
                    data.loc[ind:ind+duration, current_col] = self.faults['Current']['over']['max']
                    for j in range(ind,ind+duration):
                        data = self.add_to_cell(data, j, current_col, "Fault_Col")
                        data = self.add_to_cell(data, j, self.over_name, "Fault_Type")
                else:
                    cols = self.faults['Current']['over']['columns']
                    for current_col in cols:
                        data.loc[ind:ind+duration, current_col] = self.faults['Current']['over']['max']
                        for j in range(ind,ind+duration):
                            data = self.add_to_cell(data, j, current_col, "Fault_Col")
                            data = self.add_to_cell(data, j, self.over_name, "Fault_Type")
                data.loc[ind:ind+duration-1, "Fault"] = 1
            elif error_type == self.under_name:
                duration = self.duration_selector(self.faults['Current']['under']['mean'], self.faults['Current']['under']['std'], ind, data)
                if duration == -1:
                    continue
                if len(self.faults['Current']['under']['columns']) == 0:
                    current_col = np.random.choice(self.current_cols)
                    data.loc[ind:ind+duration, current_col] = self.faults['Current']['under']['min']
                    for j in range(ind,ind+duration):
                        data = self.add_to_cell(data, j, current_col, "Fault_Col")
                        data = self.add_to_cell(data, j, self.under_name, "Fault_Type")
                else:
                    cols = self.faults['Current']['under']['columns']
                    for current_col in cols:
                        data.loc[ind:ind+duration, current_col] = self.faults['Current']['under']['min']
                        for j in range(ind,ind+duration):
                            data = self.add_to_cell(data, j, current_col, "Fault_Col")
                            data = self.add_to_cell(data, j, self.under_name, "Fault_Type")
                data.loc[ind:ind+duration-1, "Fault"] = 1
        return data

    def duration_selector(self, mean, std, ind, data):
        # select duration of error
        duration = int(np.random.normal(float(mean), float(std)))
        # ensure two errors never overlap
        faults_over_duration = data.loc[ind:ind+duration, 'Fault'].tolist()
        if data.loc[ind, 'Fault'] == 1:
            return -1
        while (duration < 2) or (1 in faults_over_duration) or (ind+duration > len(data)):
            duration = int(np.random.normal(float(mean), float(std)))
            faults_over_duration = data.loc[ind:ind+duration, 'Fault'].tolist()
        return duration

    def add_faults(self, fault_array, sys_reset = False):
        if sys_reset:
            for ind, row in tqdm(self.data.iterrows(), total=self.data.shape[0], ncols=100):
                if len(np.where(fault_array[ind, :]!=None)[0]) != 0:
                    self.data = self.sys_reset(ind, self.data, fault_array[ind, :])
            self.data = self.data.reset_index(drop=True)
        else:
            # iterrate through each row and alter the data accordingly
            for ind, row in tqdm(self.data.iterrows(), total=self.data.shape[0], ncols=100):
                if len(np.where(fault_array[ind, :]!=None)[0]) != 0:
                    self.data = self.ion_fault(ind, self.data, fault_array[ind, :])
                    self.data = self.thermal(ind, self.data, fault_array[ind, :])
                    self.data = self.overcurrent(ind, self.data, fault_array[ind, :])

    def gen_faults(self, sys_reset = False, forced = None):
        if sys_reset:
            self.clock_name = self.faults['Reset']['clock']['name']
            clock_prob = float(self.faults['Reset']['clock']['prob'])
            self.noclock_name = self.faults['Reset']['noclock']['name']
            noclock_prob = float(self.faults['Reset']['noclock']['prob'])
            clock = np.atleast_2d(np.random.choice([None, self.clock_name], size = len(self.data), p=[1-clock_prob, clock_prob])).T
            no_clock = np.atleast_2d(np.random.choice([None, self.noclock_name], size = len(self.data), p=[1-noclock_prob, noclock_prob])).T
            if forced is not None:
                if forced == self.clock_name:
                    clock[int(np.random.uniform(0, len(self.data)))] = self.clock_name
                elif forced == self.noclock_name:
                    no_clock[int(np.random.uniform(0, len(self.data)))] = self.noclock_name
            sys_resets = np.hstack((clock, no_clock))
            print("Number of System Resets:", np.array(np.where(sys_resets!=None)).T.shape[0])
            return sys_resets
        else:
            self.random_name = self.faults['IRF']['random']['name']
            random_prob = float(self.faults['IRF']['random']['prob'])
            self.stale_name = self.faults['IRF']['stale']['name']
            stale_prob = float(self.faults['IRF']['stale']['prob'])
            self.zeros_name = self.faults['IRF']['zeros']['name']
            zeros_prob = float(self.faults['IRF']['zeros']['prob'])
            self.hot_name = self.faults['Thermal']['hot']['name']
            hot_prob = float(self.faults['Thermal']['hot']['prob'])
            self.failure_name = self.faults['Thermal']['failure']['name']
            failure_prob = float(self.faults['Thermal']['failure']['prob'])
            self.component_name = self.faults['Thermal']['component']['name']
            component_prob = float(self.faults['Thermal']['component']['prob'])
            self.loss_name = self.faults['Thermal']['loss']['name']
            loss_prob = float(self.faults['Thermal']['loss']['prob'])
            self.over_name = self.faults['Current']['over']['name']
            over_prob = float(self.faults['Current']['over']['prob'])
            self.under_name = self.faults['Current']['under']['name']
            under_prob = float(self.faults['Current']['under']['prob'])
            random = np.atleast_2d(np.random.choice([None, self.random_name], size = len(self.data), p=[1-random_prob, random_prob])).T
            stale = np.atleast_2d(np.random.choice([None, self.stale_name], size = len(self.data), p=[1-stale_prob, stale_prob])).T
            zeros = np.atleast_2d(np.random.choice([None, self.zeros_name], size = len(self.data), p=[1-zeros_prob, zeros_prob])).T
            hot = np.atleast_2d(np.random.choice([None, self.hot_name], size = len(self.data), p=[1-hot_prob, hot_prob])).T
            therm_sens_f = np.atleast_2d(np.random.choice([None, self.failure_name], size = len(self.data), p=[1-failure_prob, failure_prob])).T
            therm_comp = np.atleast_2d(np.random.choice([None, self.component_name], size = len(self.data), p=[1-component_prob, component_prob])).T
            therm_sens_l = np.atleast_2d(np.random.choice([None, self.loss_name], size = len(self.data), p=[1-loss_prob, loss_prob])).T
            over = np.atleast_2d(np.random.choice([None, self.over_name], size = len(self.data), p=[1-over_prob, over_prob])).T
            under = np.atleast_2d(np.random.choice([None, self.under_name], size = len(self.data), p=[1-under_prob, under_prob])).T
            if forced is not None:
                if forced == self.random_name:
                    random[int(np.random.uniform(0, len(self.data)))] = self.random_name
                elif forced == self.stale_name:
                    stale[int(np.random.uniform(0, len(self.data)))] = self.stale_name
                elif forced == self.zeros_name:
                    zeros[int(np.random.uniform(0, len(self.data)))] = self.zeros_name
                elif forced == self.hot_name:
                    hot[int(np.random.uniform(0, len(self.data)))] = self.hot_name
                elif forced == self.failure_name:
                    therm_sens_f[int(np.random.uniform(0, len(self.data)))] = self.failure_name
                elif forced == self.component_name:
                    therm_comp[int(np.random.uniform(0, len(self.data)))] = self.component_name
                elif forced == self.loss_name:
                    therm_sens_l[int(np.random.uniform(0, len(self.data)))] = self.loss_name
                elif forced == self.over_name:
                    over[int(np.random.uniform(0, len(self.data)))] = self.over_name
                elif forced == self.under_name:
                    under[int(np.random.uniform(0, len(self.data)))] = self.under_name
            all_faults = np.hstack((random, stale, zeros, hot, therm_sens_f, therm_comp, therm_sens_l, over, under))
            print("Number of Faults:", np.array(np.where(all_faults!=None)).T.shape[0])
            return all_faults

    def random_walk(self, mean, std, size):
        variable = [np.random.normal(mean, std)]*size
        walk = np.random.normal(0, std, size-1)
        for i in range(len(walk)):
            # keep the value between 3 std
            if abs(variable[i] + walk[i] - mean) > (3*std):
                if abs(variable[i] - walk[i] - mean) > (3*std):
                    #special case
                    deviation = True
                    while deviation:
                        candidate = variable[i] + np.random.normal(0, std)
                        if (candidate - mean) > (3*std):
                            continue
                        else:
                            variable[i+1] = candidate
                            deviation = False
                else:
                    variable[i+1] = variable[i] - walk[i]
            else:
                variable[i+1] = variable[i] + walk[i]
        return variable

    def gen_data_from_config(self, path_to_yaml):
        
        with open(path_to_yaml, 'r') as yamlfile:
            cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
        
        self.outfile = cfg['out_file_name']
        self.sample_rate = cfg['sample_rate']
        self.orbit_period = cfg['orbit_period']
        self.num_orbits = cfg['num_orbits']
        tel = cfg['telemetry']
        # used for sesnor loss case
        self.heater = cfg['telemetry']['Heater_I']
        self.faults = cfg['faults']
        self.reset_names = [cfg['faults']['Reset']['clock']['name'], cfg['faults']['Reset']['noclock']['name']]
        
        data = pd.DataFrame()
        
        # Number of samples in sunlight (55 min) and eclipse (35 min)
        total_per_orbit = self.orbit_period*60*self.sample_rate
        self.total_samples = total_per_orbit*self.num_orbits

        for name in tel:
            # Same for entire orbit
            if name in ['Heater_V', 'OBC_V1', 'OBC_V3', 'OBC_V8', 'OBC_I1', 'OBC_I3']:
                mean = tel[name]['mean']
                std = tel[name]['std']
                data[name] = self.random_walk(mean, std, size=self.total_samples)
            
            # Different vals for sun vs eclipse
            elif name in ['V_bat', 'I_bus', 'Heater_I', 'OBC_Temp', 'Bat_Temp']:
                sun_mean = tel[name]['sun']['mean']
                sun_std = tel[name]['sun']['std']
                sun_samples = round(total_per_orbit*tel[name]['sun']['dur'])
                ecl_mean = tel[name]['ecl']['mean']
                ecl_std = tel[name]['ecl']['std']
                ecl_samples = round(total_per_orbit*tel[name]['ecl']['dur'])
                data[name] = np.array([np.concatenate((self.random_walk(sun_mean, sun_std, sun_samples),
                                                    self.random_walk(ecl_mean, ecl_std, ecl_samples)))
                                                    for i in range(self.num_orbits)]).flatten()
                
            # Increase during sun, decrease during eclipse
            elif name in ['Panel_Temp']:
                sun_min = tel[name]['sun']['min']
                sun_max = tel[name]['sun']['max']
                sun_std = tel[name]['sun']['std']
                sun_samples = round(total_per_orbit*tel[name]['sun']['dur'])
                ecl_min = tel[name]['ecl']['min']
                ecl_max = tel[name]['ecl']['max']
                ecl_std = tel[name]['ecl']['std']
                ecl_samples = round(total_per_orbit*tel[name]['ecl']['dur'])
                data[name] = np.array([np.concatenate((np.random.normal(np.linspace(sun_min, sun_max, sun_samples), sun_std), 
                                np.random.normal(np.linspace(ecl_max, ecl_min, ecl_samples), ecl_std)))
                                for i in range(self.num_orbits)]).flatten()
            
            # OBC_I8: special case
            elif name in ['OBC_I8']:
                bl_mean = tel[name]['baseline']['mean']
                bl_std = tel[name]['baseline']['std']
                on_mean = tel[name]['active']['mean']
                on_std = tel[name]['active']['std']
                data[name] = np.array([np.concatenate((self.random_walk(bl_mean, bl_std, 5*60*self.sample_rate),
                                self.random_walk(on_mean, on_std, 10*60*self.sample_rate),
                                self.random_walk(bl_mean, bl_std, 5*60*self.sample_rate),
                                self.random_walk(on_mean, on_std, 10*60*self.sample_rate),
                                self.random_walk(bl_mean, bl_std, 60*60*self.sample_rate)))
                                for i in range(self.num_orbits)]).flatten()
                
            # Radio_I8: special case
            elif name in ['Radio_I8']:
                num_on = tel[name]['active']['dur']*60*self.sample_rate
                samples_per_dl = total_per_orbit*16/tel[name]['dl_per_day'] #16 = orbits/day
                idx_on = (np.array(range(math.floor(self.total_samples/samples_per_dl)))+1)*samples_per_dl
                idx_on = np.array([idx for idx in idx_on if idx+num_on<self.total_samples]).astype(int)
                idx_use = [0] + [idx_on[int(i/2)] if i%2==0
                                else idx_on[int(i/2)]+num_on
                                for i in range(len(idx_on)*2)] + [self.total_samples]
                bl_mean = tel[name]['baseline']['mean']
                bl_std = tel[name]['baseline']['std']
                on_mean = tel[name]['active']['mean']
                on_std = tel[name]['active']['std']
                data[name] = np.concatenate([self.random_walk(bl_mean, bl_std, idx_use[i+1]-idx_use[i]) if i%2==0
                                        else self.random_walk(on_mean, on_std, num_on)
                                        for i in range(len(idx_use)-1)])
                    
        data['clk'] = np.array(range(self.total_samples))
        data['Fault'] = [0]*len(data)
        data['Fault_Type'] = [None]*len(data)
        data['Fault_Col'] = [None]*len(data)
        
        
        return data

    def add_to_cell(self, data, ind, name, col):
        # if ind > (len(data) - 1):
        #     return data
        # check if empty, if no, add it
        if data.loc[ind, col] is None:
            data.loc[ind, col] = name
        # if it has a string, turn into list and append
        elif isinstance(data.loc[ind, col], str):
            data.loc[ind, col] = [data.loc[ind, col]], [name]
        # if list, append
        elif isinstance(data.loc[ind, col], list):
            data.loc[ind, col].append(name)
        return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', default = False, action = "store_true", help='graph the results', required=False)
    parser.add_argument('-f', '--fault', default = None,  help='force a particular fault (IRF_R, IRF_S, IRF_Z, K_H, K_F, K_L, I_O, I_U, SR_C, SR_N). Fault names can also be found in the config', required=False)
    args = parser.parse_args()

    sim = SimData(args, 'config_fault_irfr.yaml')

    if args.graph:
        sim.data.reset_index(drop = True)
        fig, ax = plt.subplots(4, 4, figsize=(10, 7.5))
        ax_count = 0
        fault_data = sim.data[sim.data['Fault'] == 1]
        if len(fault_data[fault_data['Fault_Type'].isin(sim.reset_names)]) > 0:
            reset = True
        else:
            reset = False
        inds = fault_data.index.tolist()
        new_inds = np.random.choice(inds, 10)
        no_fault_data = sim.data[sim.data['Fault'] != 1]
        for i in range(15):
            name = sim.data.columns[i]
            ax[int(i/4), i%4].plot(no_fault_data.index.tolist(), no_fault_data[name])
            fault_inds = []
            for ind, row in fault_data.iterrows():
                if name in row['Fault_Col']:
                    fault_inds.append(ind)
            ax[int(i/4), i%4].scatter(fault_data.loc[fault_inds].index.tolist(), fault_data.loc[fault_inds, name], c='red')
            if reset:
                resets = fault_data[fault_data['Fault_Type'].isin(sim.reset_names)]
                for ind_j, row_j in resets.iterrows():
                    ax[int(i/4), i%4].vlines(ind_j, min(sim.data[name]), max(sim.data[name]), color='red')
            ax[int(i/4), i%4].set_title(name)
            ax[int(i/4), i%4].set_xlabel('Clock')
            ax[int(i/4), i%4].set_ylabel(name)
        plt.show()