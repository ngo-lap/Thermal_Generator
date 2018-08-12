from optlang import Objective, Variable, Constraint, Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
import timeit
from bokeh.plotting import figure, output_file, ColumnDataSource, output_notebook
from bokeh.io import show
from bokeh.models import HoverTool
from bokeh.layouts import column

sns.set()
plt.rc('font', size=15)

# Generator Parameters


class Thermal_Generator(object):

    def __init__(self, Name: str, price: pd.DataFrame, Efficiency=[0.58, 0.47], Power=[100, 40], Emission_Factor=0.25, CF=0.75, Startup_cold_time=2, Startup_dep_cost=60,
                 Startup_fuel=2.8, Minimum_downtime=None, Cost_fixed_OM=28100, Cost_var_OM=2.3):

        self.Name = Name
        self.Efficiency = Efficiency   # Efficiencies of 2 levels of operations, [0.58, 0.47]
        self.Power = Power             # Power levels of the generators [MW], [100, 40]
        self.Horizon = len(price)      # Numbers of days
        assert (len(self.Efficiency) == len(self.Power)), 'Efficiency and Power lists must have the same length'

        self.Energy = self.Power*1            # Energy generated in 1 hor
        self.Emission_Factor = Emission_Factor      # Emission Factor of Coal, [tCO_2/MWh_thermal], 0.25
        self.Emission_Intensity = [self.Emission_Factor/effi for effi in self.Efficiency]      # Emission Intensity of 2 levels of operations  [tCO_2/MWhe]

        self.CF = CF                                 # Capacity Factor, 0.75
        self.Startup_cold_time = Startup_cold_time   # Time for cold startup coal (3), and 2 for CCGT [hour], 2
        self.Startup_dep_cost = Startup_dep_cost        # Startup depriciation costs (60 for CCGT, 105 for coal) [€], 60
        self.Startup_fuel = Startup_fuel          # Startup fuel requirement (2.8 for CCGT,2.7 for coal) [MW_thermal / MW_e], 2.8
        self.Minimum_downtime = Minimum_downtime        # Minimum downtime
        self.Cost_fixed_OM = Cost_fixed_OM      # [€/MW/year] [IEA2015, 0.81 is the 2018 exchange rate in 2018, 0.9189 is in 2015] 30568 for ccgt , 41303 for coal-fired, 30568*0.9189
        self.Cost_var_OM = Cost_var_OM         # [€/MWh_e] [IEA 2015, the 0.81 is the exchange rate in 2018] 2.5 for ccgt, 4.4 for coal.

        self.N_mode = len(self.Efficiency)

        class empty_class:
            pass

        self.input_price = price
        self.Commodity_Price = empty_class
        self.Commodity_Price.electricity_price = price['electricity'].values
        self.Commodity_Price.fossil_price = price['fossil'].values
        self.Commodity_Price.carbon_price = price['carbon'].values
        self.Commodity_Price.fuel_price = {}

        for n in range(self.N_mode):
            self.Commodity_Price.fuel_price[n] = self.Commodity_Price.fossil_price/self.Efficiency[n]

        self.Commodity_Price.fuel_price = pd.DataFrame(index=price.index, data=self.Commodity_Price.fuel_price)
        self.Operation_Profile = {'STMC': None, 'Energy_Profile': None, 'Capacity Factor': 'Only exists for yearly horizon'
                                                , 'Start-up Numbers': None}
        self.Finance_Metrics = {'Revenue': 0, 'OPEX': 0, 'Gross Profit': 0, 'Average STMC': 0}
        self.optim_model = None
        self.solutions = None
        self._var = {}      # list of lists of Variable
        self._cons = {}     # list of lists of Constraint

    def optimization_problem(self):

        """This method is to build the mathematical optimization problem for the generator"""

        print('Building the problem - Please wait')
        print('Variables')
        # Parameters abbreviation
        N = self.Horizon
        efficiency = self.Efficiency
        power = self.Power
        energy = self.Energy
        mode = self.N_mode
        alpha = self.Startup_cold_time
        beta = self.Minimum_downtime
        price_elec = self.Commodity_Price.electricity_price
        price_carbon = self.Commodity_Price.carbon_price
        price_fuel = self.Commodity_Price.fuel_price
        price_fossil = self.Commodity_Price.fossil_price

        # Variable Registration
        # -----------------------------------------------------------------------------------
        model = Model(name=self.Name)
        X = [[]]*mode   # list of lists, representing state variables for each mode of operation
        S = [Variable(name='start_' + str(t), type='binary') for t in range(N)]
        F = [Variable(name='shutdown_' + str(t), type='binary') for t in range(N)]

        for m in range(mode):

            X[m] = [Variable(name='state_mode_' + str(m) + '_' + str(t), type='binary') for t in range(N)]

            for t in range(alpha):
                X[m][t].set_bounds(0, 0)

        print('Constraints')
        # Constraints Registration
        # -----------------------------------------------------------------------------------
        # ctr_initial_states = [[]]*mode
        ctr_unique_mode = [[]]*N
        ctr_start_shut = [[]]*N
        ctr_init_state = [[]]*mode
        ctr_start_01 = [[]]*(N-alpha-1)
        ctr_start_02 = [[]]*(N-alpha)

        # Initial States Constraints
        for m in range(mode):

            ctr_init_state[m] = [[]]*(alpha+1)

            for t in range(alpha+1):
                ctr_init_state[m][t] = Constraint(X[m][t], lb=0, ub=0, name='ctr_initial_states_m_' + str(m) + str(t))

        # Listed constraints
        for t in range(N):

            # 1.1 Unique mode constraint:
            ctr_unique_mode[t] = Constraint(sum(X[m][t] for m in range(mode)), ub=1, name='ctr_unique_mode_' + str(t))

            # 1.2 Startup - shutdown constraint:
            ctr_start_shut[t] = Constraint(S[t] + F[t], ub=1, name='ctr_start_shut_' + str(t))

        for i, t in enumerate(range(alpha, N-1)):     # 2, 21

            # 1.3 Startup - shutdown constraint 2 :
            ctr_start_01[i] = Constraint(S[t - alpha] - F[t] - sum(X[m][t+1] - X[m][t] for m in range(mode)), lb=0, ub=0, name='ctr_start_01_' + str(t))

            # 1.4 Minimum startup time :
            ctr_start_02[i] = Constraint(sum(sum(X[m][t - k] for k in range(1, alpha)) for m in range(mode)) + alpha*S[t-alpha],
                                         ub=alpha, name='ctr_start_02_' + str(t))

        ctr_start_02[N-alpha-1] = Constraint(sum(sum(X[m][N - 1 - k] for k in range(1, alpha)) for m in range(mode)) + alpha*S[N - 1 - alpha],
                                             ub=alpha, name='ctr_start_02_' + str(N - 1))

        # 1.5 Capacity Factor constraint : will be done below

        # 1.6 Minimum shutdown time : will be done below

        # Objective function :
        # -----------------------------------------------------------------------------------

        print('Objective')
        obj_list = [[]]*(mode+1)
        obj_func = Objective(0, direction='max')
        obj_coeff_dict = {}

        obj_coeff_start = dict(zip(S, [-self.Power[0]*(self.Startup_dep_cost + self.Startup_fuel*price_fossil[t]) for t in range(N)]))
        obj_coeff_dict.update(obj_coeff_start)

        for m in range(mode):

            # obj_list[m] = Objective(energy[m]*sum(price_elec[t]*X[m][t] - price_fuel[m].values[t]*X[m][t] - price_carbon[t]*self.Emission_Intensity[m]*X[m][t]
            #                                        - X[m][t]*self.Cost_var_OM for t in range(N)), direction='max')

            obj_coeff_rev = dict(zip(X[m], [energy[m]*(price_elec[t] - price_fuel[m].values[t] - price_carbon[t]*self.Emission_Intensity[m]
                                                       - self.Cost_var_OM) for t in range(N)]))
            obj_coeff_dict.update(obj_coeff_rev)

        # for elem in obj_list:
        #
        #     obj_func += elem.expression

        # Add variables and constraints to the model :

        var_list = []
        cons_list = []

        for m in range(mode):
            var_list.extend(X[m])
            cons_list.extend(ctr_init_state[m])

        var_list.extend(S)
        var_list.extend(F)

        cons_list.extend(ctr_unique_mode)
        cons_list.extend(ctr_start_shut)
        cons_list.extend(ctr_start_01)
        cons_list.extend(ctr_start_02)

        # 1.6 Minimum shutdown time :
        if self.Minimum_downtime is not None:

            ctr_min_time = [[]] * (N - beta)

            for i, t in enumerate(range(N - beta)):
                ctr_min_time[i] = Constraint(
                    sum(sum(X[m][t + k] for k in range(1, beta + 1)) for m in range(mode)) + beta * F[t],
                    lb=0, ub=beta, name='ctr_min_time_' + str(t))

            cons_list.extend(ctr_min_time)
            self._cons['ctr_min_time'] = ctr_min_time

        model.add(var_list)

        # 1.5 Capacity Factor Constraint :
        # ------------------------------------------------------------------------------------------------

        index = self.input_price.index
        time_interval = (index[-1] - index[0]).days

        if time_interval >= self.CF * 365:

            coeff_capacity_factor_dict = {}
            print('Capacity Factor Constraint Activated ')
            for m in range(mode):
                dict_tempo = dict(zip(X[m], [1]*N))
                coeff_capacity_factor_dict.update(dict_tempo)

            ctr_capacity_factor = Constraint(0, name='ctr_capacity_factor')
            model.add(ctr_capacity_factor)
            model.constraints['ctr_capacity_factor'].ub = self.CF*365*24
            ctr_capacity_factor.set_linear_coefficients(coeff_capacity_factor_dict)
            self._cons['ctr_capacity_factor'] = ctr_capacity_factor

        # 1.6 Minimum shutdown time :

        # for i, t in enumerate(range(N-beta)):

        # ctr_min_time[i] = Constraint(sum(sum(X[m][t + k] for k in range(1, beta + 1)) for m in range(mode)) + beta * F[t],
        #     lb=0, ub=beta, name='ctr_min_time_' + str(t))

        # ctr_min_time[i] = Constraint(0, lb=0, name='ctr_min_time_' + str(t))
        # model.add(ctr_min_time[i])
        # model.constraints['ctr_min_time_' + str(t)].ub = beta
        # dict_tempo = {F[t]: beta}
        #
        # for m in range(mode):
        #
        #     dict_tempo_1 = dict(zip(X[m][t:(t+beta+1)], [1]*beta))
        #     dict_tempo.update(dict_tempo_1)
        #
        # ctr_min_time[i].set_linear_coefficients(dict_tempo)

        # Add other constraints and objective function
        # ------------------------------------------------------------------------------------------------

        model.add(cons_list)
        model.objective = obj_func
        obj_func.set_linear_coefficients(obj_coeff_dict)

        self.optim_model = model

        for m in range(mode):
            self._var['state_mode_' + str(m)] = X[m]

        self._var['Start'] = S
        self._var['Shut'] = F
        self._cons.update({'ctr_init_state': ctr_init_state, 'ctr_unique_mode': ctr_unique_mode, 'ctr_start_shut': ctr_start_shut,
                          'ctr_start_01': ctr_start_01, 'ctr_start_02': ctr_start_02})
        print('Object Creation Finished')

    def solve_optim_problem(self):

        print('Solving the problem - please wait')
        assert self.optim_model.objective.direction == 'max', 'We must maximize the objective function'
        self.optim_model.optimize()

        results = self.optim_model.status
        print('Solver Status :', results.upper())
        print('Objective Value : ', self.optim_model.objective.value)

        if results != 'optimal':
            print('The problem is infeasible or unbounded, please check again your configuration and data !!!!!')

    def solution_values(self):

        sol = self._var
        for key, val in sol.items():
            sol[key] = [vari.primal for vari in val]

        sol_df = pd.DataFrame(index=self.input_price.index, data=sol)
        sol_df = sol_df.astype('int')
        self.solutions = sol_df
        # self.solutions = sol_df.astype('int')

        # Operational Profile Calculation :
        # --------------------------------------------------------------------------------------------------------------

        # Rename the column as integer corresponding to the mode number
        state_mode = sol_df[sol_df.columns[sol_df.columns.str.contains('state_mode')]]
        col_str = state_mode.columns.tolist()
        col_num = [int(name.replace('state_mode_', '')) for name in col_str]
        state_mode.columns = col_num

        # Calculation
        energy_profile = state_mode.apply(lambda x: x * self.Energy[x.name], axis=0)
        self.Operation_Profile['Energy_Profile'] = energy_profile
        self.Operation_Profile['Energy_Profile']['Total'] = energy_profile.sum(axis=1)

        STMC = state_mode.apply(lambda x: x * (self.Commodity_Price.fuel_price[x.name] + self.Emission_Intensity[x.name]
                                               * self.input_price['carbon'] + self.Cost_var_OM), axis=0).sum(axis=1)

        index = self.input_price.index
        time_interval = (index[-1] - index[0]).days

        self.Operation_Profile['STMC'] = STMC
        self.Operation_Profile['Start-up Numbers'] = sol_df['Start'].sum()
        self.Operation_Profile['Capacity Factor'] = state_mode.sum().sum() / (365 * 24)

        STMC_nonzero = STMC.replace(0, np.nan) + \
                       self.Cost_fixed_OM * np.array(self.Power).max()*(time_interval/365)/(energy_profile.sum(axis=1).sum())

        self.Finance_Metrics['Revenue'] = (self.input_price['electricity']*energy_profile.sum(axis=1)).sum()
        self.Finance_Metrics['Gross Profit'] = self.optim_model.objective.value - \
                                             self.Cost_fixed_OM * np.array(self.Power).max()*time_interval/365
        self.Finance_Metrics['OPEX'] = -(self.Finance_Metrics['Gross Profit'] - self.Finance_Metrics['Revenue'])
        self.Finance_Metrics['Average STMC'] = STMC_nonzero.mean()

        print('Gross Profit: ', int(self.Finance_Metrics['Gross Profit'])*1e-6, '€ mil')

        return sol_df

    def visualization_non_interactive(self):

        energy_profile = self.Operation_Profile['Energy_Profile']['Total']
        STMC = self.Operation_Profile['STMC']
        Start = self.solutions['Start'] * self.Energy[0]
        Shut = self.solutions['Shut'] * self.Energy[0]

        fig, ax = plt.subplots(nrows=2, sharex=True)

        ax[0].step(energy_profile.index, energy_profile, label='Energy Generation', where='post', linewidth=1)
        ax[0].step(energy_profile.index, Start, label='Startup', color='green', where='post', linewidth=1)
        ax[0].step(energy_profile.index, Shut, label='Shutdown', color='red', where='post', linewidth=1)
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax[0].set_ylabel('MWh')
        ax[0].legend(shadow='False', loc='best')

        ax[1].step(self.input_price.index, self.input_price['electricity'],
                   label='DAM price', where='post', linewidth=1)
        ax[1].step(self.input_price.index, STMC, '--', label='STMC', where='post', linewidth=1, color='red')

        ax[1].set_ylabel('€/MWh')
        ax[1].legend(shadow='False', loc='best')

        fig.autofmt_xdate()
        plt.margins(0.02)
        plt.show()

    def visualization_interactive(self, mode=None):

        electricity = self.input_price.electricity
        energy_profile = self.Operation_Profile['Energy_Profile']['Total']
        STMC = self.Operation_Profile['STMC']
        Start = self.solutions['Start'] * self.Energy[0]
        Shut = self.solutions['Shut'] * self.Energy[0]
        old_df = pd.concat([electricity, energy_profile, STMC, Start, Shut], axis=1)
        old_df.columns = ['electricity', 'energy_profile', 'STMC', 'Start', 'Shut']
        old_df.loc[old_df['Start'] == 0, 'Start'] = np.nan
        df = old_df.reset_index()
        df['date_tooltips'] = [x.strftime("%Y-%m-%d %H:%M:%S") for x in df['time']]
        source = ColumnDataSource(df)

        fig = figure(x_axis_label='Time', y_axis_label='€/MWh', x_axis_type='datetime',
                     tools='wheel_zoom, box_zoom, pan, reset', logo=None
                     , plot_height=330, title='Price and Cost')

        fig0 = figure(x_axis_label='Time', y_axis_label='MWh', x_axis_type='datetime',
                      tools='wheel_zoom, box_zoom, pan, reset', plot_height=330, plot_width=1425,
                      title='Dispatch Profile')

        if mode == 'Notebook':
            fig.plot_width = 950
            fig0.plot_width = 950
            output_notebook()
        else:
            fig.plot_width = 1400
            fig0.plot_width = 1400
            output_file('First_Try_Generator.html')

        fig.background_fill_color = 'beige'
        fig.background_fill_alpha = 0.5

        fig.line(x='time', y='electricity', legend='DAM price', source=source)
        fig.line(x='time', y='STMC', legend='Marginal Cost', color='red', source=source)

        hover = HoverTool(tooltips=[('Time', '@date_tooltips'), ('DAM Price', '@electricity'), ('Marginal Cost', '@STMC')],
                          formatters={'time': 'datetime'})  # ('Time', '@time{%F}')
        fig.add_tools(hover)

        fig0.background_fill_color = 'beige'
        fig0.background_fill_alpha = 0.5

        fig0.line(x='time', y='energy_profile', legend='Generation', source=source)
        fig0.vbar(x='time', width=5, top='Start', legend='Start-up', fill_color='red', line_color='red', source=source)

        fig0.add_tools(hover)

        fig0.x_range = fig.x_range
        fig0.y_range = fig.y_range

        layout = column(fig0, fig)
        show(layout)


if __name__ == '__main__':

    data = pd.read_csv('data/CCGT_UK.csv', index_col='time', parse_dates=True)
    # price_data = data.loc['2015-08-01 00:00:00':'2015-10-01 23:00:00']      # to see the shut down effect : '2015-08-01 00:00:00':'2015-10-01 23:00:00'
    gen = Thermal_Generator(Name='Demo', price=data)

    tic = timeit.default_timer()
    gen.optimization_problem()
    toc_prob = timeit.default_timer()

    print('Intinialization time :', toc_prob - tic)

    gen.solve_optim_problem()
    toc_solve = timeit.default_timer()

    print('Solving time :', toc_solve - toc_prob)

    output = gen.solution_values()
    # gen.visualization_non_interactive()
    gen.visualization_interactive()
    print('Average STMC: ', gen.Finance_Metrics['Average STMC'])

    state_mode_out = output[output.columns[output.columns.str.contains('state_mode')]]

    print('Capacity Factor: ', state_mode_out.sum().sum()/(365*24))
    print('Number of Startup: ', gen.solutions['Start'].sum())
    # print(gen.optim_model.constraint_values)