from optlang import Objective, Variable, Constraint, Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
import timeit

sns.set()
plt.rc('font', size=15)

# Generator Parameters


class Thermal_Generator(object):

    def __init__(self, price: pd.DataFrame, Efficiency=[0.58, 0.47], Power=[100, 40], Emission_Factor=0.25, CF=0.75, Startup_cold_time=2, Startup_dep_cost=60,
                 Startup_fuel=2.8, Startup_fixed_cost=7000, Name='Thermal_Generator'):

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
        self.Startup_fixed_cost = Startup_fixed_cost    # Fixed Startup Cost [€], 7000

        self.Cost_fixed_OM = 30568*0.9189       # [€/MW/year] [IEA2015, 0.81 is the 2018 exchange rate in 2018, 0.9189 is in 2015] 30568 for ccgt , 41303 for coal-fired, 30568*0.9189
        self.Cost_var_OM = 2.5*0.9189           # [€/MWh_e] [IEA 2015, the 0.81 is the exchange rate in 2018] 2.5 for ccgt, 4.4 for coal.

        self.N_mode = len(self.Efficiency)

        class empty_class:
            pass

        self.input_price = price
        self.Commodity_Price = empty_class
        self.Commodity_Price.electricity_price = price['electricity'].values
        self.Commodity_Price.gas_price = price['gas'].values
        self.Commodity_Price.carbon_price = price['carbon'].values
        self.Commodity_Price.fuel_price = np.zeros((self.N_mode, self.Horizon))

        for n in range(self.N_mode):
            self.Commodity_Price.fuel_price[n] = self.Commodity_Price.gas_price/self.Efficiency[n]

        self.Operation_Profile = {'Energy_Generation': [], 'STMC': [], 'Energy_Profile': None}
        self.optim_model = None
        self.solutions = None
        self.var = {}
        self.cons = {}

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
        price_elec = self.Commodity_Price.electricity_price
        price_carbon = self.Commodity_Price.carbon_price
        price_fuel = self.Commodity_Price.fuel_price
        price_gas = self.Commodity_Price.gas_price

        # Variable Registration
        # -----------------------------------------------------------------------------------
        model = Model(name='Thermal Generation')
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

        for m in range(mode):

            ctr_init_state[m] = [[]]*(alpha+1)

            for t in range(alpha+1):
                ctr_init_state[m][t] = Constraint(X[m][t], lb=0, ub=0, name='ctr_initial_states_m_' + str(m) + str(t))

        for t in range(N):

            # 1.1 Unique mode constraint:
            ctr_unique_mode[t] = Constraint(sum(X[m][t] for m in range(mode)), ub=1, name='ctr_unique_mode_' + str(t))

            # 1.2 Startup - shutdown constraint:
            ctr_start_shut[t] = Constraint(S[t] + F[t], ub=1, name='ctr_start_shut_' + str(t))

        for i, t in enumerate(range(alpha, N-alpha)):     # 2, 21

            # 1.3 Startup - shutdown constraint 2 :
            ctr_start_01[i] = Constraint(S[t - alpha] - F[t] - sum(X[m][t+1] - X[m][t] for m in range(mode)), lb=0, ub=0, name='ctr_start_01_' + str(t))

            # 1.4 Minimum startup time :
            ctr_start_02[i] = Constraint(sum(sum(X[m][t - k] for k in range(1, alpha)) for m in range(mode)) + alpha*S[t-alpha],
                                         ub=alpha, name='ctr_start_02_' + str(t))

        ctr_start_02[N-alpha-1] = Constraint(sum(sum(X[m][N - 1 - k] for k in range(1, alpha)) for m in range(mode)) + alpha*S[N - 1 - alpha],
                                        ub=alpha, name='ctr_start_02_' + str(N - 1))

        # Objective function :
        # -----------------------------------------------------------------------------------

        print('Objective')
        obj_list = [[]]*(mode+1)
        obj_func = Objective(0, direction='max')

        obj_list[-1] = Objective(-sum(self.Power[0]*S[t]*(self.Startup_dep_cost + self.Startup_fuel*price_gas[t]) for t in range(N)), direction='max')

        for m in range(mode):

            obj_list[m] = Objective(energy[m]*sum(price_elec[t]*X[m][t] - price_fuel[m, t]*X[m][t] - price_carbon[t]*self.Emission_Intensity[m]*X[m][t]
                                                   - X[m][t]*self.Cost_var_OM for t in range(N)), direction='max')

        for elem in obj_list:

            obj_func += elem.expression

        print('DOne with objective')
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

        model.add(var_list)
        model.add(cons_list)

        model.objective = obj_func
        self.optim_model = model

        for m in range(mode):
            self.var['state_mode_' + str(m)] = X[m]

        self.var['Start'] = S
        self.var['Shut'] = F
        self.cons = {'ctr_init_state': ctr_init_state, 'ctr_unique_mode': ctr_unique_mode, 'ctr_start_shut': ctr_start_shut,
                     'ctr_start_01': ctr_start_01, 'ctr_start_02': ctr_start_02}

    def solve_optim_problem(self, solver='glpk'):

        print('Solving the problem - please wait')
        assert self.optim_model.objective.direction == 'max', 'We must maximize the objective function'
        self.optim_model.optimize()

        results = self.optim_model.status
        print('Solver Status :', results.upper())
        print('Objective Value : ', self.optim_model.objective.value)

        return results

    def solution_values(self):

        sol = self.var

        for key, val in sol.items():

            sol[key] = [vari.primal for vari in val]

        sol_df = pd.DataFrame(index=self.input_price.index, data=sol)
        self.solutions = sol_df

        return sol_df

    def profile_visualization(self, start=None, end=None):

        if start is None or end is None:
            data_vis = self.solutions
        else:
            data_vis = self.solutions.loc[start:end]

        start_data = data_vis['Start']*self.Energy[0]
        shut_data = data_vis['Shut'] * self.Energy[0]
        # start_data[start_data == 0] = np.nan
        # shut_data[shut_data == 0] = np.nan

        fig, ax = plt.subplots(nrows=2, sharex=True)

        state_mode = data_vis[data_vis.columns[data_vis.columns.str.contains('state_mode')]]

        # Convert from state to integer
        col_str = state_mode.columns.tolist()
        col_num = [int(name.replace('state_mode_', '')) for name in col_str]
        state_mode.columns = col_num
        energy_profile = state_mode.apply(lambda x: x * self.Energy[x.name], axis=0)
        energy_mode = state_mode.apply(lambda x: x * self.Energy[x.name], axis=0).sum(axis=1)

        self.Operation_Profile['Energy_Generation'] = energy_mode
        self.Operation_Profile['Energy_Profile'] = energy_profile

        # Calculating STMC :

        STMC = state_mode.apply(lambda x: x*(self.Commodity_Price.fuel_price[data_vis.index, x.name] +
                self.Emission_Intensity[x.name]*self.input_price['carbon'] + self.Cost_var_OM), axis=0).sum(axis=1)

        self.Operation_Profile['STMC'] = STMC

        ax[0].step(energy_mode.index, energy_mode, label='Energy Generation', where='post', linewidth=1)
        ax[0].step(energy_mode.index, start_data, label='Startup', color='green', where='post', linewidth=1)
        ax[0].step(energy_mode.index, shut_data, label='Shutdown', color='red', where='post', linewidth=1)
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax[0].set_ylabel('MWh')
        ax[0].legend(shadow='False', loc='best')

        ax[1].step(self.input_price.index, self.input_price.loc[start:end, 'electricity'],
                   label='DAM price', where='post', linewidth=1)
        ax[1].step(self.input_price.index, STMC, '--', label='STMC', where='post', linewidth=1, color='red')

        ax[1].set_ylabel('€/MWh')
        ax[1].legend(shadow='False', loc='best')

        fig.autofmt_xdate()
        plt.margins(0.02)
        plt.show()


if __name__ == '__main__':

    data = pd.read_csv('agg_data.csv', index_col='time', parse_dates=True)
    price_data = data.loc['2015-06-15 00:00:00':'2015-11-22 23:00:00']      # to see the shut down effect : '2015-08-01 00:00:00':'2015-10-01 23:00:00'
    gen = Thermal_Generator(price=price_data)

    tic = timeit.default_timer()
    gen.optimization_problem()
    toc_prob = timeit.default_timer()

    print('Intinialization time :', toc_prob - tic)

    gen.solve_optim_problem()
    toc_solve = timeit.default_timer()

    print('Solving time :', toc_solve - toc_prob)

    output = gen.solution_values()
    gen.profile_visualization()
    # print(output)
