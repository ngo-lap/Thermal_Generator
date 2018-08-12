# Thermal_Generator
First Official Version for Thermal_Generator

## General Description 

`Thermal_Generator` is a package specifically designed to simulate the operational plan (dispatch profile) for thermal generators (most notably coal-fired, gas-fired and combined cycle gas turbine generators)
participating in the electricity day-ahead market. A linear optimization model will be generated based on the given prices and configuration for the 
generator within the specified time horizon. The model is then solved using `glpk` solver and then the operational and economic metrics will be calculated
based on the obtained solutions.

The package also provides the function `EPEX_Scrapping` for scrapping the electricity day-ahead market price published on 
[EPEX SPOT Aunction Data](https://www.epexspot.com/en/market-data/dayaheadauction) for three different regions : France, Germany and Czech Republic.

## Who is it for? 

This project is created based on the need for a realistic and robust tool for feasibility analysis of energy portfolios involving coal-fired and gas-fired 
power generators participating in day-ahead markets. These analyses are a crucial part of the energy project management coursework in the Master Program -
Energy Environment: Science Technology and Management ([STEEM](https://portail.polytechnique.edu/graduatedegree/master/steem/program-structure)) at Ecole Polytechnique. 

## Future Prospect 

We are currently considering the possibility of incorporating a storage system into this project in order to widen the applicable types of generators
(hydroelectric generator or a renewable system with a battery system). Any contribution and feedback about this possibility as well as the current project will 
be greatly appreciated. 

## Dependencies 

This project relies on : the Python package for optimization problem [optlang](https://pypi.org/project/optlang/) for the model formulation, 
the interactive visualization library [Bokeh](https://bokeh.pydata.org/en/latest/) and the standard `Matplotlib` and `seaborn` visualization packages. After installinf `optlang`, necessary components for the model need to be imported as follows : 

`from optlang import Objective, Variable, Constraint, Model`

The `EPEX_Scrapping` function also requires `requests` and `BeautifulSoup` which are already installed in the standard Anaconda Distribution. 

## Instruction 

We suggest you to read through the `Model and Syntax` notebook first to apprehend the basic assumption of the model. Then the `Tutorials` notebook will briefly demonstrates the procedure for working with `Thermal_Generator`. Finally, a case study is provided in `Case Study - Carbon Price Mechanisms` notebook, which will apply `Thermal_Generator` to investigate the carbon price mechanisms on generators running on coal and gas in two different market : Germany and the United Kingdom.
