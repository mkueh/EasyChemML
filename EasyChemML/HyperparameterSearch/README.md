# Hyperparameter-Search

## Usage

###Hyperparameter Search
 
select the hyperparameter search algorithm in the settings.hjson under hyperparamterSearch_settings
    
possible search algorithm:
    
    - grid_search: A normal Gridsearch
        -- needs the step parameter in int,float Hyperparameter
        -- no param needed
        
####Example

    using_searchalgo: 'grid_search'
    using_searchalgo_param: {} 

###Hyper-Parameter

EasyChemML knows three typs of Hyperparameter

    - intRange      : int
    - floatRange    : flt
    - Categorically : cat

for define a Parameter as hyperparameter (at the moment only possible in the model_settings part) it is necessary to define 
the parameter as dict with the following entries:

    - for int
    {typ: int, range:[start,end,(step)], (count)}
    
    - for float
    {typ: flt, range:[start,end,(step)], (count)}
    
    -for cat
    {typ: cat, items:[x_1,...,x_n]}
    
    Parameters surrounded by () are only required by some hyperparameter search algorithms
    
If a parameter is already a dict, it can no longer be processed by the tool. The interpreter always assumes that if a parameter is a dict, it is also a hyperparameter. In this case the parameter must be packed in a Categorical Hyperparameter.

####Examples

    using_algo : 'scikit_randomforest_r'
    using_algosettings : {'n_estimators': {typ:cat, items:[100,1000,10000]},'random_state':42,'n_jobs': -2}
    
A dict as Parameter (wrapped in a categorical parameter)

    using_algosettings : {'coulmns': {typ:cat, items:[{feature:'x', target:'y'}]},'random_state':42,'n_jobs': -2}

 ## Contact
 - https://www.wi.uni-muenster.de/de/institut/pi/personen/marius-kuehnemund-1
 - https://www.uni-muenster.de/Chemie.oc/glorius/felix_strieth_kalthoff.html
 - https://www.uni-muenster.de/Chemie.oc/glorius/philipp_pflueger.html
 - https://www.uni-muenster.de/Chemie.oc/glorius/frederik_sandfort.html