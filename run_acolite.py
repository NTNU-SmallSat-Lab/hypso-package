# https://odnature.naturalsciences.be/remsem/acolite-forum/viewtopic.php?t=238

# add acolite clone to Python path and import acolite
import sys, os
user_home = os.path.expanduser("~")
sys.path.append(user_home+'/Projects/acolite')
import acolite as ac

# scenes to process
bundles = ['/home/cameron/Dokumenter/Data/frohavet/frohavet_2024-05-06_1017Z-l1b.nc']
# alternatively use glob
# import glob
# bundles = glob.glob('/path/to/scene*')

# output directory
odir = '/home/cameron/Dokumenter/Data/frohavet/'

# optional 4 element limit list [S, W, N, E] 
limit = None

# optional file with processing settings
# if set to None defaults will be used
settings_file = None

# run through bundles
for bundle in bundles:
    # import settings
    settings = ac.acolite.settings.load(settings_file)
    # set settings provided above
    settings['limit'] = limit
    settings['inputfile'] = bundle
    settings['output'] = odir
    # other settings can also be provided here, e.g.
    # settings['s2_target_res'] = 60
    # settings['dsf_path_reflectance'] = 'fixed'
    # settings['l2w_parameters'] = ['t_nechad', 't_dogliotti']

    # process the current bundle
    ac.acolite.acolite_run(settings=settings)