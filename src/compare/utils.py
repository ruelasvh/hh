'''
utils.py

Some relevant dictionaries and functions for running ...
'''
import pandas as pd

fileDirSLAC = {

	# Sig
	'X200_S70_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801577.Py8EG_A14NNPDF23LO_XHS_X200_S70_4b.e8448_a899_r13145_p5511_TREE/',   
	'X300_S70_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801578.Py8EG_A14NNPDF23LO_XHS_X300_S70_4b.e8448_a899_r13145_p5511_TREE/',
	'X300_S100_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801579.Py8EG_A14NNPDF23LO_XHS_X300_S100_4b.e8448_a899_r13145_p5511_TREE/', 
	'X300_S170_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801580.Py8EG_A14NNPDF23LO_XHS_X300_S170_4b.e8448_a899_r13145_p5511_TREE/',
	
	'X400_S70_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801581.Py8EG_A14NNPDF23LO_XHS_X400_S70_4b.e8448_a899_r13145_p5511_TREE/',
	'X400_S100_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801582.Py8EG_A14NNPDF23LO_XHS_X400_S100_4b.e8448_a899_r13145_p5511_TREE/', 
	'X400_S170_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801583.Py8EG_A14NNPDF23LO_XHS_X400_S170_4b.e8448_a899_r13145_p5511_TREE/',
	'X400_S200_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801584.Py8EG_A14NNPDF23LO_XHS_X400_S200_4b.e8448_a899_r13145_p5511_TREE/',
	'X400_S250_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801585.Py8EG_A14NNPDF23LO_XHS_X400_S250_4b.e8448_a899_r13145_p5511_TREE/',

	'X750_S70_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801586.Py8EG_A14NNPDF23LO_XHS_X750_S70_4b.e8448_a899_r13145_p5511_TREE/',
	'X750_S100_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801587.Py8EG_A14NNPDF23LO_XHS_X750_S100_4b.e8448_a899_r13145_p5511_TREE/',
	'X750_S170_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801588.Py8EG_A14NNPDF23LO_XHS_X750_S170_4b.e8448_a899_r13145_p5511_TREE/', 
	'X750_S200_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801589.Py8EG_A14NNPDF23LO_XHS_X750_S200_4b.e8448_a899_r13145_p5511_TREE/',
	'X750_S250_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801590.Py8EG_A14NNPDF23LO_XHS_X750_S250_4b.e8448_a899_r13145_p5511_TREE/',
	'X750_S300_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801591.Py8EG_A14NNPDF23LO_XHS_X750_S300_4b.e8448_a899_r13145_p5511_TREE/',
	'X750_S400_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801592.Py8EG_A14NNPDF23LO_XHS_X750_S400_4b.e8448_a899_r13145_p5511_TREE/',
	'X750_S500_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801593.Py8EG_A14NNPDF23LO_XHS_X750_S500_4b.e8448_a899_r13145_p5511_TREE/',
	
	'X1000_S70_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801594.Py8EG_A14NNPDF23LO_XHS_X1000_S70_4b.e8448_a899_r13145_p5511_TREE/',
	'X1000_S100_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801595.Py8EG_A14NNPDF23LO_XHS_X1000_S100_4b.e8448_a899_r13145_p5511_TREE/',
	'X1000_S170_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801596.Py8EG_A14NNPDF23LO_XHS_X1000_S170_4b.e8448_a899_r13145_p5511_TREE/',
	'X1000_S200_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801597.Py8EG_A14NNPDF23LO_XHS_X1000_S200_4b.e8448_a899_r13145_p5511_TREE/',
	'X1000_S250_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801598.Py8EG_A14NNPDF23LO_XHS_X1000_S250_4b.e8448_a899_r13145_p5511_TREE/',
	'X1000_S300_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801599.Py8EG_A14NNPDF23LO_XHS_X1000_S300_4b.e8448_a899_r13145_p5511_TREE/',
	'X1000_S400_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801600.Py8EG_A14NNPDF23LO_XHS_X1000_S400_4b.e8448_a899_r13145_p5511_TREE/',
	'X1000_S500_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801601.Py8EG_A14NNPDF23LO_XHS_X1000_S500_4b.e8448_a899_r13145_p5511_TREE/',
	'X1000_S750_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/signals/user.dabattul.ntup_SH4b_AF3_24022023.801602.Py8EG_A14NNPDF23LO_XHS_X1000_S750_4b.e8448_a899_r13145_p5511_TREE/',

	# pythia QCD samples
	'JZ1_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/QCD/user.dabattul.ntup_bkg_24022023.364701.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW.e7142_s3681_r13145_p5511_TREE/',
	'JZ2_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/QCD/user.dabattul.ntup_bkg_24022023.364702.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW.e7142_s3681_r13145_p5511_TREE/',
	'JZ3_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/QCD/user.dabattul.ntup_bkg_24022023.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.e7142_s3681_r13145_p5511_TREE/',
	'JZ4_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/QCD/user.dabattul.ntup_bkg_24022023.364704.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW.e7142_s3681_r13145_p5511_TREE/',
	'JZ5_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/QCD/user.dabattul.ntup_bkg_24022023.364705.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW.e7142_s3681_r13145_p5511_TREE/',
	# all-had ttbar
	'allhad_ttbar_mc20e-MAR23': '/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/ttbar/user.dabattul.ntup_bkg_24022023.410471.PhPy8EG_A14_ttbar_hdamp258p75_allhad.e6337_s3681_r13145_p5511_TREE/',

	# data
	'data18-MAR23':'/gpfs/slac/atlas/fs1/d/nhartman/SH4b/data/data/user.dabattul.ntup_data_24022023.data18_13TeV.period*_TREE/',
	
}


# Slimmed down version of the triggers from utils for j the NR ones
triggers_NR = {
    2016: ['HLT_2j35_bmv2c2060_split_2j35_L14J15p0ETA25',
           'HLT_j100_2j55_bmv2c2060_split'],
    2017: ['HLT_2j15_gsc35_bmv2c1040_split_2j15_gsc35_boffperf_split_L14J15p0ETA25',
           'HLT_j110_gsc150_boffperf_split_2j35_gsc55_bmv2c1070_split_L1J85_3J30'],
    2018: ['HLT_2j35_bmv2c1060_split_2j35_L14J15p0ETA25',
           'HLT_j110_gsc150_boffperf_split_2j45_gsc55_bmv2c1070_split_L1J85_3J30']
}

# Parsing years
mcToYr = {'mc20a': 2016,
          'mc20d': 2017,
          'mc20e': 2018,
    	  'r13167': 2016, # mc20a
          'r13144': 2017,  # mc20d
          'r13145': 2018, # mc20e
        }

L = {
        15  : 3.2,
        16  : 24.6,
        17  : 43.65,
        18  : 57.7,
        #
        2015: 3.2,
        2016: 24.6,
        2017: 43.65,
        2018: 57.7
    }


physToDSID = {'semilep_ttbar': 410470,
              'allhad_ttbar':  410471,
			  'JZ0': 364700,
			  'JZ1': 364701,
			  'JZ2': 364702,
              'JZ3': 364703,
              'JZ4': 364704,
	      	  'JZ5': 364705,
			  'JZ6': 364706,
			  'JZ7': 364707,
			  'JZ8': 364708,
			  'JZ9': 364709,
			  'JZ10':3647010,
			  'JZ11':3647011,
			  'JZ12':3647012,
			  # Scalar signals
			  'X200_S70' : 801577,   
			  'X300_S70' : 801578,
			  'X300_S100': 801579, 
			  'X300_S170': 801580,
	
			  'X400_S70' : 801581,
			  'X400_S100': 801582, 
			  'X400_S170': 801583,
			  'X400_S200': 801584,
			  'X400_S250': 801585,

			  'X750_S70':  801586,
			  'X750_S100': 801587,
			  'X750_S170': 801588, 
			  'X750_S200': 801589,
			  'X750_S250': 801590,
			  'X750_S300': 801591,
			  'X750_S400': 801592,
			  'X750_S500': 801593,
	
			  'X1000_S70':  801594,
			  'X1000_S100': 801595,
			  'X1000_S170': 801596,
			  'X1000_S200': 801597,
			  'X1000_S250': 801598,
			  'X1000_S300': 801599,
			  'X1000_S400': 801600,
			  'X1000_S500': 801601,
			  'X1000_S750': 801602,
			  # These were the JZ slices we were looking at for the R21 HH4b analysis
            #   'JZ2': 800285,
            #   'JZ3': 800286,
            #   'JZ4': 800287
			# New NR default - pythia
              'SMNR_pythia':600463,
              'k10_pythia': 600464,
			  # herwig -- alt
			  'SMNR':600043,
              'k10': 600044
             }

stdCols = ["mc_sf", "njets","ntag","m_SH","dEta_SH","X_SH"
           # HC cols
           "pt_h1","eta_h1","phi_h1","m_h1",
           "pt_h2", "eta_h2", "phi_h2","m_h2",
		   ]

truthCols = ['correct']
# ,'sameParent','unique','dRmatch','goodJets',
#              'truth_mhh','truth_pthh',
#              'h0_pt','h0_eta','h0_phi','h0_barcode',
#              'h1_pt','h1_eta','h1_phi','h1_barcode',
#              'b0_pt','b0_eta','b0_phi','b0_parent_barcode','b0_jidx','b0_drMatch',
#              'b1_pt','b1_eta','b1_phi','b1_parent_barcode','b1_jidx','b1_drMatch',
#              'b2_pt','b2_eta','b2_phi','b2_parent_barcode','b2_jidx','b2_drMatch',
#              'b3_pt','b3_eta','b3_phi','b3_parent_barcode','b3_jidx','b3_drMatch']

def getJetCols(vs=['pt','eta','phi','E'] ,nSelectedJets=4,truth=False):
	'''
	Return the kinmatics of the jets considered (i.e, for pairAGraph or other baselines)
	'''
	# if truth: vs += ['bidx','drMatch']

	return [f'j{i}_{v}' for i in range(nSelectedJets) for v in vs]

# def getTruthJetCols(vs=['idx','dR','pt','eta','phi','E'] ,nSelectedJets=4):
# 	'''
# 	Return the 4-vec for the truth matched jets
# 	'''

# 	return [f'j{i}_t{v}' for i in range(nSelectedJets) for v in vs]


def getSubDir(physicsSample,mc,prodTag,nSelectedJets,pTcut=40,**kwargs):
	'''
	Reconstructs the subDir for the sample settings with my directory naming convention
	'''

	if 'data' in physicsSample:
		subDir = f'{physicsSample}-{prodTag}'
	else:
		subDir = f'{physicsSample}_{mc}-{prodTag}'
	if nSelectedJets != 4:
		subDir += f'-{nSelectedJets}jets'
	if pTcut != 40:
		subDir += f'-{pTcut}GeV'
	return subDir

def getDataTag(inputFile):
	'''
	'''
	parts = inputFile.split('.')
	periodTag = parts[7]
	fileNumTag = parts[-3]

	if physicsSample is None: physicsSample = parts[8]
	if prodTag is None: prodTag = inputFile.split('-')[1]
	if year is None: year = 2000+int(physicsSample[-2:])


def read_tsv(tsvFile = "../data/xsec.tsv"):
	'''
	Read  the tsv file (this function assumes that my dihiggs repo is parallel
	to the hh-resolved-reconstruction repo), and it returns a pandas DataFrame
	with the rows corresponding to the DISD numbers, and the columns to the
	relevant physics information that we want to extract.
	'''

	# To figure out the format of the .tsv file, I looked at Beojean's MCConfig.cpp file.
	db_cols = ['physics_short','xsec','gen_filter_eff','k_factor','rel_uncert_up','rel_uncert_down','generator']
	db_entry = pd.read_csv(tsvFile,sep='\t \t|\t\t|\t',index_col=0,engine='python',names=db_cols)

	db_entry['xsec'] *= 1000

	return db_entry


def get_xsec(kl):
    '''
    Taken from lhcxswg twiki
    '''
    return 70.3874-50.4111*kl+11.0595*(kl**2)
