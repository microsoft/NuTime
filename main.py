import os
import copy
import warnings
import itertools
from config import *
from pipeline import *

import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

dataset_dict = {
    'ucr': ['DiatomSizeReduction','InsectWingbeatSound','EthanolLevel','PowerCons','SmoothSubspace','Coffee','ACSF1','Lightning7','InlineSkate','CricketY','EOGVerticalSignal','LargeKitchenAppliances','GestureMidAirD2','Plane','AllGestureWiimoteX','SonyAIBORobotSurface1','ShapesAll','DodgerLoopWeekend','Meat','FaceAll','Earthquakes','MixedShapesSmallTrain','PhalangesOutlinesCorrect','HandOutlines','Strawberry','Wafer','HouseTwenty','Yoga','ECG5000','BeetleFly','ProximalPhalanxTW','Ham','GestureMidAirD1','UWaveGestureLibraryX','DodgerLoopDay','TwoLeadECG','ItalyPowerDemand','MedicalImages','GesturePebbleZ1','ECG200','NonInvasiveFetalECGThorax2','SwedishLeaf','ECGFiveDays','ShakeGestureWiimoteZ','CricketX','DodgerLoopGame','UWaveGestureLibraryAll','PigCVP','Worms','PigArtPressure','StarLightCurves','NonInvasiveFetalECGThorax1','ElectricDevices','WordSynonyms','GesturePebbleZ2','Herring','FacesUCR','GunPointOldVersusYoung','MelbournePedestrian','ToeSegmentation2','WormsTwoClass','DistalPhalanxOutlineCorrect','MoteStrain','UMD','FreezerSmallTrain','SmallKitchenAppliances','SyntheticControl','DistalPhalanxTW','BME','SemgHandSubjectCh2','Computers','Chinatown','Trace','Rock','UWaveGestureLibraryZ','SemgHandGenderCh2','UWaveGestureLibraryY','Adiac','Beef','TwoPatterns','Haptics','Phoneme','PickupGestureWiimoteZ','ScreenType','MixedShapesRegularTrain','FordA','AllGestureWiimoteZ','Fish','MiddlePhalanxTW','AllGestureWiimoteY','ProximalPhalanxOutlineCorrect','GunPointAgeSpan','BirdChicken','InsectEPGRegularTrain','GunPointMaleVersusFemale','InsectEPGSmallTrain','MiddlePhalanxOutlineAgeGroup','OliveOil','Car','FreezerRegularTrain','ProximalPhalanxOutlineAgeGroup','FordB','GunPoint','DistalPhalanxOutlineAgeGroup','Symbols','Crop','Wine','SonyAIBORobotSurface2','RefrigerationDevices','FiftyWords','ToeSegmentation1','PLAID','ChlorineConcentration','OSULeaf','CricketZ','FaceFour','Fungi','SemgHandMovementCh2','CinCECGTorso','MiddlePhalanxOutlineCorrect','CBF','Lightning2','PigAirwayPressure','EOGHorizontalSignal','Mallat','GestureMidAirD3','ArrowHead','ShapeletSim'],
    'uea': ['HandMovementDirection','PEMS-SF','Heartbeat','LSST','Cricket','ArticularyWordRecognition','SelfRegulationSCP2','CharacterTrajectories','EthanolConcentration','Libras','StandWalkJump','JapaneseVowels','EigenWorms','BasicMotions','Handwriting','FaceDetection','RacketSports','FingerMovements','NATOPS','SelfRegulationSCP1','AtrialFibrillation','DuckDuckGeese','UWaveGestureLibrary','SpokenArabicDigits','PenDigits','MotorImagery','PhonemeSpectra','InsectWingbeat','Epilepsy','ERing'],
    'tfc': ['SleepEEG', 'HAR', 'Gesture', 'Epilepsy', 'FD-A', 'FD-B', 'ECG', 'EMG'],
    'all-merged': ['all-merged'],
    'ucr-merged': ['ucr-merged']
}

def run(config):
    pipeline = Pipeline(config)
    save_config(config)
    best_test_acc, best_test_mf1 = pipeline.run_exp()
    return best_test_acc, best_test_mf1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Representation')

    # Arguments
    parser.add_argument('--config_path', type=str, default='./configs/default_ssl.json')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--archives', type=str, default='ucr,uea,tfc')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--exp_tag', type=str, default='test')
    parser.add_argument('--load_checkpoint_path', type=str, default='')
    parser.add_argument('--few_shot', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    
    args = parser.parse_args()
    config_path = args.config_path
    seeds = args.seeds
    archive_list = args.archives.split(',')
    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    output_dir = args.output_dir
    exp_tag = args.exp_tag
    load_checkpoint_path = args.load_checkpoint_path
    few_shot = args.few_shot
    force = args.force
    
    # Read the default config
    default_config = json.load(open(config_path, 'r'))
    for archive in archive_list:
        dataset_list = dataset_dict[archive]
        for seed in range(1, seeds+1):
            for dataset in dataset_list:
                # Define customized config
                custom_config = {}
                custom_config['dataset'] = dataset
                custom_config['seed'] = seed
                custom_config['load_checkpoint_path'] = load_checkpoint_path
                
                config = Config()
                config.update_by_dict(default_config)
                config.update_by_dict(custom_config)
                
                # Load few shot config
                if few_shot:
                    few_shot_config = json.load(open('./configs/default_few_shot.json', 'r'))
                    config.update_by_dict(few_shot_config)
                
                config.set_tag()
                config.dataset_dir = dataset_dir
                config.output_dir = f'{output_dir}/{exp_tag}/s{seed}'
                config.log_dir = f'{log_dir}/{archive}/{exp_tag}/s{seed}'
                config.log_file = f'{config.log_dir}/{config.tag}'
                config.transformer_mlp_dim = config.transformer_heads * config.transformer_head_dim
                if dataset == 'InsectWingbeatSound' or dataset == 'InsectWingbeat':
                    config.batch_size = 512
                    config.eval_batch_size = 512
                
                # Run the experiment
                if not os.path.exists(config.log_file) or force:
                    print(f"Start run {dataset}, log: {config.log_file}")
                    best_test_acc, best_test_mf1 = run(config)