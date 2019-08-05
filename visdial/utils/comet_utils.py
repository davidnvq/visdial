from comet_ml import Experiment, OfflineExperiment


def get_comet_experiment(config, is_online=False):
    if is_online:
        return Experiment(api_key='2z9VHjswAJWF1TV6x4WcFMVss',
                          project_name=config['comet_project'],
                          workspace='lightcv')
    else:
        return OfflineExperiment(
            project_name=config['comet_project'],
            workspace='lightcv',
            offline_directory="/home/quanguet/comet_tmp")
