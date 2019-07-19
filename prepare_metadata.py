import os

from common_blocks import utils

utils.check_env_vars()
CONFIG = utils.read_config(config_path=os.getenv('CONFIG_PATH'))
LOGGER = utils.init_logger()

#    ______   ______   .__   __.  _______  __    _______      _______.
#   /      | /  __  \  |  \ |  | |   ____||  |  /  _____|    /       |
#  |  ,----'|  |  |  | |   \|  | |  |__   |  | |  |  __     |   (----`
#  |  |     |  |  |  | |  . `  | |   __|  |  | |  | |_ |     \   \
#  |  `----.|  `--'  | |  |\   | |  |     |  | |  |__| | .----)   |
#   \______| \______/  |__| \__| |__|     |__|  \______| |_______/
#

PARAMS = CONFIG.parameters


#   __________   ___  _______   ______  __    __  .___________. __    ______   .__   __.
#  |   ____\  \ /  / |   ____| /      ||  |  |  | |           ||  |  /  __  \  |  \ |  |
#  |  |__   \  V  /  |  |__   |  ,----'|  |  |  | `---|  |----`|  | |  |  |  | |   \|  |
#  |   __|   >   <   |   __|  |  |     |  |  |  |     |  |     |  | |  |  |  | |  . `  |
#  |  |____ /  .  \  |  |____ |  `----.|  `--'  |     |  |     |  | |  `--'  | |  |\   |
#  |_______/__/ \__\ |_______| \______| \______/      |__|     |__|  \______/  |__| \__|
#

def prepare_stacking_data():
    LOGGER.info('preparing stacking metadata')
    raw_dir = os.path.join(PARAMS.stacking_data_dir, 'raw')
    grouped_by_id_dir = os.path.join(PARAMS.stacking_data_dir, 'predictions_by_id')
    joined_dir = os.path.join(PARAMS.stacking_data_dir, 'joined_predictions')

    for dirpath in [PARAMS.stacking_data_dir, grouped_by_id_dir, joined_dir]:
        os.makedirs(dirpath, exist_ok=True)

    LOGGER.info('grouping predictions by id')
    utils.group_predictions_by_id(raw_dir=raw_dir, grouped_by_id_dir=grouped_by_id_dir)
    LOGGER.info('joining predictions')
    utils.join_id_predictions(grouped_by_id_dir=grouped_by_id_dir, joined_predictions_dir=joined_dir)
    meta = utils.generate_metadata_stacking(metadata_filepath=PARAMS.metadata_filepath,
                                            joined_predictions_dir=joined_dir)
    meta.to_csv(PARAMS.metadata_filepath, index=None)


def prepare_metadata():
    LOGGER.info('creating metadata')
    meta = utils.generate_metadata(train_images_dir=PARAMS.train_images_dir,
                                   test_images_dir=PARAMS.test_images_dir,
                                   depths_filepath=PARAMS.depths_filepath
                                   )
    meta.to_csv(PARAMS.metadata_filepath, index=None)


#  .___  ___.      ___       __  .__   __.
#  |   \/   |     /   \     |  | |  \ |  |
#  |  \  /  |    /  ^  \    |  | |   \|  |
#  |  |\/|  |   /  /_\  \   |  | |  . `  |
#  |  |  |  |  /  _____  \  |  | |  |\   |
#  |__|  |__| /__/     \__\ |__| |__| \__|
#

if __name__ == '__main__':
    prepare_metadata()
    # prepare_stacking_data()
