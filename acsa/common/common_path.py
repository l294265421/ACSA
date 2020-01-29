import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

original_data_dir = project_dir + '/datasets/'

common_data_dir = project_dir + '/data/'
common_code_dir = project_dir + '/acsa/'

stopwords_filepath = original_data_dir + 'common/stopwords.txt'


def get_task_data_dir(is_original=False):
    """

    :return: 保存子任务的数据的目录
    """
    if not is_original:
        return '%s/' % common_data_dir
    else:
        return '%s/' % original_data_dir
