# Copyright 2019 The Lowes UNCC group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ================================
"""Generic Utility functions"""

import os
from pathlib import Path

def get_absolute_data_path(data_workspace: str = 'data', parent_up_limit=2, sub_folder_name: str = None):
    """
    Gets the absolute path for the 'data' directory.

    ***Note that this assumes the script is in the lowes-product-classifier***

    :param sub_folder_name: A sub folder name. It is the user's responsibility if the sub folder has separators such as
    '/' or '\'. Please use os.sep if this is the case.
    :param data_workspace: Name of the workspace. Default is 'data'
    :param parent_up_limit: The number of upper folders to look through to find the directory
    :return: The absolute path to the workspace. IE a string like:
    /Users/jlaivins/PycharmProjects/Lowes-Product-Classifier/lowes-product-classifier/data/
    """
    absolute_path = ''
    for i in range(-1, parent_up_limit):
        if i == -1:
            curr = str(Path().absolute())
        else:
            curr = str(Path().absolute().parents[i])
        if data_workspace in os.listdir(curr):
            absolute_path = curr + os.sep + data_workspace
            break

    # If the user specifies a sub folder, add it
    if sub_folder_name is not None:
        absolute_path += sub_folder_name

    return absolute_path