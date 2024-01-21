import shutil
import os
import time

def copy_folder(src, dst):
    """
    Copy all contents of the folder at 'src' to the folder at 'dst'.

    :param src: Source folder path.
    :param dst: Destination folder path.
    """
    try:
        # Check if the destination directory exists. If not, create it.
        if not os.path.exists(dst):
            os.makedirs(dst)

        # Copy each item in the source directory to the destination directory
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                new_folder_name = f"{item}_{int(time.time())}"
                d = os.path.join(dst, new_folder_name)
                shutil.copytree(s, d, dirs_exist_ok=True)  # dirs_exist_ok is available from Python 3.8
            else:
                shutil.copy2(s, d)  # Use copy2 to preserve metadata

        print(f"All contents of '{src}' have been copied to '{dst}'")
    except Exception as e:
        print(f"An error occurred: {e}")
