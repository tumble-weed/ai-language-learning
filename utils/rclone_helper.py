from rclone_python import rclone
import os

def upload_files(files, folder_id, remote="gdrive:"):
    links = []
    for filepath in files:
        print(f"Uploading {filepath} ...")
        rclone.copy(
            filepath, 
            remote,
            args = [f"--drive-root-folder-id={folder_id}"]
        )
        
        file_remote_path = f"{remote}{os.path.basename(filepath)}"

        result_link = rclone.link(
            file_remote_path,
            args = [f"--drive-root-folder-id={folder_id}"]
        )

        links.append(result_link)
    return links