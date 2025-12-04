from rclone_python import rclone
import os

# def upload_files(files, folder_id, remote="gdrive:"):
#     links = []
#     for filepath in files:
#         print(f"Uploading {filepath} ...")
#         rclone.copy(
#             filepath, 
#             remote,
#             args = [f"--drive-root-folder-id={folder_id}"]
#         )
        
#         file_remote_path = f"{remote}{os.path.basename(filepath)}"

#         result_link = rclone.link(
#             file_remote_path,
#             args = [f"--drive-root-folder-id={folder_id}"]
#         )

#         links.append(result_link)
#     return links

from rclone_python import rclone
import os

def upload_files(files, dropbox_folder="", remote="dropbox:"):
    links = []

    # Ensure folder ends with /
    if dropbox_folder and not dropbox_folder.endswith("/"):
        dropbox_folder += "/"

    for filepath in files:
        filename = os.path.basename(filepath)
        dropbox_path = f"{remote}{dropbox_folder}{filename}"

        print(f"Uploading {filepath} to {dropbox_path} ...")

        # Upload to Dropbox folder
        rclone.copy(
            filepath,
            f"{remote}{dropbox_folder}"
        )

        # Create shareable link (Dropbox-specific)
        result = rclone.link(
            dropbox_path
            # args=["--dropbox-shared-link"]
        )

        # Change dl=0 to raw=1 front the generated link
        if result:
            result = result.replace("&dl=0", "&raw=1").strip()

        links.append(result)

    return links
