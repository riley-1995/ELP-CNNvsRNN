from boxsdk import OAuth2, Client
import os
import re

def traverse_directory(current_location, target_path):
    path_items = target_path.split('/')

    for path_item in path_items:

        folder_items = current_location.get_items()
        for folder_item in folder_items:

            if folder_item.name == path_item:
                current_location = folder_item
                print(f"current location: {current_location}")
    
    return current_location

if __name__ == '__main__':

    auth = OAuth2(
        client_id=os.environ['BOX_DEVELOPER_TOKEN'],
        client_secret=os.environ['BOX_DEVELOPER_TOKEN'],
        access_token=os.environ['BOX_DEVELOPER_TOKEN']
    )

    client = Client(auth)

    user = client.user().get()
    
    target_folder_id = None
    target_folder_shared_weblink = 'https://cornell.box.com/s/sciad6du7n20bx06guolu0qfa5r689kd'
    target_folder = 'Shared Folder - Sounds'

    shared_item = client.get_shared_item(target_folder_shared_weblink)

    target_path = 'Rumble/Training/Sounds'

    if shared_item.type == 'folder':
        print(shared_item)
        print(shared_item.name)

        sounds_folder = traverse_directory(shared_item, target_path)
        sound_files = [i for i in sounds_folder.get_items()]
        print(sound_files[0].content())


        
