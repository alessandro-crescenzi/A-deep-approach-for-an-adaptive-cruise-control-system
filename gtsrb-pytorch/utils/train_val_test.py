import os, random
import zipfile

mapping = {
    "00000": "00000",
    "00001": "00001",
    "00002": "00002",
    "00003": "00003",
    "00004": "00004",
    "00005": "00005",
    "00006": "00006",
    "00007": "00007",
    "00008": "00008",
    "00013": "00009",
    "00014": "00010",
    "00032": "00011"
}


def initialize_data(folder):
    train_zip = folder + '/train_images.zip'
    test_zip = folder + '/test_images.zip'

    # extract train_data.zip to train_data
    train_folder = folder + '/train_images'
    if not os.path.isdir(train_folder):
        print(train_folder + ' not found, extracting ' + train_zip)
        zip_ref = zipfile.ZipFile(train_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()

    # extract test_data.zip to test_data
    test_folder = folder + '/test_images'
    if not os.path.isdir(test_folder):
        print(test_folder + ' not found, extracting ' + test_zip)
        zip_ref = zipfile.ZipFile(test_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()

    # prepend each img filename in train folder to its class e.g. 00000_00001 -> 00003_00000_00001 (for class 3)
    if os.path.isdir(train_folder):
        for dirs in os.listdir(train_folder):
            dir_name = dirs
            dirs = os.path.join(train_folder, dirs)
            if os.path.isdir(dirs):
                for file in os.listdir(dirs):
                    if file.count('_') == 1:
                        os.rename(os.path.join(dirs, file), os.path.join(dirs, dir_name + "_" + file))

    # take 100 images from unknown class and put in folder 00043
    tmp_folder = os.path.join(train_folder + '/00043')
    if not os.path.isdir(tmp_folder) and len(os.listdir(train_folder)) != 13:
        os.mkdir(tmp_folder)
        if os.path.isdir(train_folder):
            for dirs in os.listdir(train_folder):
                dir_name = dirs
                dirs = os.path.join(train_folder, dirs)
                if os.path.isdir(dirs) and dir_name not in mapping.keys():
                    list_files = random.sample([x for x in os.listdir(dirs) if x.startswith("000")], 100)  # how many
                    # images you want to move in unknown class?
                    for file in list_files:
                        os.rename(os.path.join(dirs, file), os.path.join(tmp_folder, file))

    # make validation_data by using 30% images in each class
    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)

        for dirs in os.listdir(train_folder):
            if dirs.startswith('000'):
                os.mkdir(val_folder + '/' + dirs)
                dir_name = dirs
                dirs = os.path.join(train_folder, dirs)
                num_files = len(os.listdir(dirs))
                list_files = random.sample([x for x in os.listdir(dirs) if x.startswith("000")],
                                           int(num_files * 0.3))  # train-val split probability

                for f in list_files:
                    os.rename(dirs + '/' + f, val_folder + '/' + dir_name + '/' + f)
