The content in the Dataset flolder could be download from 'https://drive.google.com/drive/folders/1nsTQ-3-hvgD_Y6rXdRYJmG4Y9wVrrUM-?usp=sharing'


The trainging and testing code is included in main.py where three parameters are related to the Dataset folder
    parser.add_argument('--data-path', default='Dataset', help='dataset')
    parser.add_argument('--train-idxfile', default='Dataset/devel/IndexFile_tra.hdf5', help='name of train index file')
    parser.add_argument('--val-idxfile', default='Dataset/devel/IndexFile_test.hdf5', help='name of val index file')