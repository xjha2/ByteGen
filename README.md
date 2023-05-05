## ByteGen: A Code Comment Generation Way without Using Source Code

The whole project of ByteGen





### Environment setup

Install the following libraries in the Python environment

~~~
numpy
tqdm
nltk
prettytable
torch>=1.3.0
~~~





### Dataset

The dataset used by ByteGen has been uploaded to [ByteGenData](https://drive.google.com/drive/folders/1mcHtkgfBtQiqF0F1hlnkU1G5HY2C8bvM?usp=sharing).

You can download the dataset and put it in the `data/java` folder. If you want to train your own dataset, please replace the file in data folder and change the path of data in `main/train_two_encoder.py`. 





### Train

In the stage of train, you need to modify some parameters in `main/train_two_encoder.py`. 

* Change the name of training dataset file.

  ~~~python
  // line 81
  files.add_argument('--train_src', nargs='+', type=str, default='train_story.txt',
                         help='Preprocessed train source file')
  
  // line 83
  files.add_argument('--train_cfg', nargs='+', type=str, default='train_cfg.txt',
                         help='Preprocessed train cfg file')
  
  // line 89
  files.add_argument('--train_tgt', nargs='+', type=str, default='train_summ.txt',
                         help='Preprocessed train target file')
  ~~~

* Set the preprocessed dev file as evaluation dataset

  ~~~python
  // line 91
  files.add_argument('--dev_src', nargs='+', type=str, default='eval_story.txt',
                         help='Preprocessed dev source file')
  
  // line 93
  files.add_argument('--dev_cfg', nargs='+', type=str, default='eval_cfg.txt',
                         help='Preprocessed dev cfg file')
  
  // line 99
  files.add_argument('--dev_tgt', nargs='+', type=str, default='eval_summ.txt',
                         help='Preprocessed dev target file')
  ~~~

* Set only_test as **False**.







### Evaluation

After training, you can set the preprocessed dev file as evaluation dataset to start testing.

~~~python
// line 91
files.add_argument('--dev_src', nargs='+', type=str, default='test_story.txt',
                       help='Preprocessed dev source file')

// line 93
files.add_argument('--dev_cfg', nargs='+', type=str, default='test_cfg.txt',
                       help='Preprocessed dev cfg file')

// line 99
files.add_argument('--dev_tgt', nargs='+', type=str, default='test_summ.txt',
                       help='Preprocessed dev target file')
~~~



Then, set the only-test as **True**

