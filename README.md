1. Create resources folder, and download segmentation model from http://ospm9rsnd.bkt.clouddn.com/model/ltp_data_v3.4.0.zip and unzip it at resources folder.
2. Install required package, 'pip install pyltp'.
3. Download a pre-trained word embedding file from https://pan.baidu.com/s/1qYMEESG and put it at resources folder.
4. Copy trained model from https://pan.baidu.com/s/1hsAJHJm (密码：5byh） to runs folder and unzip it.
5. Go to root of the project folder, run 'export PYTHONPATH=src:$PYTHONPATH'
6. Run 'python src/intent/intent_detector.py "你喜不喜欢我"', you will get predicted label 1 and the time elapsed for this prediction excluding initialization time.

7. To train, copy data file to data/intent/small/intent.train, if you have test file, copy it to data/intent/small/intent.test
8. Run 'python src/utils/segment.py', you will get 2 new files at data/intent/small
9. Run 'python src/intent/train.py'
